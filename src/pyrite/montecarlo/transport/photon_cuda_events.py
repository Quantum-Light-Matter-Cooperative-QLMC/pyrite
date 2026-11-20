"""Device event waves with host lineage/RNG and exact reference bookkeeping.

Every flight and conditional event is sampled on CUDA; host orchestration owns
the existing lineage keys, Philox streams and fsum ledgers. Prefix/capacity
retries regenerate the same stream prefix from its saved post-flight state.
No CPU sampler fallback, rejected-history loss or physical cascade truncation.
Validation: photon-history-queue, photon-charged-coupling
"""

from collections import defaultdict
from dataclasses import dataclass
from math import fsum

import numpy as np

from ..._backend import BackendResourceError
from ...materials.photon_cross_sections import _atomic_number
from .._resources import admitted_chunk, resolve_resource_policy
from ._photon_cuda_relaxation import host_relaxation_table
from .pair_production import PAIR_THRESHOLD_EV
from .photon_cuda import scene_first_interactions_cuda
from .photon_navigation import PHOTON_CHANNEL_ORDER
from .photon_pair_cuda import sample_pairs_cuda
from .photon_photoelectric_cuda import sample_photoelectric_cuda, sample_vacancy_relaxation_cuda
from .photon_scattering_cuda import sample_photon_scattering_cuda

_INITIAL_DRAW_WIDTH = 32
_INITIAL_CAPACITY = 32


@dataclass(frozen=True)
class CudaCollision:
    photons: tuple = ()  # (daughter ordinal, energy eV, unit direction)
    charged: tuple = ()  # (ordinal, species, producing channel, kinetic eV, direction)
    terminal_binding_eV: float = 0.0
    transition_defect_eV: float = 0.0
    positron_rest_eV: float = 0.0


def _admit(backend, requested, bytes_per_row):
    policy = resolve_resource_policy(backend)
    free = int(backend.xp.cuda.runtime.memGetInfo()[0])
    budget = min(64 << 20, free // 2, (policy.device_budget_bytes or free) // 4)
    return admitted_chunk(
        requested_chunk=requested,
        bins=bytes_per_row,
        itemsize=1,
        intermediates=2,
        budget_bytes=budget,
        minimum_chunk=1,
    )


def cuda_first_flights(
    origins, directions, energies, uniforms, layers, boxes, *, batch_size, **footprint
):
    """Bound navigation storage; retain input order and the original two draws."""
    from ..._backend import select_cupy_backend

    backend = select_cupy_backend()
    n = len(energies)
    output = {"distance_ang": np.full(n, np.inf)}
    output.update({name: np.full(n, -1, np.int64) for name in ("layer", "channel", "element")})
    compositions = [layer[2] for layer in layers] + [box.composition for box in boxes]
    if not n:
        return output
    row_bytes = 256 + 16 * len(compositions) + 80 * sum(len(c) for c in compositions)
    chunk = _admit(backend, batch_size, row_bytes)
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        hit = scene_first_interactions_cuda(
            origins[sl], directions[sl], energies[sl], uniforms[sl], layers, boxes, **footprint
        )
        for name, value in hit.items():
            output[name][sl] = backend.xp.asnumpy(value)
    return output


def _prefix(state, width):
    """Restore the complete Philox state, including its partly consumed buffer."""
    bitgen = np.random.Philox(0)
    bitgen.state = state
    return np.random.Generator(bitgen).random(width)


def _sample(element, channel, energy, rays, uniforms, cutoff, capacity):
    options = {"require_complete": False}
    name = PHOTON_CHANNEL_ORDER[channel]
    if name == "photoelectric":
        return sample_photoelectric_cuda(
            element, energy, rays, uniforms, cutoff_eV=cutoff, capacity=capacity, **options
        )
    if name == "incoherent":
        return _compton_with_relaxation(element, energy, rays, uniforms, cutoff, capacity)
    if name == "coherent":
        return sample_photon_scattering_cuda(
            element, energy, rays, uniforms, channel=name, **options
        )
    return sample_pairs_cuda(
        energy, rays, uniforms, atomic_number=_atomic_number(element), **options
    )


def _compton_with_relaxation(element, energy, rays, uniforms, cutoff, capacity):
    """Compton event, then its vacancy cascade on the same draw row (CPU order).

    A complete Compton row continues at its ``draws_used``; the cascade's
    status (0, 1 longer prefix, 4 capacity) and total draw count replace the
    event's. Incomplete Compton rows keep their status and skip the cascade.
    Validation: photon-compton-scattering, eadl-sampled-relaxation
    """
    from ..._backend import select_cupy_backend

    cp = select_cupy_backend().xp
    event = sample_photon_scattering_cuda(
        element, energy, rays, uniforms, channel="incoherent", require_complete=False
    )
    status = cp.asnumpy(event["status"])
    designators = np.where(status == 0, cp.asnumpy(event["subshell_designator"]), 0.0)
    relaxed = sample_vacancy_relaxation_cuda(
        element,
        designators.astype(np.int64),
        cp.asnumpy(event["draws_used"]),
        uniforms,
        cutoff_eV=cutoff,
        capacity=capacity,
    )
    complete = cp.asarray(status == 0)
    event.update(relaxed)
    event["status"] = cp.where(complete, relaxed["status"], cp.asarray(status))
    return event


def _relaxation_products(row, arrays, table, first):
    """Relaxation photons/electrons from ordinal ``first``, terminal binding, defect."""
    from .photon_histories import ELECTRON, RELAXATION_CHANNEL

    nt, nh = int(arrays["n_transitions"][row]), int(arrays["n_terminal"][row])
    ids = arrays["transition_ids"][row, :nt]
    holes = arrays["terminal_shell_indices"][row, :nh]
    photons, charged, defects = [], [], []
    for ordinal, tid in enumerate(ids, start=first):
        energy = float(table["energy"][tid])
        direction = arrays["emission_direction"][row, ordinal - first].copy()
        if table["kind"][tid] == 0:
            photons.append((ordinal, energy, direction))
        else:
            charged.append((ordinal, ELECTRON, RELAXATION_CHANNEL, energy, direction))
        defects.append(
            table["binding"][table["parent"][tid]]
            - fsum(
                table["binding"][d] for d in (table["first"][tid], table["second"][tid]) if d >= 0
            )
            - energy
        )
    return photons, charged, fsum(table["binding"][i] for i in holes), fsum(defects)


def _photoelectric_product(row, arrays, element, channel):
    from .photon_histories import ELECTRON

    photons, charged, binding, defect = _relaxation_products(
        row, arrays, host_relaxation_table(element), 1
    )
    electron = (
        0,
        ELECTRON,
        channel,
        float(arrays["electron_energy_eV"][row]),
        arrays["electron_direction"][row].copy(),
    )
    return CudaCollision(tuple(photons), (electron, *charged), binding, defect)


def _compton_product(row, arrays, element, channel):
    """Photon 0, recoil 1, then vacancy relaxation from ordinal 2 (CPU twin)."""
    from .photon_histories import ELECTRON

    photons, charged, binding, defect = _relaxation_products(
        row, arrays, host_relaxation_table(element), 2
    )
    photon = (0, float(arrays["photon_energy_eV"][row]), arrays["photon_direction"][row].copy())
    electron = (
        1,
        ELECTRON,
        channel,
        float(arrays["electron_energy_eV"][row]),
        arrays["electron_direction"][row].copy(),
    )
    return CudaCollision((photon, *photons), (electron, *charged), binding, defect)


def _product(row, arrays, element, channel):
    from .photon_histories import ELECTRON, POSITRON

    name = PHOTON_CHANNEL_ORDER[channel]
    if name == "photoelectric":
        return _photoelectric_product(row, arrays, element, channel)
    if name == "incoherent":
        return _compton_product(row, arrays, element, channel)
    if name == "coherent":
        photon = (0, float(arrays["photon_energy_eV"][row]), arrays["photon_direction"][row].copy())
        return CudaCollision((photon,))
    charged = tuple(
        (
            ordinal,
            species,
            channel,
            float(arrays[f"{label}_energy_eV"][row]),
            arrays[f"{label}_direction"][row].copy(),
        )
        for ordinal, species, label in ((0, ELECTRON, "electron"), (1, POSITRON, "positron"))
    )
    return CudaCollision(charged=charged, positron_rest_eV=PAIR_THRESHOLD_EV)


def _retry_group(backend, indices, active, rngs, element, channel, cutoff, batch_size):
    """Only unfinished events are retried; prefix buffers never accumulate."""
    states = {i: rngs[i].bit_generator.state for i in indices}
    jobs, output = [(np.asarray(indices), _INITIAL_DRAW_WIDTH, _INITIAL_CAPACITY)], {}
    while jobs:
        owners, width, capacity = jobs.pop()
        if width >= 2**31 or capacity >= 2**31 - 1:
            raise BackendResourceError("CUDA photon event workspace exceeds int32 indexing")
        row_bytes = 2048 + 8 * width + 64 * capacity
        chunk = _admit(backend, min(batch_size, 4096), row_bytes)
        for start in range(0, len(owners), chunk):
            part = owners[start : start + chunk]
            energy = np.array([active[i].energy for i in part])
            rays = np.array([active[i].direction for i in part])
            initial = width == _INITIAL_DRAW_WIDTH and capacity == _INITIAL_CAPACITY
            uniforms = np.array(
                [rngs[i].random(width) if initial else _prefix(states[i], width) for i in part]
            )
            result = _sample(element, channel, energy, rays, uniforms, cutoff, capacity)
            arrays = {name: backend.xp.asnumpy(values) for name, values in result.items()}
            codes = arrays["status"]
            if np.any(codes == 3):
                raise ValueError("CUDA photon conditional event has no positive finite support")
            if np.any(codes == 2):
                raise RuntimeError("CUDA photon conditional rejection budget exhausted")
            if np.any((codes < 0) | (codes > 4)):
                raise RuntimeError("invalid CUDA photon conditional event status")
            for row in np.flatnonzero(codes == 0):
                output[int(part[row])] = _product(row, arrays, element, channel)
            if np.any(codes == 1):
                jobs.append((part[codes == 1], width * 2, capacity))
            if np.any(codes == 4):
                jobs.append((part[codes == 4], width, capacity * 2))
    return output


def cuda_collision_products(active, flight, collided, compositions, rngs, *, cutoff_eV, batch_size):
    """Group independent collisions by element/channel while preserving labels."""
    from ..._backend import select_cupy_backend

    backend = select_cupy_backend()
    groups = defaultdict(list)
    for i in np.flatnonzero(collided):
        region, atom = int(flight["layer"][i]), int(flight["element"][i])
        groups[(compositions[region][atom][0], int(flight["channel"][i]))].append(int(i))
    products = {}
    for (element, channel), indices in groups.items():
        products.update(
            _retry_group(backend, indices, active, rngs, element, channel, cutoff_eV, batch_size)
        )
    return products


def apply_cuda_collision(queue, parent, position, product):
    """Use the same lineage, cutoffs and ledger owners as the CPU queue."""
    for ordinal, energy, direction in product.photons:
        queue.photon(parent, ordinal, position, direction, energy)
    if product.positron_rest_eV:
        queue.book("positron_rest_eV", parent.row, parent.weight * product.positron_rest_eV)
    for ordinal, species, channel, energy, direction in product.charged:
        queue.particle(parent, ordinal, species, channel, position, direction, energy)
    if product.terminal_binding_eV:
        queue.book("terminal_binding_eV", parent.row, parent.weight * product.terminal_binding_eV)
    if product.transition_defect_eV:
        queue.book("transition_defect_eV", parent.row, parent.weight * product.transition_defect_eV)
