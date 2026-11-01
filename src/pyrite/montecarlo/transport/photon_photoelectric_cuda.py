"""Explicit-draw CUDA photoabsorption and canonical sampled EADL relaxation."""

from numbers import Integral

import numpy as np

from ...materials.crystal import M_E_EV
from ...materials.photon_differential import subshell_photoionization_cross_sections_ang2
from ._photon_cuda_relaxation import device_relaxation_table, host_relaxation_table
from ._photon_cuda_sampling import validate_event_inputs
from ._photon_photoelectric_cuda_kernel import photoelectric_kernel
from .photon_sampling import MAX_REJECTIONS


def _photoelectric_rates(element, energy, table):
    designators, rates = subshell_photoionization_cross_sections_ang2(element, energy)
    primaries = table["primary_shells"]
    if tuple(designators) != tuple(table["designators"][primaries]):
        raise ValueError("EPDL subshell ordering changed relative to EADL staging")
    accessible = energy[None, :] >= table["binding"][primaries, None]
    rates = np.where(accessible, rates, 0.0).T
    if not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("photoionization rates must be finite and non-negative")
    return np.ascontiguousarray(rates)


def _require_events(cp, status):
    codes = cp.asnumpy(status)
    if np.any(codes == 1):
        raise ValueError("photoelectric uniform sequence exhausted before the event ended")
    if np.any(codes == 2):
        raise RuntimeError("photoelectron angular rejection budget exhausted")
    if np.any(codes == 3):
        raise ValueError("photoionization weights must have a positive finite total")
    if np.any(codes == 4):
        raise BufferError("photoelectric emission capacity exhausted; grow capacity and retry")


def sample_photoelectric_cuda(
    element, energy_eV, direction, uniforms, *, cutoff_eV, capacity=32, require_complete=True
):
    """Subshell absorption, Sauter photoelectron and full sampled EADL tree.

    Host inputs are energies (N,) [eV], unit directions (N,3), uniform prefixes
    (N,L), and the positive relaxation cutoff [eV]. Canonical pinned EPDL/EADL
    bindings must match. Draw order matches ``sample_photoelectric`` followed
    by the queue's isotropic relaxation directions: subshell, angular proposals,
    photoelectron azimuth, every depth-first vacancy decay, then two draws per
    emitted relaxation particle. Terminal holes consume no decay draw.

    Returns device photoelectron energy/direction, primary subshell designator,
    transition IDs/directions, terminal shell indices, counts, draw counts and
    statuses. Transition IDs index the radiative-before-Auger flattened EADL
    arrays in ``_photon_cuda_relaxation.host_relaxation_table``; terminal indices
    index that table's shells, preserving duplicate holes. Only the counted
    prefixes are meaningful. Queue bookkeeping uses those immutable host table
    records with fsum for terminal bindings and signed transition defects.

    Status 0 is complete; 1 needs a longer prefix of the same stream; 2 exhausts
    the reference angular rejection budget; 3 has no finite positive subshell
    support; 4 needs more emission/terminal buffer capacity. Nonzero status
    raises unless ``require_complete=False``. A queue must retry 1/4 without
    discarding prior draws or truncating the cascade. ``capacity`` is workspace,
    not a physical limit. The bounded vacancy stack accepts <=64 ordered shells
    (the canonical Z<=100 data fit), independently of total emitted particles.

    Same assumptions and limiting cases as the CPU owner: isolated neutral
    atomic data, independent vacancies, Sauter K-shell angular approximation
    for every shell, no recoil energy; cutoff holes retain their binding.
    No RNG, photon tracking or charged propagation occurs here.
    Validation: photon-photoelectric-sampling, eadl-sampled-relaxation
    """
    energy, rays, draws = validate_event_inputs(energy_eV, direction, uniforms)
    if isinstance(cutoff_eV, bool) or not np.isfinite(cutoff_eV) or cutoff_eV <= 0:
        raise ValueError("relaxation cutoff must be finite and positive")
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, Integral)
        or not 1 <= capacity < 2**31 - 1
    ):
        raise ValueError("capacity must be a positive int32-sized integer")
    if type(require_complete) is not bool:
        raise TypeError("require_complete must be a bool")
    from ..._backend import select_cupy_backend

    cp = select_cupy_backend().xp
    n, capacity = len(energy), int(capacity)
    electron, primary = cp.empty((n, 4)), cp.empty(n, dtype=cp.int32)
    transitions = cp.empty((n, capacity), dtype=cp.int32)
    terminal = cp.empty((n, capacity + 1), dtype=cp.int32)
    emitted = cp.empty((n, capacity, 3))
    counts, used, status = (
        cp.empty((n, 2), dtype=cp.int32),
        cp.empty(n, dtype=cp.int64),
        cp.empty(n, dtype=cp.int32),
    )
    if n:
        host = host_relaxation_table(element)
        rates = _photoelectric_rates(element, energy, host)
        table = device_relaxation_table(cp, element, cp.cuda.Device().id)
        photoelectric_kernel(cp)(
            ((n + 127) // 128,),
            (128,),
            (
                cp.asarray(energy),
                cp.asarray(rays),
                cp.asarray(draws),
                cp.asarray(rates),
                table["primary_shells"],
                table["designators"],
                table["binding"],
                table["offsets"],
                table["probability"],
                table["first"],
                table["second"],
                np.int64(n),
                np.int32(draws.shape[1]),
                np.int32(rates.shape[1]),
                np.int32(capacity),
                np.int32(MAX_REJECTIONS),
                np.float64(cutoff_eV),
                np.float64(M_E_EV),
                np.float64(np.pi),
                electron,
                primary,
                transitions,
                terminal,
                emitted,
                counts,
                used,
                status,
            ),
        )
        if require_complete:
            _require_events(cp, status)
    return {
        "electron_energy_eV": electron[:, 0],
        "electron_direction": electron[:, 1:4],
        "subshell_designator": primary,
        "transition_ids": transitions,
        "emission_direction": emitted,
        "terminal_shell_indices": terminal,
        "n_transitions": counts[:, 0],
        "n_terminal": counts[:, 1],
        "draws_used": used,
        "status": status,
    }
