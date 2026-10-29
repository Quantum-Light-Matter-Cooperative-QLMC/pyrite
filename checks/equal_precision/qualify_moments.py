"""Bounded #788 component qualification and matched moment microbenchmarks.

Run on the remote GPU node with released BremsLib tables. No transport is
performed. All timings are synchronized, warm, and exclude source loading.
"""

import argparse
import hashlib
import json
import platform
import time
from functools import partial
from pathlib import Path

import cupy as cp
import numpy as np

from pyrite.montecarlo._cupy_jit import jit
from pyrite.montecarlo.transport import _jit_radiative_device as exact_device
from pyrite.montecarlo.transport._radiative_moments_device import UploadedRadiativeMoments
from pyrite.montecarlo.transport.hard_radiative import pack_radiative_layer_tables
from pyrite.montecarlo.transport.radiative_moments import (
    MomentPolicy,
    _exact_layer,
    prepare_radiative_moments,
)
from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables


@jit.rawkernel()
def exact_probe(
    nel,
    nt,
    z,
    density,
    incident,
    nominal,
    top,
    chi,
    plasma,
    max_el,
    max_t,
    n_red,
    layer,
    energy,
    cutoff,
    output,
):
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < energy.size:
        output[i, 0] = exact_device._rad_layer_moment(
            np.int32(0),
            layer,
            nel,
            nt,
            z,
            density,
            incident,
            nominal,
            top,
            chi,
            max_el,
            max_t,
            n_red,
            energy[i],
            cutoff,
            plasma,
        )
        output[i, 1] = exact_device._rad_layer_moment(
            np.int32(1),
            layer,
            nel,
            nt,
            z,
            density,
            incident,
            nominal,
            top,
            chi,
            max_el,
            max_t,
            n_red,
            energy[i],
            cutoff,
            plasma,
        )


def measure(call, repeats=5):
    call()
    cp.cuda.Device().synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        cp.cuda.Device().synchronize()
        times.append(time.perf_counter() - start)
    return {
        "seconds": times,
        "median_s": float(np.median(times)),
        "range_s": [min(times), max(times)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--queries", type=int, default=65536)
    args = parser.parse_args()
    protocol = Path(__file__).with_name("moment-policy-788.json")
    tables = load_bremsstrahlung_tables(["Si", "W"])
    if set(tables) != {"Si", "W"}:
        raise ValueError("released Si/W tables required; no fallback")
    compositions = [
        [("Si", 0.049942993373787194)],
        [("W", 0.06322199557658834)],
        [("Si", 0.025), ("W", 0.03)],
    ]
    lo = max(10000.0, *(t.minimum_incident_energy_keV * 1e3 for t in tables.values()))
    hi = min(1e8, *(t.maximum_incident_energy_keV * 1e3 for t in tables.values()))
    queries = np.geomspace(lo, hi, args.queries)
    query_device = cp.asarray(queries)
    record = {
        "schema": "radiative-moment-components-788-v1",
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "sync": {
            key: value.strip()
            for line in Path(".pyrite-sync").read_text().splitlines()
            if (parts := line.partition(":"))[1]
            for key, value in [(parts[0], parts[2])]
            if key
            in {
                "code_digest",
                "code_revision",
                "code_dirty",
                "code_synced_at",
                "code_tables_digest",
            }
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "coverage_eV": [lo, hi],
        "queries": args.queries,
        "threads_per_block": 128,
        "scope": "moment evaluations only; no transport, histories, batching, replay or equal-precision speedup",
        "profiler_limit": "synchronized wall timings; no exclusive FP64/occupancy attribution",
        "cases": [],
    }
    # Warm the CPU reference before recording table preparation cost.
    packed = pack_radiative_layer_tables(compositions, tables, lo, hi, dielectric=True)
    _exact_layer(packed, 0, queries[:2], 1000.0)
    for cutoff in [1000.0, 10000.0]:
        for rtol in [2e-4, 1e-4]:
            policy = MomentPolicy(rtol=rtol)
            start = time.perf_counter()
            prepared = prepare_radiative_moments(
                compositions, tables, lo, hi, cutoff, policy=policy
            )
            build_s = time.perf_counter() - start
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            uploaded = UploadedRadiativeMoments.from_host(prepared)
            cp.cuda.Device().synchronize()
            upload_s = time.perf_counter() - start
            device_packed = tuple(cp.asarray(a.ravel()) for a in packed)
            exact_output = cp.empty((queries.size, 2), dtype=cp.float64)
            for layer in range(len(compositions)):

                def exact_gpu(
                    device_packed=device_packed,
                    layer=layer,
                    cutoff=cutoff,
                    exact_output=exact_output,
                ):
                    exact_probe(
                        ((queries.size + 127) // 128,),
                        (128,),
                        (
                            *device_packed,
                            np.int32(packed[2].shape[1]),
                            np.int32(packed[4].shape[2]),
                            np.int32(packed[5].shape[2]),
                            np.int32(layer),
                            query_device,
                            np.float64(cutoff),
                            exact_output,
                        ),
                    )

                expected = _exact_layer(packed, layer, queries, cutoff)
                actual_cpu = np.array([prepared.evaluate(layer, e) for e in queries])
                actual_gpu = uploaded.evaluate(layer, query_device).get()
                exact_gpu()
                reference_gpu = exact_output.get()
                if not all(
                    np.all(np.isfinite(a))
                    for a in (expected, actual_cpu, actual_gpu, reference_gpu)
                ):
                    raise ValueError("nonfinite moment probe; qualification cannot pass")
                budgets = np.array(policy.atol) + policy.rtol * np.abs(expected)
                component_error = np.max(np.abs(actual_gpu - expected) / budgets, axis=0)
                # Shared table CPU/CUDA parity is separate from interpolation error.
                parity = float(
                    np.max(np.abs(actual_cpu - actual_gpu) / (1e-24 + 2e-12 * np.abs(actual_cpu)))
                )
                exact_parity = float(
                    np.max(
                        np.abs(reference_gpu - expected[:, :2])
                        / (1e-20 + 2e-10 * np.abs(expected[:, :2]))
                    )
                )
                if max(component_error) > 1 or parity > 1 or exact_parity > 1:
                    raise ValueError(
                        f"qualification failed: {component_error=}, {parity=}, {exact_parity=}"
                    )
                case = {
                    "cutoff_eV": cutoff,
                    "rtol": rtol,
                    "layer": layer,
                    "composition": compositions[layer],
                    "digest": prepared.digest,
                    "identity": json.loads(prepared.identity_json),
                    "nodes": prepared.energy_eV[layer].size,
                    "heldout_budget_fraction": prepared.max_error_fraction[layer],
                    "dense_budget_fraction": component_error.tolist(),
                    "same_table_cpu_cuda_budget_fraction": parity,
                    "exact_cpu_cuda_budget_fraction": exact_parity,
                    "build_all_layers_s": build_s,
                    "upload_all_layers_s": upload_s,
                    "host_bytes_all_layers": sum(
                        a.nbytes
                        for a in (*prepared.energy_eV, *prepared.log_energy, *prepared.moments)
                    ),
                    "device_bytes_all_layers": uploaded.energy.nbytes
                    + uploaded.logs.nbytes
                    + uploaded.values.nbytes,
                    "gpu_exact_two_components": measure(exact_gpu),
                    "gpu_table_three_components": measure(
                        partial(uploaded.evaluate, layer, query_device)
                    ),
                    "cpu_exact_three_components": measure(
                        partial(_exact_layer, packed, layer, queries, cutoff)
                    ),
                    "status": "passed",
                }
                record["cases"].append(case)
    record["compiled_exact_resources"] = [
        {
            "registers_per_thread": cp.cuda.driver.funcGetAttribute(4, kernel.ptr),
            "local_bytes_per_thread": cp.cuda.driver.funcGetAttribute(3, kernel.ptr),
        }
        for kernel, _ in exact_probe._cache.values()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
