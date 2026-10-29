"""Amplification probe for #792: marginal cost of exact ELSEPA flight rates.

An isolated copy of the CPU or CUDA transport core recomputes the flight-start
per-element ELSEPA rate sum ``K`` extra times with slightly perturbed energies
(so the compiler cannot reuse the first evaluation). The result only feeds a
branch that is never taken (rates are positive), so trajectories stay
bit-identical to ``K = 0``; the probe refuses any changed trajectory digest.
The timing slope over ``K`` is the cost of one flight-start rate sum: the most
a layer-total LUT could save at that call site. The multi-element selection
recomputation (the only other rate site) runs at most once per elastic event
and is bounded separately by the same per-sum cost. No production module
imports this probe.
"""

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

PACKAGE = "pyrite.montecarlo.transport"
CPU_ANCHOR = "            lam_ang = 1e8 / total_rate\n"
CPU_PROBE = """\
            if elastic_model_code == 2 and not lut:
                for amp_k in range({k}):
                    probe_rate = 0.0
                    probe_E = E_j * (1.0 + (amp_k + 1) * 1e-13)
                    for probe_el in range(n_el):
                        probe_rate += _elsepa_rate_scalar(
                            probe_E,
                            el_logE,
                            el_log_rate,
                            el_start[L, probe_el],
                            el_len[L, probe_el],
                        )
                    if probe_rate < 0.0:
                        total_rate += 1.0
"""
CUDA_ANCHOR = "        mean_free_path_ang = np.float64(1e8) / total_rate\n"
CUDA_PROBE = """\
        if elastic_model_code == I32_TWO:
            amp_k = I32_ZERO
            while amp_k < np.int32({k}):
                amp_k += I32_ONE
                probe_rate = F64_ZERO
                probe_log = log_E_eV + np.float64(1e-13) * amp_k
                probe_el = I32_ZERO
                while probe_el < n_elements:
                    probe_row = el_start[element_row_offset + probe_el]
                    probe_count = el_len[element_row_offset + probe_el]
                    probe_lo = _log_grid_lower(el_logE, probe_row, probe_count, probe_log)
                    probe_f = _log_grid_fraction(
                        el_logE, probe_row, probe_count, probe_log, probe_lo
                    )
                    probe_low = el_log_rate[probe_row + probe_lo]
                    probe_rate += xp.exp(
                        probe_low
                        + probe_f * (el_log_rate[probe_row + probe_lo + I32_ONE] - probe_low)
                    )
                    probe_el += I32_ONE
                if probe_rate < F64_ZERO:
                    total_rate += F64_ONE
"""


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.SourceLoader):
    """Serve one transformed transport module from a recorded file."""

    def __init__(self, name, path):
        self.name, self.path = name, path

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        return importlib.util.spec_from_file_location(fullname, self.path, loader=self)

    def get_filename(self, fullname):
        return str(self.path)

    def get_data(self, path):
        return Path(path).read_bytes()


def install(core, k, audit):
    """Transform the core source before the transport package imports it."""
    if any(name.startswith(PACKAGE) for name in sys.modules):
        raise RuntimeError("install the probe before importing transport")
    spec = importlib.util.find_spec("pyrite.montecarlo")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("pyrite.montecarlo is not importable")
    root = Path(spec.submodule_search_locations[0]) / "transport"
    module, anchor, probe = (
        ("cores", CPU_ANCHOR, CPU_PROBE)
        if core == "per-electron"
        else ("_jit_kernel", CUDA_ANCHOR, CUDA_PROBE)
    )
    original = (root / f"{module}.py").read_text()
    if original.count(anchor) != 1:
        raise ValueError(f"{module}: flight-rate anchor inventory changed")
    source = original if k == 0 else original.replace(anchor, probe.format(k=k) + anchor)
    destination = audit.with_suffix(f".{module}.py")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)
    sys.meta_path.insert(0, _Finder(f"{PACKAGE}.{module}", destination))
    return {
        "module": module,
        "amplification": k,
        "reference_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("amplification", type=int)
    parser.add_argument("audit", type=Path)
    args, adapter_args = parser.parse_known_args()
    core = adapter_args[adapter_args.index("--core") + 1]
    if args.amplification < 0:
        parser.error("amplification must be nonnegative")
    # Separate Numba caches per transformed source keep timings honest.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(args.audit.parent / "numba-cache"))
    source = install(core, args.amplification, args.audit)
    namespace = runpy.run_path(str(Path(__file__).with_name("pyrite_adapter.py")))
    digests = []

    def hashed(original):
        def score(result, count):
            import numpy as np

            digest = hashlib.sha256()
            for key, array in sorted(result.items()):
                if isinstance(array, np.ndarray):
                    digest.update(key.encode())
                    digest.update(array.tobytes())
            digests.append(
                {
                    "primaries": count,
                    "segments": int(np.asarray(result["L_ang"]).size),
                    "arrays_sha256": digest.hexdigest(),
                }
            )
            return original(result, count)

        return score

    scope = namespace["main"].__globals__
    for name in ("score_segments", "transport_scores"):
        scope[name] = hashed(scope[name])
    sys.argv = [sys.argv[0], *adapter_args]
    namespace["main"]()
    sample_path = Path(adapter_args[adapter_args.index("--output") + 1])
    sample = json.loads(sample_path.read_text())
    sample.update(
        schema="elsepa-rate-amplification-probe-v1",
        probe=source,
        trajectory_digests=digests,
    )
    sample_path.write_text(json.dumps(sample, indent=2) + "\n")
    args.audit.write_text(json.dumps({"probe": source, "blocks": digests}, indent=2) + "\n")


if __name__ == "__main__":
    main()
