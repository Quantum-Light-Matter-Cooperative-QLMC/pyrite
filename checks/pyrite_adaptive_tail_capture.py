"""Capture the actual CXR stopping scores for one catalog case; replay VOV policies.

Run from the PyRITE repository root, in the production backend/environment.
This diagnostic runs the stopping transport only; it does not score spectra,
write simulation checkpoints or change the profile. A capture is empirical
finite-prefix evidence, not a population confidence guarantee.
"""

import argparse
import json
import os
import runpy
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.1, 1.0)


def diagnostics(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2 or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("expected at least two finite nonnegative scores")
    mean = x.mean()
    if mean <= 0:
        raise ValueError("zero mean")
    d = (x / mean - 1.0) ** 2
    m2 = d.sum()
    m4 = np.dot(d, d)
    top = np.argpartition(d, -min(10, x.size))[-min(10, x.size) :]
    top = top[np.argsort(d[top])[::-1]]
    return {
        "n": int(x.size),
        "mean": float(mean),
        "rse": float(np.sqrt(m2 / (x.size * (x.size - 1)))),
        "vov_gate": float(m4 / m2**2) if m2 else 0.0,
        "top": [
            {
                "history": int(i),
                "score_over_mean": float(x[i] / mean),
                "variance_share": float(d[i] / m2) if m2 else 0.0,
                "fourth_moment_share": float(d[i] ** 2 / m4) if m4 else 0.0,
            }
            for i in top
        ],
    }


def replay(values, precision, brem_count):
    from pyrite.montecarlo.runner.adaptive import run_stopping_rule

    n = len(values)
    # Isolate the CXR guard. The actual independent brem stop can only delay
    # a CXR decision. A scalar stop preceding it is not a production stop.
    base = replace(
        precision,
        observables=("cxr",),
        brem_target_rse=None,
        brem_min_histories=None,
        brem_max_histories=None,
        max_histories=n,
    )
    ref = diagnostics(values)
    results = []
    for threshold in THRESHOLDS:
        p = replace(base, max_variance_of_variance=threshold)
        record = run_stopping_rule(p, lambda a, b, _: {"cxr": values[a:b]})
        count = int(record["realized_electrons"])
        stat = record["statistics"]["cxr"]
        results.append(
            {
                "vov_threshold": threshold,
                "scalar_stop": count,
                "before_brem_freeze": bool(count < brem_count),
                "stop_reason": record["stop_reason"],
                "at_capture_end": count == n,
                "mean_error_vs_capture": float(values[:count].mean() / ref["mean"] - 1),
                "reported_rse": stat["relative_se"],
                "vov": stat["variance_of_variance"],
                "fixed_capture_cv_rse_proxy": ref["rse"] * np.sqrt(n / count),
                "passes": stat["passes"],
            }
        )
    return {
        "finite_capture": ref,
        "scalar_cxr_replays": results,
        "note": "Finite capture comparison only. Capture-end convergence may be an added decision point; no inference beyond the observed prefix. Scalar CXR stops before brem freeze are not actual joint stops.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="cleo_fiber")
    parser.add_argument("--material", default="hopg")
    parser.add_argument("--energy", type=float, default=300)
    parser.add_argument("--tilt", type=float, default=50)
    parser.add_argument("--output", type=Path, default=Path("adaptive-tail-50deg"))
    args = parser.parse_args()
    from pyrite._backend import _to_cpu
    from pyrite.montecarlo.runner import adaptive, independent

    os.environ["PROFILE"] = args.profile
    helper = runpy.run_path("checks/adaptive_block_size_1252.py")
    case, precision, _ = helper["profile_case"](args.material, args.energy, args.tilt, "auto")
    if precision.batch_means_band_eV is not None:
        raise ValueError("measure-only route does not support batch means")
    if "cxr" not in precision.observables:
        raise ValueError("CXR monitoring required")
    prefix = args.output
    prefix.parent.mkdir(parents=True, exist_ok=True)
    raw = Path(str(prefix) + ".f64")
    meta_path = Path(str(prefix) + ".json")
    scores_path = Path(str(prefix) + ".npz")
    if any(p.exists() for p in (raw, meta_path, scores_path)):
        raise FileExistsError(f"choose an unused output prefix: {prefix}")
    meta = {
        "profile": args.profile,
        "material": args.material,
        "energy_keV": args.energy,
        "tilt_deg": args.tilt,
        "precision": precision.to_dict(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    original = independent.case_measure
    high = 0
    with raw.open("xb") as stream:

        def factory(case, settings):
            measure = original(case, settings)

            def capture(start, stop, block):
                nonlocal high
                result = measure(start, stop, block)
                if "cxr" in result:
                    values = np.asarray(_to_cpu(result["cxr"]), dtype="<f8")
                    if values.shape != (stop - start,):
                        raise ValueError("unexpected score shape")
                    if start == 0:
                        high = 0
                        stream.seek(0)
                        stream.truncate()
                    if start > high:
                        raise RuntimeError("gap in captured history indices")
                    # A transport retry or brem cutoff transition may overwrite
                    # a suffix. Indexed writes preserve the final prefix.
                    stream.seek(8 * start)
                    stream.write(values.tobytes())
                    stream.flush()
                    high = stop
                    stream.truncate(8 * high)
                    print(json.dumps({"capture_start": start, "capture_stop": stop}), flush=True)
                return result

            return capture

        independent.case_measure = factory
        try:
            transport = adaptive.transport_case_adaptive(
                case,
                precision,
                keep_segments_on_device=True,
                block_consumer=lambda _: None,
            )
        finally:
            independent.case_measure = original
    stats = transport["adaptive_sampling"]
    n = int(stats["realized_electrons"])
    nb = int(stats["realized_electrons_brem"])
    if high < n:
        raise RuntimeError(f"captured only {high} scores for {n} realized histories")
    values = np.fromfile(raw, dtype="<f8", count=n)
    if values.size != n:
        raise RuntimeError("incomplete score capture")
    diag = diagnostics(values)
    expected = stats["statistics"]["cxr"]
    np.testing.assert_allclose(diag["mean"], expected["mean"], rtol=1e-10, atol=0)
    np.testing.assert_allclose(diag["rse"], expected["relative_se"], rtol=1e-8, atol=1e-15)
    np.savez_compressed(scores_path, cxr=values)
    meta.update({"adaptive_sampling": stats, "replay": replay(values, precision, nb)})
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["replay"], indent=2), flush=True)
    print(f"Saved {meta_path} and {scores_path}; raw capture retained at {raw}")


if __name__ == "__main__":
    main()
