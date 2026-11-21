"""Audit/replay source-cone quadrature on identical captured incident primaries.

Never launches transport. Actual replay is explicit compute; --audit-only checks
retention without importing the scorer, loading tables or accessing a GPU.
No timing ratio, confidence gate or numerical qualification is produced.
"""

import argparse
import json
import runpy
from pathlib import Path

import numpy as np

from pyrite._util.hashing import file_sha256
from pyrite.perf._track_capture import load_index, read_shard, source_fingerprint

HERE = Path(__file__).resolve().parent


def rules():
    """One-factor detector ladders, the point candidates and the integrated corner."""
    values = {(a, 32) for a in (4, 8, 16, 32)} | {(32, e) for e in (4, 8, 16, 32)}
    values.add((4, 4))
    return [
        dict(id=f"integrated-a{a}-e{e}", mode="fixed", order=a, energy_order=e)
        for a, e in sorted(values)
    ] + [dict(id=f"point-e{e}", mode="point", order=8, energy_order=e) for e in (8, 32)]


def contrasts():
    result = []
    for control in ("angular", "energy"):
        for label, left, right in (("nominal-L3", 4, 32), ("L1-L2", 8, 16), ("L2-L3", 16, 32)):
            names = [
                f"integrated-a{q}-e32" if control == "angular" else f"integrated-a32-e{q}"
                for q in (left, right)
            ]
            result.append(dict(id=f"{control}-{label}", left=names[0], right=names[1]))
    result.extend(
        dict(id=f"point-e{q}-minus-integrated-L3", left=f"point-e{q}", right="integrated-a32-e32")
        for q in (8, 32)
    )
    result.append(dict(id="detector-corner", left="integrated-a4-e4", right="integrated-a32-e32"))
    return result


def audit(root, max_bytes):
    index = load_index(root, max_bytes)
    for shard in index["shards"]:
        read_shard(root, shard)
    return index


def replay(root, index, tables, scorer, chunk, max_score_bytes):
    """Retain per-primary scores/differences; never infer zero-variance convergence."""
    n = index["histories"]
    planned = rules()
    slots = len(planned) + len(contrasts()) + 1
    if chunk <= 0 or n * 8 * slots > max_score_bytes:
        raise ValueError("replay score storage exceeds byte ceiling, or invalid scoring chunk")
    metadata = index["metadata"]
    context = metadata["context"] | dict(brem=tables, chunk=chunk, orders=[])
    values = {rule["id"]: np.empty(n, dtype=np.float64) for rule in planned}
    for shard in index["shards"]:
        segments, ids = read_shard(root, shard)
        for rule in planned:
            scores = scorer(segments, shard["count"], metadata["cone"], context | rule)
            score = np.asarray(scores["cone_source_photons_ge10kev"])
            if score.shape != (ids.size,) or not np.all(np.isfinite(score)) or np.any(score < 0):
                raise ValueError("replay scorer lost incident primaries or produced invalid scores")
            values[rule["id"]][ids] = score
    differences = {
        f"difference-{contrast['id']}": values[contrast["left"]] - values[contrast["right"]]
        for contrast in contrasts()
    }
    return dict(primary_ids=np.arange(n, dtype=np.int64), **values, **differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new output JSON; replay also writes an adjacent score NPZ",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--max-capture-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--max-score-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--chunk", type=int, default=20000)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        parser.error("output JSON/NPZ must be new")
    index = audit(args.capture, args.max_capture_bytes)
    result = dict(
        schema="pyrite.geant4-performance.cone-replay.v1",
        purpose="accuracy-only",
        capture_sha256=file_sha256(args.capture / "capture.json"),
        capture_manifest_sha256=index["manifest_sha256"],
        capture_metadata=index["metadata"],
        replay_script_sha256=file_sha256(Path(__file__)),
        histories=index["histories"],
        numerical_gate="unmeasured",
        timing_comparison_eligible=False,
        pairing="identical saved rows and explicit incident-primary IDs; not transport CRN",
        audit_only=args.audit_only,
        scoring_chunk=args.chunk,
    )
    if not args.audit_only:
        metadata = index["metadata"]
        if metadata["pyrite_python_source_sha256"] != source_fingerprint() or metadata[
            "harness_sha256"
        ] != file_sha256(HERE / "pyrite_bench.py"):
            raise ValueError(
                "source differs from captured scorer/transport; use the matching checkout"
            )
        from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

        context = metadata["context"]
        tables = load_bremsstrahlung_tables([name for name, _ in context["composition"]])
        # The coupled scorer also audits the saved partition for every shard.
        for shard in index["shards"]:
            identity = sorted((el, t.atomic_number, t.key, t.digest) for el, t in tables.items())
            if identity != sorted(
                tuple(v) for v in shard["scalars"]["radiative"]["bremslib_tables"]
            ):
                raise ValueError("replay BremsLib tables differ from captured transport")
        scorer = runpy.run_path(str(HERE / "pyrite_bench.py"))["continuous_cone_scores"]
        arrays = replay(args.capture, index, tables, scorer, args.chunk, args.max_score_bytes)
        from pyrite._backend import BACKEND, REAL
        from pyrite.perf.precision import primary_moments

        args.output.parent.mkdir(parents=True, exist_ok=True)
        raw = args.output.with_suffix(".npz")
        with raw.open("xb") as stream:
            np.savez(stream, **arrays)
        result.update(
            rules=rules(),
            contrasts=contrasts(),
            backend=BACKEND.name,
            real_dtype=str(np.dtype(REAL)),
            raw_scores_sha256=file_sha256(raw),
            observables={
                key: primary_moments(value) for key, value in arrays.items() if key != "primary_ids"
            },
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, allow_nan=False, indent=1)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                histories=index["histories"],
                audit_only=args.audit_only,
                numerical_gate="unmeasured",
            )
        )
    )


if __name__ == "__main__":
    main()
