"""Count returned transport work without changing the frozen adapter's physics."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np


def main():
    output = Path(sys.argv.pop(1))
    adapter = Path(__file__).with_name("pyrite_adapter.py")
    namespace = runpy.run_path(str(adapter))
    original = namespace["score_segments"]
    blocks = []

    def audit(result, n):
        ids = np.asarray(result["electron_id"])
        events, counts = np.unique(result["event_kind"], return_counts=True)
        rows = np.bincount(ids, minlength=n)
        blocks.append(
            {
                "primaries": n,
                "segments": int(ids.size),
                "segments_per_primary_mean": float(rows.mean()),
                "segments_per_primary_max": int(rows.max()),
                "segment_count_quantiles": np.quantile(rows, [0, 0.5, 0.9, 0.99, 1]).tolist(),
                "event_counts": {str(int(k)): int(v) for k, v in zip(events, counts, strict=True)},
                "mean_material_path_um": float(np.sum(result["L_ang"]) / n / 1e4),
                "returned_array_bytes_sum": sum(
                    v.nbytes for v in result.values() if isinstance(v, np.ndarray)
                ),
                "scalar_counts": {
                    k: int(v) for k, v in result.items() if k.startswith("n_") and np.isscalar(v)
                },
            }
        )
        return original(result, n)

    namespace["main"].__globals__["score_segments"] = audit
    namespace["main"]()
    output.write_text(
        json.dumps(
            {
                "schema": "pyrite.equal-precision.work-audit.v1",
                "interpretation": "First complete request is discarded warmup; second is measured. Returned segments exclude capacity replays. Instrumented run is diagnostic only.",
                "command": sys.argv,
                "blocks": blocks,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
