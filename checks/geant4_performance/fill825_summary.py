"""Markdown table for a ``fill825.sbatch`` evidence directory (#825/#838).

Usage: ``python3 fill825_summary.py evidence/fill825/<job>``. Standard library only.
Exact-variant rows of one scene must share one raw-score digest (launch geometry and
schedule never change results); a mismatch is flagged in the last column.
"""

import json
import statistics
import sys
from pathlib import Path


def row(sample_path):
    stats_path = sample_path.with_name(sample_path.name.replace(".json", ".stats.json"))
    sample = json.loads(sample_path.read_text())
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    seed = str(sample["seed"])
    per_seed = stats.get("per_seed", {}).get(seed, {})
    grids = [g for g, _ in stats.get("launch_geometry", [])]
    regs = sorted({k["num_regs"] for k in stats.get("kernels", {}).values()})
    occ = sorted(set(stats.get("blocks_per_sm", {}).values()))
    t = sample["timings_s"]
    c = sample["controls"]
    return dict(
        scene=sample["scene"]["id"],
        histories=sample["histories"],
        schedule=c["schedule"],
        scratch=c["scratch_mib"],
        threads=(stats.get("launch_geometry") or [[0, 0]])[-1][1],
        variant=stats.get("variant", "?"),
        warm=t["warm_request"],
        transport=t["warm_transport"],
        kernel=per_seed.get("kernel_s", float("nan")),
        launches=per_seed.get("launches", 0),
        blocks=statistics.median(grids) if grids else 0,
        regs=regs,
        occ=occ,
        rss=sample["peak_rss_kib"] / 1024**2,
        digest=sample["raw_scores_sha256"][:10],
    )


def main():
    root = Path(sys.argv[1])
    samples = sorted(root.glob("*.json"))
    rows = [row(p) for p in samples if not p.name.endswith(".stats.json") and p.name != "sync.json"]
    exact = {}
    for r in rows:
        if r["variant"] == "exact":
            exact.setdefault(r["scene"], set()).add(r["digest"])
    print(
        "| scene | schedule | scratch MiB | thr | variant | warm s | transport s | kernel s "
        "| launches | median blocks | regs | blocks/SM | RSS GiB | digest |"
    )
    print("|" + "---|" * 14)
    for r in sorted(
        rows, key=lambda r: (r["scene"], r["histories"], r["schedule"], r["scratch"], r["threads"])
    ):
        flag = " MISMATCH" if r["variant"] == "exact" and len(exact[r["scene"]]) > 1 else ""
        print(
            f"| {r['scene']} n={r['histories']} | {r['schedule']} | {r['scratch']} | {r['threads']} | {r['variant']} "
            f"| {r['warm']:.2f} | {r['transport']:.2f} | {r['kernel']:.2f} | {r['launches']} "
            f"| {r['blocks']:.0f} | {r['regs']} | {r['occ']} | {r['rss']:.1f} "
            f"| {r['digest']}{flag} |"
        )


if __name__ == "__main__":
    main()
