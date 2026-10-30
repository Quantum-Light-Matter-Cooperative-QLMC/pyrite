"""Refuse benchmark execution without an *enforced* memory cgroup (#807).

A Slurm ``--mem`` *reservation* alone does not protect a shared compute node:
this launcher execs the child only when its current v2 memory cgroup has a
finite memory.max no larger than the requested bound. Inherited containment
also applies to timeout's descendants and CUDA/Python allocations.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def enforced_cgroup_budget(root: Path = Path("/sys/fs/cgroup")) -> tuple[int, int] | None:
    """Return (limit, current) for this process's finite memory cgroup."""
    candidates = [root]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                candidate = root / line.partition("::")[2].lstrip("/")
                if candidate.exists():
                    candidates.insert(0, candidate)
    except OSError:
        pass
    for folder in candidates:
        try:
            limit = (folder / "memory.max").read_text().strip()
            if limit == "max":
                continue
            return int(limit), int((folder / "memory.current").read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def validate_memory_containment(*, max_bytes: int, budget: tuple[int, int] | None) -> None:
    if budget is None:
        raise RuntimeError(
            "no enforced cgroup v2 memory.max; Slurm --mem is only a reservation. "
            "Enable task/cgroup memory enforcement or run samples in an "
            "independently limited cgroup. Refusing unsafe qualification."
        )
    limit, used = budget
    if limit <= 0 or limit > max_bytes:
        raise RuntimeError(
            f"memory cgroup limit {limit} bytes is missing/above the required "
            f"maximum {max_bytes}; refusing benchmark sample"
        )
    if limit - used < min(1 << 30, limit // 10):
        raise RuntimeError(
            f"insufficient memory cgroup headroom before benchmark: "
            f"limit={limit}, current={used} bytes"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.max_bytes <= 0:
        parser.error("supply a positive --max-bytes and a child command after --")
    try:
        validate_memory_containment(max_bytes=args.max_bytes, budget=enforced_cgroup_budget())
    except RuntimeError as error:
        print(f"UNCONTAINED SAMPLE: {error}", file=sys.stderr, flush=True)
        sys.exit(78)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
