"""Dependency-free primary test inventory and GitHub Actions matrix policy.

Executable directly on the CI runner before installing any environment.
"""

import json
import os
from pathlib import Path

PRIMARY_TEST_SUITE_NAMES = (
    "core",
    "packaging",
    "apps",
    "cli",
    "quantum",
    "channeling",
    "spectrum-coherent",
    "spectrum-brem",
    "spectrum",
    "transport-cascades",
    "transport-collisions",
    "transport-runner",
    "transport",
    "energy-grid",
    "materials",
    "xsgen",
)

CI_ENVIRONMENTS = ("default", "py314")
# Keep measured JIT-disabled shard budgets; unassigned new suites go to data.
ONLINE_SHARDS = {
    "collisions": ("transport-collisions",),
    "transport": ("transport",),
    "runner": ("transport-runner",),
    "spectrum": (
        "transport-cascades",
        "spectrum",
        "spectrum-coherent",
        "spectrum-brem",
        "channeling",
        "quantum",
    ),
    "core": ("core",),
}


def ci_matrices() -> dict:
    """Return compiled and JIT-disabled matrices from the same suite inventory."""
    assigned = [suite for suites in ONLINE_SHARDS.values() for suite in suites]
    if len(assigned) != len(set(assigned)) or not set(assigned) <= set(PRIMARY_TEST_SUITE_NAMES):
        raise ValueError("invalid or duplicate online shard suite")
    if len(PRIMARY_TEST_SUITE_NAMES) != len(set(PRIMARY_TEST_SUITE_NAMES)):
        raise ValueError("duplicate primary suite")
    remainder = [suite for suite in PRIMARY_TEST_SUITE_NAMES if suite not in assigned]
    shards = [
        {"shard": shard, "suites": " ".join(suites)} for shard, suites in ONLINE_SHARDS.items()
    ]
    if remainder:
        shards.append({"shard": "data", "suites": " ".join(remainder)})
    return {
        "focused": {
            "suite": [suite for suite in PRIMARY_TEST_SUITE_NAMES if suite != "quantum"],
            "environment": list(CI_ENVIRONMENTS),
        },
        "quantum": {"environment": list(CI_ENVIRONMENTS)},
        "online": {"include": shards},
    }


def main() -> None:
    matrices = ci_matrices()
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.writelines(
                f"{name}={json.dumps(matrix, separators=(',', ':'))}\n"
                for name, matrix in matrices.items()
            )
    else:
        print(json.dumps(matrices, indent=2))


if __name__ == "__main__":
    main()
