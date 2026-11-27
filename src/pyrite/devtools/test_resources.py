"""Adaptive pytest worker allocation and native-thread limits."""

import os
import sys
from pathlib import Path

from pyrite.devtools import repo_root

ROOT = repo_root()

# Measured per-worker peak RSS over the full suite is ~1.2-1.8 GiB; budget 2 GiB
# so a laptop with other work running does not swap or OOM.
TEST_WORKER_MEMORY_BYTES = 2 * 1024**3
MAX_TEST_WORKERS = 6


def default_test_workers() -> int:
    """xdist worker count for an untargeted run: ``PYRITE_TEST_WORKERS`` or a
    CPU- and available-memory-bounded default (1 means serial)."""
    override = os.environ.get("PYRITE_TEST_WORKERS")
    if override:
        return max(1, int(override))
    import psutil

    by_memory = psutil.virtual_memory().available // TEST_WORKER_MEMORY_BYTES
    return max(1, min(os.cpu_count() or 1, by_memory, MAX_TEST_WORKERS))


def _selects_test_paths(pytest_args: list[str]) -> bool:
    return any(
        not arg.startswith("-") and ("::" in arg or (ROOT / arg).exists() or Path(arg).exists())
        for arg in pytest_args
    )


def parallel_pytest_args(pytest_args: list[str]) -> list[str]:
    """Prepend xdist workers unless the caller supplies a targeted selection.

    Call with forwarded arguments only; ``test-suite`` adds its owned paths
    afterwards. Runs that name test paths, choose ``-n``/``--numprocesses``,
    or disable xdist are forwarded unchanged.
    """
    explicit = any(
        arg in {"-n", "--numprocesses", "no:xdist"}
        or arg.startswith(("-n=", "--numprocesses="))
        or (arg.startswith("-n") and arg[2:].isdigit())
        for arg in pytest_args
    )
    if explicit or _selects_test_paths(pytest_args):
        return pytest_args
    workers = default_test_workers()
    if workers < 2:
        return pytest_args
    print(
        f"pyrite-dev: {workers} xdist workers (PYRITE_TEST_WORKERS=1 for serial)",
        file=sys.stderr,
    )
    return ["-n", str(workers), "--dist", "worksteal", *pytest_args]


#: Native thread pools capped per xdist worker (#1021). numpy and scipy each
#: load their own OpenBLAS, so N workers x default pools oversubscribe cores.
BLAS_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _xdist_workers(pytest_args: tuple[str, ...]) -> int:
    """Worker count requested by ``-n``/``--numprocesses`` (``auto`` = CPUs)."""
    for i, arg in enumerate(pytest_args):
        if arg in {"-n", "--numprocesses"} and i + 1 < len(pytest_args):
            value = pytest_args[i + 1]
        elif arg.startswith(("-n=", "--numprocesses=")):
            value = arg.split("=", 1)[1]
        elif arg.startswith("-n") and arg[2:].isdigit():
            value = arg[2:]
        else:
            continue
        if value in {"auto", "logical"}:
            return os.cpu_count() or 1
        return int(value) if value.isdigit() else 1
    return 1


def blas_thread_env(pytest_args: tuple[str, ...]) -> dict[str, str]:
    """Single-threaded BLAS/OpenMP for parallel runs; caller-set values win."""
    if _xdist_workers(pytest_args) < 2:
        return {}
    env = {name: "1" for name in BLAS_THREAD_VARS if name not in os.environ}
    if env:
        print(
            "pyrite-dev: 1 BLAS/OpenMP thread per xdist worker "
            "(set OPENBLAS_NUM_THREADS/OMP_NUM_THREADS to override)",
            file=sys.stderr,
        )
    return env
