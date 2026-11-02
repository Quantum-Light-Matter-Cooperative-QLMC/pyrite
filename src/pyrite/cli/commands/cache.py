"""Inspect and clear PyRITE's compiled-kernel (JIT) caches.

The cache logic lives in :mod:`pyrite._jit_cache`; this module only renders it.
Building the group imports neither Numba nor CuPy.
"""

import click

from ...console.output import (
    CLIError,
    confirm_destructive,
    emit_diagnostic,
    emit_json,
    emit_result,
    emit_table,
    output_option,
)

_BACKENDS = click.Choice(("all", "numba", "cupy"))


def _size(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _selected(backend: str):
    from ... import _jit_cache

    return [
        location
        for location in _jit_cache.cache_locations()
        if backend in ("all", location.backend)
    ]


@click.group("cache")
def command() -> None:
    """Inspect, warm, and clear compiled CPU (Numba) and GPU (CuPy) kernel caches.

    PyRITE compiles its kernels on first use and the native Numba and CuPy
    caches reuse them in later processes. The pyrite command and the Python API
    (import pyrite) store them under the PyRITE user cache directory unless
    NUMBA_CACHE_DIR or CUPY_CACHE_DIR is already set to a non-empty value. The
    older in-tree __pycache__ location beside the PyRITE sources is reported as
    a legacy location.

    \b
    Examples:
      pyrite cache status
      pyrite cache status --backend numba -o json
      pyrite cache warm
      pyrite cache warm --profile standard -o json
      pyrite cache clear
      pyrite cache clear --backend numba --yes
    """


@command.command("status")
@click.option(
    "--backend", type=_BACKENDS, default="all", show_default=True, help="Caches to report."
)
@output_option
def status_command(backend: str, json_output: bool) -> None:
    """Report cache locations, sizes, and per-family Numba freshness.

    A Numba index is current when its recorded source stamp and Numba version
    match this installation, so its signatures load without compiling (cache
    hits); stale indexes recompile on next use (misses). CuPy files are
    hash-named and reported only as a total. Read-only.
    """
    from ... import _jit_cache

    records = [_jit_cache.location_summary(loc) for loc in _selected(backend)]
    if json_output:
        emit_json("pyrite.cache.status.v1", {"locations": records})
        return
    rows = []
    for record in records:
        files = record.get("files", record.get("pyrite_files", 0))
        size = record.get("bytes", record.get("pyrite_bytes", 0))
        state = "active" if record["active"] else "inactive"
        rows.append(
            (record["backend"], record["source"], state, files, _size(size), record["path"])
        )
    emit_table(("BACKEND", "SOURCE", "STATE", "FILES", "SIZE", "PATH"), rows)
    for record in records:
        families = record.get("families") or []
        if not families:
            continue
        emit_result("")
        emit_result(f"Numba kernel families in {record['source']} cache:")
        emit_table(
            ("FAMILY", "INDEXES", "CURRENT", "STALE", "SIGNATURES", "SIZE"),
            [
                (
                    f["family"],
                    f["indexes"],
                    f["current"],
                    f["stale"],
                    f["overloads"],
                    _size(f["bytes"]),
                )
                for f in families
            ],
        )


@command.command("clear")
@click.option(
    "--backend", type=_BACKENDS, default="all", show_default=True, help="Caches to clear."
)
@click.option("-y", "--yes", is_flag=True, help="Delete the exact previewed paths.")
def clear_command(backend: str, yes: bool) -> None:
    """Delete PyRITE-owned compiled kernels; preview unless --yes.

    Removes the managed cache directories whole, and only PyRITE package
    entries from a user-set NUMBA_CACHE_DIR or the legacy in-tree __pycache__.
    A user-set or default CuPy cache is never touched. Kernels recompile on next
    use; results do not change.
    """
    from ... import _jit_cache

    targets = _jit_cache.clear_plan(_selected(backend))
    deletions = [t for t in targets if t.kind != "skip"]
    for target in targets:
        if target.kind == "skip":
            emit_diagnostic(f"skip {target.backend} {target.path}: {target.reason}")
    if not deletions:
        emit_result("no PyRITE-owned compiled kernels to clear")
        return
    for target in deletions:
        emit_result(f"delete {target.backend} {target.kind} {target.path}")
    if not confirm_destructive(yes, f"Delete {len(deletions)} cache path(s)?"):
        return
    try:
        removed = _jit_cache.execute_clear(deletions)
    except OSError as exc:
        raise CLIError(f"cache clear failed: {exc}") from exc
    emit_result(f"removed {len(removed)} cache path(s)")


@command.command("warm")
@click.option(
    "--profile",
    type=click.Choice(("minimal", "standard")),
    default="minimal",
    show_default=True,
    help="Representative cases to compile: minimal (default physics) or standard "
    "(adds continuous stopping, uncoupled continuum, and secondary transport).",
)
@output_option
def warm_command(profile: str, json_output: bool) -> None:
    """Compile and cache common kernels ahead of the first simulation.

    Runs each profile case once with one electron on the backend that
    PYRITE_MC_BACKEND selects and discards the result, so later processes
    sharing these caches load the kernels instead of compiling them. Takes
    tens of seconds per case on a cold cache and about a second when already
    warm. Uses the cache locations `pyrite cache status` reports; physics
    tables the cases need must already be installed. Exits 1 if any case
    fails; the remaining cases still run.
    """
    from ... import _jit_warm

    report = _jit_warm.warm(
        profile, progress=lambda case: emit_diagnostic(f"warming {case.name}: {case.covers}")
    )
    failed = [case for case in report["cases"] if case["status"] != "ok"]
    if json_output:
        errors = [{"case": case["case"], "message": case["error"]} for case in failed]
        emit_json("pyrite.cache.warm.v1", report, errors=errors)
    else:
        cupy = report["backend"] in ("cuda", "rocm")
        headers = ("CASE", "STATUS", "SECONDS", "NUMBA_COMPILES", "NUMBA_S")
        rows = []
        for case in report["cases"]:
            ok = case["status"] == "ok"
            row = [
                case["case"],
                case["status"],
                f"{case['seconds']:.1f}",
                case["numba_compiles"] if ok else "-",
                f"{case['numba_compile_s']:.1f}" if ok else "-",
            ]
            if cupy:
                row.append(case["cupy"]["nvrtc_compiles"] if ok else "-")
            rows.append(tuple(row))
        emit_table((*headers, "NVRTC_COMPILES") if cupy else headers, rows)
        for case in failed:
            emit_diagnostic(f"{case['case']} failed: {case['error']}")
    if failed:
        raise click.exceptions.Exit(1)
