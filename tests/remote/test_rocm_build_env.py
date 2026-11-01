"""The AMD prelude's CuPy-HIP build environment (``rocm_build_env.sh``, #888).

Runs the sourced script against fake ROCm layouts: a complete install must pass
through untouched, and a distribution layout gets the compiler, header and
roctracer gaps filled exactly once.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import pyrite.remote

SCRIPT = Path(pyrite.remote.__file__).with_name("rocm_build_env.sh")
_ENV_KEYS = ("ROCM_HOME", "ROCM_PATH", "CC", "CXX", "CPATH", "LIBRARY_PATH", "LD_LIBRARY_PATH")


def _source(tmp_path, rocm_home, *, path, extra=None):
    env = {
        "PATH": path,
        "HOME": str(tmp_path),
        "ROCM_HOME": str(rocm_home),
        "HCC_AMDGPU_TARGET": "gfx1151",
        **(extra or {}),
    }
    probe = "; ".join(f'echo "{key}=${{{key}:-}}"' for key in _ENV_KEYS)
    completed = subprocess.run(  # noqa: S603
        ["/bin/bash", "-uc", f'. "$1" && _pyrite_rocm_env "$2"; echo "rc=$?"; {probe}', "_"]
        + [str(SCRIPT), str(tmp_path / "checkout")],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    values = dict(line.split("=", 1) for line in completed.stdout.splitlines())
    return values, completed.stderr


def _bin_with(tmp_path, name, **tools):
    """A PATH directory holding only the named tools (symlinks or scripts)."""
    directory = tmp_path / name
    directory.mkdir()
    for tool in ("grep", "mkdir", "cp", "cat"):
        os.symlink(shutil.which(tool), directory / tool)
    for tool, target in tools.items():
        if isinstance(target, Path):
            os.symlink(target, directory / tool)
        else:
            (directory / tool).write_text(f"#!/bin/sh\n{target}\n")
            (directory / tool).chmod(0o755)
    return str(directory)


def test_complete_rocm_install_passes_through(tmp_path):
    rocm = tmp_path / "opt-rocm"
    (rocm / "include" / "roctracer").mkdir(parents=True)
    (rocm / "include" / "roctracer" / "roctracer_ext.h").write_text("")
    for header in ("roctx.h", "roctracer/roctx.h"):
        (rocm / "include" / header).write_text("")
    (rocm / "lib").mkdir()
    (rocm / "lib" / "libroctx64.so").write_text("")
    for lib in ("hipblas", "hipsparse", "hipfft", "rocsolver"):
        (rocm / "include" / f"{lib}.h").write_text("")
    path = _bin_with(tmp_path, "bin", **{"c++": "true"})

    values, stderr = _source(tmp_path, rocm, path=path)

    assert values["rc"] == "0"
    assert values["ROCM_PATH"] == str(rocm)
    assert values["CC"] == values["CXX"] == values["CPATH"] == ""
    assert stderr == ""
    assert not (tmp_path / "checkout" / ".rocm-shim").exists()


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler for the stub libraries")
def test_distribution_rocm_gets_hip_clang_flat_headers_and_roctracer_stubs(tmp_path):
    rocm = tmp_path / "usr"
    for lib in ("hipblas", "hipsparse", "hipfft", "rocsolver"):
        (rocm / "include" / lib).mkdir(parents=True)
        (rocm / "include" / lib / f"{lib}.h").write_text("")
    hip_bin = tmp_path / "llvm" / "bin"
    hip_bin.mkdir(parents=True)
    os.symlink(shutil.which("cc"), hip_bin / "clang")
    os.symlink(shutil.which("cc"), hip_bin / "clang++")
    # No c++ on PATH; hipconfig names the HIP clang directory. The stub
    # compile still needs the assembler and linker.
    tools = {tool: Path(shutil.which(tool)) for tool in ("as", "ld")}
    path = _bin_with(tmp_path, "bin", hipconfig=f"echo {hip_bin}", **tools)
    shim = tmp_path / "checkout" / ".rocm-shim"

    values, stderr = _source(tmp_path, rocm, path=path)

    assert values["rc"] == "0"
    assert values["CXX"] == str(hip_bin / "clang++")
    assert values["CPATH"] == f"{shim}/include"
    assert values["LD_LIBRARY_PATH"] == values["LIBRARY_PATH"] == f"{shim}/lib"
    assert (
        shim / "include" / "hipblas.h"
    ).read_text() == "#pragma once\n#include <hipblas/hipblas.h>\n"
    assert (shim / "include" / "roctracer" / "roctracer_ext.h").exists()
    assert (shim / "lib" / "libroctx64.so").exists()
    assert (shim / "lib" / "libroctracer64.so").exists()
    assert "evaluation shim" in stderr

    # A second job reuses the shim without rebuilding it.
    stamp = (shim / "lib" / "libroctx64.so").stat().st_mtime_ns
    _source(tmp_path, rocm, path=path)
    assert (shim / "lib" / "libroctx64.so").stat().st_mtime_ns == stamp


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler for the stub libraries")
def test_real_roctx_is_kept_and_stale_roctx_stubs_are_removed(tmp_path):
    # Ubuntu's libroctx-dev layout: roctracer/roctx.h and a multiarch link
    # library, but no roctracer. A shim from before the install holds stubs.
    rocm = tmp_path / "usr"
    (rocm / "include" / "roctracer").mkdir(parents=True)
    (rocm / "include" / "roctracer" / "roctx.h").write_text("")
    multiarch = rocm / "lib" / "x86_64-linux-gnu"
    multiarch.mkdir(parents=True)
    (multiarch / "libroctx64.so").write_text("")
    shim = tmp_path / "checkout" / ".rocm-shim"
    (shim / "lib").mkdir(parents=True)
    (shim / "include" / "roctracer").mkdir(parents=True)
    for stale in ("lib/libroctx64.so", "include/roctx.h", "include/roctracer/roctx.h"):
        (shim / stale).write_text("stale stub")
    path = _bin_with(tmp_path, "bin", **{"c++": "true", "cc": Path(shutil.which("cc"))})
    path = f"{path}:{Path(shutil.which('as')).parent}"

    values, stderr = _source(tmp_path, rocm, path=path)

    assert values["rc"] == "0"
    assert not (shim / "lib" / "libroctx64.so").exists()
    assert not (shim / "include" / "roctracer" / "roctx.h").exists()
    assert (shim / "include" / "roctx.h").read_text() == (
        "#pragma once\n#include <roctracer/roctx.h>\n"
    )
    assert (shim / "lib" / "libroctracer64.so").exists()
    assert (shim / "include" / "roctracer" / "roctracer_ext.h").exists()
    assert "no-op roctx=0, no-op roctracer=1" in stderr
