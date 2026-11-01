"""Exact CUDA kernel compiled from its generated source with options (#796).

``_jit_rawmodule`` recompiles the source ``cupyx.jit`` generated for the #827
variant, optionally with a register cap, launch bounds or ``__ldg`` loads of
the arrays the kernel only reads. None of these may change a result bit: every
returned array must equal the JIT-compiled kernel's. The comparisons need CUDA
and run through ``pyrite remote``.

Validation: gpu-transport-core
"""

import ast
from importlib import resources

import numpy as np
import pytest

from tests.montecarlo.transport.test_transport_continuation_cuda import (
    _HAS_CUDA,
    REFERENCE,
    _assert_identical,
    _continued,
)
from tests.montecarlo.transport.test_transport_kernel_variants_cuda import CASES

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")


def _kernel_parameters():
    tree = ast.parse(
        resources.files("pyrite.montecarlo.transport").joinpath("_jit_kernel.py").read_text()
    )
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_transport_kernel"
    )
    return [arg.arg for arg in kernel.args.args]


def _read_only_names():
    tree = ast.parse(
        resources.files("pyrite.montecarlo.transport").joinpath("_jit_rawmodule.py").read_text()
    )
    assign = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and node.targets[0].id == "TRANSPORT_READONLY"
    )
    return {element.value for element in assign.value.args[0].elts}


def test_read_only_names_are_kernel_parameters():
    names = _read_only_names()
    assert names <= set(_kernel_parameters())
    # Outputs and continuation state are written; they must never be read-only.
    assert not {n for n in names if n.startswith("seg_")}
    assert names.isdisjoint({"clock", "pos", "dirs", "E_keV", "exit_code", "stragg_dE"})
    assert names.isdisjoint({"cont_i32", "cont_u64", "cont_f64"})


def _builds():
    from pyrite.montecarlo.transport._jit_rawmodule import TRANSPORT_READONLY, KernelBuild

    return {
        "raw": KernelBuild(),
        "readonly": KernelBuild(readonly=TRANSPORT_READONLY),
    }


def _with_build(run, monkeypatch, build, *args, **kwargs):
    from pyrite.montecarlo.transport import _jit_launch
    from pyrite.montecarlo.transport._jit_rawmodule import RawModuleKernel

    original = _jit_launch.transport_kernel_variant
    monkeypatch.setattr(_jit_launch, "_SPECIALIZE_TRANSPORT_KERNEL", True)
    reference = run(*args, **kwargs)
    kernels = {}

    def variant(*key):
        if key not in kernels:
            kernels[key] = RawModuleKernel(original(*key), build)
        return kernels[key]

    monkeypatch.setattr(_jit_launch, "transport_kernel_variant", variant)
    candidate = run(*args, **kwargs)
    assert kernels
    return reference, candidate


@pytest.mark.hardware
@requires_cuda
@pytest.mark.parametrize("build", ["raw", "readonly"])
@pytest.mark.parametrize(("run", "kw"), CASES)
@pytest.mark.parametrize("schedule", ["full-history", "continuation"])
def test_rebuilt_kernel_matches_jit(run, kw, schedule, build, monkeypatch):
    config = REFERENCE if schedule == "full-history" else _continued(7, 37)
    reference, candidate = _with_build(run, monkeypatch, _builds()[build], config, **kw)
    _assert_identical(reference, candidate)
    assert np.asarray(reference["electron_id"]).size > 0
