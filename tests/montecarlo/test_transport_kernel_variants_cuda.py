"""Exact CUDA kernel specialized per feature configuration (#827).

Each variant compiles the kernel's switches as constants; the monolithic kernel
reads them at run time. Both come from one Python body, so every returned array
must be bitwise identical between the two. The source-structure test runs
without a device; the comparisons need CUDA and run through ``pyrite remote``.

Validation: gpu-transport-core
"""

import ast
from importlib import resources

import numpy as np
import pytest

from tests.montecarlo.test_transport_continuation_cuda import (
    _HAS_CUDA,
    NO_LUT,
    REFERENCE,
    _assert_identical,
    _continued,
    _elsepa,
    _layered,
    _radiative,
    _shell,
    _table,
)

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")

SWITCHES = (
    "elastic_model_code",
    "energy_model_code",
    "finite_footprint",
    "sbethe_on",
    "straggle_on",
    "inelastic_on",
    "sec_on",
    "rad_on",
    "cont_on",
)
# Secondary directions stay runtime-switched: see `TransportKernelSwitches`.
SPECIALIZED = tuple(name for name in SWITCHES if name != "sec_on")


def _module_tree(name):
    package = resources.files("pyrite.montecarlo.transport")
    return ast.parse(package.joinpath(f"{name}.py").read_text())


def test_switches_are_module_globals_not_kernel_parameters():
    tree = _module_tree("_jit_kernel")
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_transport_kernel"
    )
    parameters = [arg.arg for arg in kernel.args.args]
    bound = {
        node.targets[0].id
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    for name in SWITCHES:
        # A parameter of the same name would shadow the variant's constant.
        assert name not in parameters
        assert f"{name}_arg" in parameters
        assert name in bound
    assert not kernel.decorator_list
    switches = next(
        node
        for node in _module_tree("_jit_kernel_variants").body
        if isinstance(node, ast.ClassDef) and node.name == "TransportKernelSwitches"
    )
    fields = [node.target.id for node in switches.body if isinstance(node, ast.AnnAssign)]
    assert tuple(fields) == SPECIALIZED


def _paired(run, monkeypatch, *args, **kwargs):
    """Run with the monolithic kernel, then with the specialized one."""
    from pyrite.montecarlo.transport import _jit_launch

    monkeypatch.setattr(_jit_launch, "_SPECIALIZE_TRANSPORT_KERNEL", False)
    reference = run(*args, **kwargs)
    monkeypatch.setattr(_jit_launch, "_SPECIALIZE_TRANSPORT_KERNEL", True)
    return reference, run(*args, **kwargs)


def _sbethe_elsepa(config, **kw):
    """The #821 benchmark shape: ELSEPA, midpoint, SBETHE, coupled BremsLib."""
    from pyrite.montecarlo import shell_configuration
    from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

    if not shell_configuration._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    return _elsepa(
        config,
        stopping_tables=[resolve_catalog_table("silicon").arrays()],
        E_cut_keV=10.0,
        max_dE_frac=0.05,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables={"Si": _table(1e5, atomic_number=14)},
        **kw,
    )


CASES = [
    pytest.param(_layered, {"energy_model": "frozen"}, id="sr-prism-layers-frozen"),
    pytest.param(_layered, {"energy_model": "midpoint"}, id="sr-prism-layers-midpoint"),
    pytest.param(_layered, {"straggling": True}, id="sr-urban-straggling"),
    pytest.param(_radiative, {}, id="sr-radiative"),
    pytest.param(_radiative, {"straggling": True}, id="sr-radiative-straggling"),
    pytest.param(_shell, {}, id="shell"),
    pytest.param(_shell, {"secondaries": True}, id="shell-secondaries"),
    pytest.param(_shell, {"straggling": True}, id="shell-straggling"),
    pytest.param(
        _shell, {"secondaries": True, "straggling": True, "radiative": True}, id="shell-all"
    ),
    pytest.param(_elsepa, {"energy_model": "frozen"}, id="elsepa-frozen"),
    pytest.param(_elsepa, {"energy_model": "midpoint"}, id="elsepa-midpoint"),
    pytest.param(
        _elsepa, {"elastic_model": "mott", "transport_lut_config": NO_LUT, "Ne": 60}, id="mott"
    ),
    pytest.param(_sbethe_elsepa, {}, id="elsepa-sbethe-radiative"),
]


@pytest.mark.hardware
@requires_cuda
@pytest.mark.parametrize(("run", "kw"), CASES)
@pytest.mark.parametrize("schedule", ["full-history", "continuation"])
def test_specialized_kernel_matches_monolithic(run, kw, schedule, monkeypatch):
    config = REFERENCE if schedule == "full-history" else _continued(7, 37)
    reference, specialized = _paired(run, monkeypatch, config, **kw)
    _assert_identical(reference, specialized)
    assert np.asarray(reference["electron_id"]).size > 0


@pytest.mark.hardware
@requires_cuda
def test_variants_are_cached_per_switch_tuple():
    from pyrite.montecarlo.transport._jit_kernel_variants import (
        TransportKernelSwitches,
        transport_kernel_variant,
    )

    off = dict.fromkeys(SPECIALIZED, 0)
    plain = TransportKernelSwitches.normalized(**(off | {"straggle_on": np.bool_(False)}))
    assert plain == TransportKernelSwitches(*([0] * len(SPECIALIZED)))
    shell = TransportKernelSwitches.normalized(**(off | {"inelastic_on": True}))
    assert transport_kernel_variant(plain) is transport_kernel_variant(plain)
    assert transport_kernel_variant(plain) is not transport_kernel_variant(shell)
