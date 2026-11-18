"""CPU recovery must switch leaf staging and restore driver device bindings."""

import numpy as np
import pytest

from pyrite.montecarlo.runner.chunking import _cpu_spectrum_backend
from pyrite.montecarlo.spectrum.cxr import _kernels, _position_twins


def test_cpu_context_switches_leaf_staging_and_restores_on_error(monkeypatch):
    device = object()
    monkeypatch.setattr(_kernels, "xp", device)
    monkeypatch.setattr(_position_twins, "xp", device)
    row = np.arange(9, dtype=np.float64).reshape(3, 3)
    with pytest.raises(RuntimeError, match="restore"):
        with _cpu_spectrum_backend():
            with _kernels._RowStager() as stage:
                np.testing.assert_array_equal(stage(row), row)
                np.testing.assert_array_equal(
                    _position_twins.stage_row("r_mid_f64", row, stage), row
                )
            raise RuntimeError("restore")
    assert _kernels.xp is device
    assert _position_twins.xp is device


def test_backend_binding_inventory_and_module_scope():
    import ast
    from pathlib import Path

    from pyrite.montecarlo import spectrum
    from pyrite.montecarlo.runner.chunking import _SPECTRUM_BACKEND_MODULES

    root = Path(spectrum.__file__).parent
    found = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in (*node.args.defaults, *node.args.kw_defaults):
                    if default is not None:
                        assert not any(
                            isinstance(name, ast.Name) and name.id in {"xp", "REAL", "BACKEND"}
                            for name in ast.walk(default)
                        ), f"resolve backend defaults at call time: {path}:{node.name}"
            if not isinstance(node, ast.ImportFrom) or not (node.module or "").endswith("_backend"):
                continue
            bindings = [a for a in node.names if a.name in {"xp", "REAL", "BACKEND"}]
            if bindings:
                assert node in tree.body, f"move backend binding to module scope: {path}"
                assert all(a.asname is None for a in bindings), path
                found.add(
                    ".".join(path.relative_to(root).with_suffix("").parts).removesuffix(".__init__")
                )
    assert found == set(_SPECTRUM_BACKEND_MODULES)


@pytest.mark.hardware
@pytest.mark.slow
def test_cuda_oom_recovers_without_device_allocations(monkeypatch, tmp_path, pytestconfig):
    import os
    import pickle
    import subprocess
    import sys

    from pyrite._backend import BACKEND
    from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
    from pyrite.montecarlo import runner

    if BACKEND.name != "cuda":
        pytest.skip("requires CUDA backend")
    import cupy as cp

    # conftest deliberately clears the ambient workspace. This hardware test
    # uses the pinned datasets/tables shipped to its isolated checkout, and
    # gives the independent NumPy subprocess the same data root.
    monkeypatch.setenv("PYRITE_HOME", str(pytestconfig.rootpath))
    case = build_cases(
        Sweep(
            material="hopg",
            thickness_ang=5000,
            beam=BeamSpec(energy_keV=30),
            tilt_deg=45,
            n_electrons=16,
            n_electrons_brem=16,
        )
    )[0]
    tp = runner._transport_case(case, transport_core="cuda", keep_segments_on_device=False)
    payload, expected = tmp_path / "input.pkl", tmp_path / "expected.pkl"
    payload.write_bytes(pickle.dumps((case, tp)))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import pickle,sys; from pyrite.montecarlo import runner; "
            "case,tp=pickle.load(open(sys.argv[1],'rb')); "
            "pickle.dump(runner._spectrum_case(case,tp),open(sys.argv[2],'wb'))",
            str(payload),
            str(expected),
        ],
        check=True,
        env={**os.environ, "PYRITE_MC_BACKEND": "cpu"},
    )
    reference = pickle.loads(expected.read_bytes())

    def oom(*args, **kwargs):
        raise cp.cuda.memory.OutOfMemoryError(1024, 1024)

    monkeypatch.setenv("PYRITE_MC_BACKEND", "auto")
    monkeypatch.setattr(runner, "_spectrum_case_retry", oom)
    # The allocator hook detects even allocations freed before pool.used_bytes().
    allocations = []
    pool = cp.get_default_memory_pool()

    def allocate(size):
        allocations.append(size)
        return pool.malloc(size)

    used = pool.used_bytes()
    with cp.cuda.using_allocator(allocate):
        with pytest.warns(RuntimeWarning, match="CPU NumPy"):
            actual = runner._spectrum_with_recovery(case, tp, engine="auto")
    assert allocations == []
    assert pool.used_bytes() == used
    assert actual["_recovery_path"] == "cpu_spectrum"
    for key in ("spec", "brem_wide"):
        np.testing.assert_allclose(actual[key], reference[key], rtol=1e-12, atol=0)
