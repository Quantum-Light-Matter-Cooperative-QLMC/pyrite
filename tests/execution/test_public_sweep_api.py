"""The persisted sweep coordinator has a governed, lazy public import."""

from pyrite.execution import run_sweep


def test_run_sweep_public_execution_facade_is_identity_preserving():
    from pyrite.runs.run import run_sweep as coordinator

    assert run_sweep is coordinator
    assert "checkpoint_dir" in __import__("inspect").signature(run_sweep).parameters
    assert "resume" in __import__("inspect").signature(run_sweep).parameters
