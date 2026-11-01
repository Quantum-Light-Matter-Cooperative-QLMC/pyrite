"""Exact CUDA kernel precision policy and its KernelTuning forwarding (#838).

Device-free: resolution, the process variable, the public tuning field and
run provenance. Kernel behavior is in ``test_transport_mixed_precision_cuda.py``.
"""

import pickle

import pytest

import pyrite as pr
import pyrite.config as cfg
from pyrite.montecarlo.runner.transport_options import (
    _TRANSPORT_TUNING,
    transport_precision_override,
    transport_tuning,
    tuned_transport_kwargs,
)
from pyrite.montecarlo.transport.precision import (
    DEFAULT_TRANSPORT_PRECISION,
    TRANSPORT_PRECISION_MODEL,
    precision_provenance,
    resolve_transport_precision,
)


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("PYRITE_MC_TRANSPORT_PRECISION", raising=False)


def test_auto_resolves_to_the_default_on_exact_cuda_and_fp64_elsewhere():
    assert resolve_transport_precision("auto", exact_cuda=True) == DEFAULT_TRANSPORT_PRECISION
    assert resolve_transport_precision("auto", exact_cuda=False) == "fp64"
    assert resolve_transport_precision("fp64", exact_cuda=True) == "fp64"
    assert resolve_transport_precision("mixed", exact_cuda=True) == "mixed"


def test_explicit_mixed_off_the_exact_cuda_core_raises():
    with pytest.raises(ValueError, match="exact CUDA core"):
        resolve_transport_precision("mixed", exact_cuda=False)
    with pytest.raises(ValueError, match="transport_precision must be one of"):
        resolve_transport_precision("fp32", exact_cuda=True)


@pytest.mark.parametrize("value", ["fp64", "mixed"])
def test_process_variable_sets_auto_on_exact_cuda_only(monkeypatch, value):
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_PRECISION", value)
    assert resolve_transport_precision("auto", exact_cuda=True) == value
    assert resolve_transport_precision("auto", exact_cuda=False) == "fp64"


def test_invalid_process_variable_raises(monkeypatch):
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_PRECISION", "half")
    with pytest.raises(ValueError, match="PYRITE_MC_TRANSPORT_PRECISION"):
        resolve_transport_precision("auto", exact_cuda=True)


def test_tuning_field_validates_and_pickles():
    with pytest.raises(ValueError, match="precision"):
        cfg.TransportKernelTuning(precision="fp32")  # type: ignore[arg-type]
    tuning = cfg.TransportKernelTuning(precision="fp64")
    assert pickle.loads(pickle.dumps(tuning)) == tuning


@pytest.mark.parametrize("precision", ["fp64", "mixed"])
def test_precision_alone_forwards_without_a_batching_override(precision):
    numerics = pr.Numerics(
        execution=cfg.KernelTuning(transport=cfg.TransportKernelTuning(precision=precision))
    )
    with transport_tuning(numerics):
        assert _TRANSPORT_TUNING.get() is None
        assert transport_precision_override() == precision
        assert tuned_transport_kwargs() == {"transport_precision": precision}
    assert transport_precision_override() is None
    assert tuned_transport_kwargs() == {}


def test_precision_and_batching_forward_together():
    tuning = cfg.TransportKernelTuning(kernel_threads=32, precision="fp64")
    with transport_tuning(pr.Numerics(execution=cfg.KernelTuning(transport=tuning))):
        kwargs = tuned_transport_kwargs()
    assert kwargs["transport_precision"] == "fp64"
    assert kwargs["per_electron_config"].kernel_threads == 32


def test_provenance_records_policy_per_backend(monkeypatch):
    cuda = precision_provenance("cuda")
    assert cuda == {
        "model": TRANSPORT_PRECISION_MODEL,
        "exact_cuda": DEFAULT_TRANSPORT_PRECISION,
        "other_cores": "fp64",
    }
    assert precision_provenance("cpu")["exact_cuda"] == "fp64"
    assert precision_provenance("cuda", "fp64")["exact_cuda"] == "fp64"
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_PRECISION", "fp64")
    assert precision_provenance("cuda")["exact_cuda"] == "fp64"
