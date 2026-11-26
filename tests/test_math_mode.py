"""Math mode switch (#1468): resolution, per-mode kernels, identity and provenance."""

import pickle
import threading
from dataclasses import replace
from types import SimpleNamespace

import numba
import numpy as np
import pytest

import pyrite as pr
from pyrite import _backend, _math_mode
from pyrite._math_mode import (
    DEFAULT_MATH_MODE,
    FAST_QUALNAME_SUFFIX,
    MATH_MODES,
    active_math_mode,
    case_math_mode,
    math_mode_scope,
    math_provenance,
    mode_njit,
    resolve_math_mode,
    with_case_math_mode,
)
from pyrite._numerics import PROFILE_NUMERICS_KEYS, validate_profile_numerics
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.lowering import build_case
from pyrite.campaign.profiles import case_content_key, dataset_identity, resolve_numerics
from pyrite.console import config as _config
from pyrite.montecarlo.case import _ABSENT
from pyrite.results.store import Settings


@pytest.fixture
def store(monkeypatch, tmp_path):
    """An isolated config store with no PYRITE_MATH override."""
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "pyrite" / "config.toml")
    monkeypatch.delenv("PYRITE_MATH", raising=False)
    _math_mode.reset_math_mode_cache()
    yield _config
    _math_mode.reset_math_mode_cache()


# ---- resolution ------------------------------------------------------------------
def test_precedence_explicit_env_config_default(store, monkeypatch):
    assert resolve_math_mode() == _math_mode.ResolvedMathMode("reproducible", "built-in default")
    store.set_stored("numerics.math", "fast")
    assert resolve_math_mode() == _math_mode.ResolvedMathMode("fast", "config store")
    monkeypatch.setenv("PYRITE_MATH", "reproducible")
    assert resolve_math_mode() == _math_mode.ResolvedMathMode("reproducible", "PYRITE_MATH")
    assert resolve_math_mode("fast") == _math_mode.ResolvedMathMode("fast", "explicit")


def test_invalid_values_name_their_source(store, monkeypatch):
    with pytest.raises(ValueError, match="math must be one of fast, reproducible"):
        resolve_math_mode("exact")
    monkeypatch.setenv("PYRITE_MATH", "turbo")
    with pytest.raises(ValueError, match="PYRITE_MATH must be one of"):
        resolve_math_mode()


def test_default_and_config_default_agree():
    assert DEFAULT_MATH_MODE == "reproducible"
    assert _config.describe("numerics.math")[:2] == ("PYRITE_MATH", DEFAULT_MATH_MODE)
    assert _config.domain("numerics.math") == " | ".join(MATH_MODES)


def test_suite_runs_reproducible():
    """conftest pins PYRITE_MATH so identity pins and baselines stay bitwise."""
    assert resolve_math_mode().mode in MATH_MODES
    assert pr.ExecutionConfig().math is None


# ---- active scope ----------------------------------------------------------------
def test_scope_sets_active_mode_and_refuses_mixing(store):
    assert active_math_mode() == "reproducible"
    with math_mode_scope("fast"):
        assert active_math_mode() == "fast"
        seen = []
        worker = threading.Thread(target=lambda: seen.append(active_math_mode()))
        worker.start()
        worker.join()
        assert seen == ["fast"]  # worker threads see the case's mode
        with math_mode_scope("fast"):
            pass
        with pytest.raises(RuntimeError, match="one math mode at a time"):
            with math_mode_scope("reproducible"):
                pass
    assert active_math_mode() == "reproducible"


def test_case_decorator_enters_the_case_mode(store):
    @with_case_math_mode
    def probe(case):
        return active_math_mode()

    assert probe({"math": "fast"}) == "fast"
    assert probe({}) == "reproducible"
    assert case_math_mode({}) == "reproducible"


# ---- per-mode Numba dispatchers --------------------------------------------------
@mode_njit(fast=True, cache=True)
def _sine_sum(values):
    total = 0.0
    for value in values:
        total += np.sin(value)
    return total


@mode_njit(fast=("contract", "afn"))
def _scaled(value):
    return 2.0 * value + 1.0


def test_reproducible_dispatcher_is_plain_njit_and_fast_differs(store):
    reproducible = _sine_sum.for_mode("reproducible")
    fast = _sine_sum.for_mode("fast")
    assert _sine_sum.for_mode("reproducible") is reproducible
    if numba.config.DISABLE_JIT:
        pytest.skip("dispatchers are plain functions with JIT disabled")
    assert "fastmath" not in reproducible.targetoptions
    assert fast.targetoptions["fastmath"] is True
    assert _scaled.for_mode("fast").targetoptions["fastmath"] == {"contract", "afn"}
    # Numba's cache index ignores fastmath flags; the fast copy gets its own.
    assert fast.py_func.__qualname__ == "_sine_sum" + FAST_QUALNAME_SUFFIX
    assert reproducible._cache._cache_file._index_name != fast._cache._cache_file._index_name


def test_call_dispatches_on_the_active_mode(store):
    values = np.linspace(0.0, 3.0, 257)
    expected = float(np.sum(np.sin(values)))
    assert _sine_sum(values) == _sine_sum.for_mode("reproducible")(values)
    with math_mode_scope("fast"):
        assert _sine_sum(values) == _sine_sum.for_mode("fast")(values)
        assert _sine_sum(values) == pytest.approx(expected, rel=1e-12)


def test_fast_flag_validation():
    with pytest.raises(ValueError, match="nonempty subset"):
        mode_njit(fast=("bogus",))(lambda: None)
    with pytest.raises(ValueError, match="fast="):
        mode_njit(fast=True, fastmath=True)(lambda: None)


# ---- GPU compile options ---------------------------------------------------------
@pytest.mark.parametrize("hip", [False, True])
def test_rawkernel_options_per_mode(hip):
    module = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=hip)))
    exact = _backend.exact_rawkernel_options(module)
    assert _backend.rawkernel_options(module, "reproducible") == exact
    assert _backend.rawkernel_options(module, "reproducible", fast=("-x",)) == exact
    fast = _backend.rawkernel_options(module, "fast", fast=("-x",))
    assert fast[-1] == "-x"
    assert ("-ffp-contract=fast" if hip else "--fmad=true") in fast
    with math_mode_scope("reproducible"):
        assert _backend.rawkernel_options(module) == exact


# ---- configuration surfaces -------------------------------------------------------
def test_execution_config_field_and_pickle():
    assert pr.ExecutionConfig(math="fast").math == "fast"
    with pytest.raises(ValueError, match="math must be one of"):
        pr.ExecutionConfig(math="exact")
    state = dict(pr.ExecutionConfig().__dict__)
    del state["math"]
    restored = object.__new__(pr.ExecutionConfig)
    restored.__setstate__(state)
    assert restored == pr.ExecutionConfig()


def test_profile_and_preset_key(store):
    assert "math" in PROFILE_NUMERICS_KEYS
    validate_profile_numerics({"math": "fast"})
    with pytest.raises(ValueError, match="math must be one of"):
        validate_profile_numerics({"math": "exact"})
    assert resolve_numerics().effective["math"] is None
    pinned = resolve_numerics({"math": "fast"})
    assert pinned.effective["math"] == "fast"
    assert pinned.sources["math"] == "profile"


def test_settings_field_validates():
    assert Settings().math is None
    with pytest.raises(ValueError, match="math must be one of"):
        Settings(math="exact")


# ---- identity --------------------------------------------------------------------
def _scene():
    return pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "hopg", thickness_ang=10_000.0, tilt_deg=30.0, footprint=pr.geometry.Footprint(5.0, 5.0)
        ),
        pr.Detector(),
        physics=pr.Physics(bremsstrahlung_model="eedl", radiative_model="uncoupled"),
    )


def _bytes(case):
    return pickle.dumps(case.to_dict(), protocol=5)


def _case(math=None):
    numerics = pr.Numerics(execution=pr.ExecutionConfig(math=math))
    return build_case(replace(_scene(), numerics=numerics))


def test_reproducible_case_is_unchanged_and_fast_case_is_keyed(store, monkeypatch):
    legacy = build_case(_scene())
    reproducible = _case("reproducible")
    fast = _case("fast")
    assert _bytes(reproducible) == _bytes(legacy)
    assert "math" not in reproducible.to_dict()
    assert case_content_key(reproducible) == case_content_key(legacy)
    assert fast.math == "fast" and case_math_mode(fast) == "fast"
    assert case_content_key(fast) != case_content_key(reproducible)
    # Unset resolves through the environment at lowering time.
    monkeypatch.setenv("PYRITE_MATH", "fast")
    assert _case(None).math == "fast"
    assert _bytes(_case("reproducible")) == _bytes(legacy)


def test_case_pickle_and_validation(store):
    fast = _case("fast")
    assert pickle.loads(pickle.dumps(fast)) == fast
    legacy = build_case(_scene())
    assert len(legacy.__getstate__()) == len(pickle.loads(pickle.dumps(legacy)).__getstate__())
    with pytest.raises(ValueError, match="math must be absent"):
        replace(legacy, math="reproducible")


def test_dataset_identity_is_divergence_only(store, monkeypatch):
    sweep = material_sweep("hopg")

    def identity(settings):
        return dataset_identity("hopg", settings, sweep, bremsstrahlung_model="eedl")

    base = identity(default_settings())
    assert identity(replace(default_settings(), math="reproducible")) == base
    fast = identity(replace(default_settings(), math="fast"))
    assert fast["resolved_parameters"]["math"] == "fast"
    assert "math" not in base["resolved_parameters"]
    # The fast kernels' generation (#1469) separates fast digests only.
    assert fast["resolved_parameters"]["math_fast_model"] == _math_mode.MATH_FAST_MODEL
    assert "math_fast_model" not in base["resolved_parameters"]
    assert fast["parameter_sha256"] != base["parameter_sha256"]
    # An unset mode resolves like the lowering does: resume never mixes modes.
    monkeypatch.setenv("PYRITE_MATH", "fast")
    assert identity(default_settings()) == fast


def test_configured_campaign_lowers_settings_math(store, monkeypatch):
    from pyrite.api import build_configured_cases
    from pyrite.campaign.config import material_sweep as configured_sweep

    sweep = configured_sweep("hopg")
    settings = replace(
        default_settings(),
        bremsstrahlung_model="eedl",
        radiative_model="uncoupled",
        radiative_cutoff_eV=None,
    )
    reproducible = build_configured_cases(sweep, settings)
    fast = build_configured_cases(sweep, replace(settings, math="fast"))
    assert {case.math for case in fast} == {"fast"}
    assert [_bytes(case) for case in reproducible] == [
        _bytes(replace(case, math=_ABSENT)) for case in fast
    ]
    assert all("math" not in case.to_dict() for case in reproducible)
    from pyrite.campaign.profiles import case_content_key

    key = case_content_key(fast[0], xsgen_tables={})
    assert key != case_content_key(replace(fast[0], math=_ABSENT), xsgen_tables={})
    monkeypatch.setattr(_math_mode, "MATH_FAST_MODEL", _math_mode.MATH_FAST_MODEL + "-next")
    assert case_content_key(fast[0], xsgen_tables={}) != key


# ---- provenance ------------------------------------------------------------------
def test_provenance_names_mode_and_stack():
    record = math_provenance("fast", gpu_model="Test GPU")
    assert record["mode"] == "fast"
    assert record["numba"] == numba.__version__
    assert isinstance(record["svml"], bool)
    assert record["cpu"] and isinstance(record["cpu_features"], list)
    assert record["gpu_model"] == "Test GPU"


def test_run_provenance_records_the_case_mode(store):
    from pyrite.api import run_provenance

    record = run_provenance(_case("fast"), None)
    assert record["math"]["mode"] == "fast"
    assert record["math"]["gpu_model"] is None
    assert run_provenance(build_case(_scene()), None)["math"]["mode"] == "reproducible"
