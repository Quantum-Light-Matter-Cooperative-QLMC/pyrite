"""``Numerics.sampling`` is the only history budget and keeps case identities (#680, #1007)."""

import inspect
import pickle
from dataclasses import FrozenInstanceError, fields, replace

import pytest

import pyrite as pr
from pyrite._precision import Precision
from pyrite._sampling import adaptive_sampling
from pyrite.campaign.profiles import case_content_key
from pyrite.config import AdaptiveSampling, FixedSampling


def _scene():
    return pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "hopg", thickness_ang=10_000.0, tilt_deg=30.0, footprint=pr.geometry.Footprint(5.0, 5.0)
        ),
        pr.Detector(),
        physics=pr.Physics(bremsstrahlung_model="eedl", radiative_model="uncoupled"),
    )


def _policy():
    return pr.config.AdaptiveSampling(
        0.1,
        40,
        80,
        20,
        observables=("cxr", "brem"),
        brem_target_rse=0.2,
        brem_max_histories=60,
    )


def test_fixed_policy_lowers_to_counts_without_adaptive_identity():
    from pyrite.campaign.lowering import build_case

    numerics = pr.Numerics(sampling=FixedSampling(cxr_histories=31, brem_histories=17))
    case = build_case(replace(_scene(), numerics=numerics))
    assert "adaptive_precision" not in case
    assert case.Ne == 31 and case.Ne_brem == 17
    default = build_case(replace(_scene(), numerics=pr.Numerics()))
    explicit = build_case(replace(_scene(), numerics=pr.Numerics(sampling=FixedSampling(450, 100))))
    assert pickle.dumps(default.to_dict(), protocol=5) == pickle.dumps(
        explicit.to_dict(), protocol=5
    )
    assert case_content_key(default) == case_content_key(explicit)


def test_adaptive_policy_keeps_stored_identity_and_brem_prefix():
    from pyrite.campaign.lowering import build_case

    policy = _policy()
    numerics = pr.Numerics(sampling=policy)
    case = build_case(replace(_scene(), numerics=numerics))
    assert case.adaptive_precision == policy.to_dict()
    assert case.adaptive_precision["observables"] == ("line", "brem")
    assert case.Ne == case.Ne_brem == policy.max_histories
    assert policy.brem_policy().max_histories == 60
    assert pickle.loads(pickle.dumps(numerics)).sampling == policy
    described = numerics.describe_sampling()
    assert described["mode"] == "adaptive"
    assert described["effective"]["maximum_brem_histories"] == 60


def test_catalog_precision_converts_to_equal_adaptive_policy():
    policy = _policy()
    stored = pr.config.AdaptiveSampling.from_dict(policy.to_dict())
    converted = adaptive_sampling(stored)
    assert converted == policy
    assert adaptive_sampling(policy) is policy


def test_default_budget_is_the_single_stored_policy():
    numerics = pr.Numerics()
    assert numerics.sampling == FixedSampling(cxr_histories=450, brem_histories=100)
    assert numerics.resolved_sampling is numerics.sampling
    assert numerics.describe_sampling() == {
        "mode": "fixed",
        "requested": {"cxr_histories": 450, "brem_histories": 100},
        "effective": {"cxr_histories": 450, "brem_histories": 100},
    }


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "5"])
def test_fixed_policy_validates_counts(value):
    with pytest.raises(ValueError, match="positive integer"):
        pr.config.FixedSampling(cxr_histories=value)


@pytest.mark.parametrize(
    ("argument", "value", "replacement"),
    [
        ("cxr_trials", 7, r"sampling=FixedSampling\(cxr_histories"),
        ("brem_trials", 7, r"sampling=FixedSampling\(brem_histories"),
        ("precision", _policy(), r"sampling=AdaptiveSampling"),
    ],
)
def test_removed_arguments_raise_naming_sampling(argument, value, replacement):
    with pytest.raises(TypeError, match=replacement):
        pr.Numerics(**{argument: value})
    with pytest.raises(TypeError, match=replacement):
        replace(pr.Numerics(), **{argument: value})
    assert not hasattr(pr.Numerics(), argument)
    assert argument not in {item.name for item in fields(pr.Numerics)}


@pytest.mark.parametrize(
    ("cls", "argument", "replacement"),
    [
        (FixedSampling, "cxr_electrons", "cxr_histories="),
        (FixedSampling, "brem_electrons", "brem_histories="),
        (FixedSampling, "cxr_trials", "cxr_histories="),
        (AdaptiveSampling, "min_electrons", "min_histories="),
        (AdaptiveSampling, "max_electrons", "max_histories="),
        (AdaptiveSampling, "block_electrons", "block_histories="),
        (AdaptiveSampling, "max_electron_share", "max_history_share="),
        (AdaptiveSampling, "min_effective_electrons", "min_effective_histories="),
        (AdaptiveSampling, "pilot_electrons", "pilot_histories="),
        (AdaptiveSampling, "brem_min_electrons", "brem_min_histories="),
        (AdaptiveSampling, "brem_max_electrons", "brem_max_histories="),
    ],
)
def test_electron_budget_names_raise_naming_histories(cls, argument, replacement):
    with pytest.raises(
        TypeError, match=rf"{cls.__name__}\({argument}=\.\.\.\) was removed; use {replacement}"
    ):
        cls(**{argument: 7})


def test_public_precision_is_removed_naming_adaptive_sampling():
    with pytest.raises(ImportError, match="use pyrite.config.AdaptiveSampling"):
        from pyrite.config import Precision  # noqa: F401
    assert "Precision" not in dir(pr.config)


def test_serialized_policy_keeps_the_stored_electron_vocabulary():
    """Stored ``adaptive_precision`` payloads and identities keep historical keys."""
    policy = AdaptiveSampling(
        0.1, 40, 80, observables=("cxr", "brem"), pilot_histories=40,
        brem_target_rse=0.2, brem_min_histories=20, brem_max_histories=60,
    )  # fmt: skip
    payload = policy.to_dict()
    assert list(payload) == [
        "target_rse", "min_electrons", "max_electrons", "observables",
        "max_electron_share", "min_effective_electrons", "max_variance_of_variance",
        "stability_checks",
        "stability_fraction", "pilot_electrons", "band_eV", "batch_means_band_eV",
        "brem_target_rse", "brem_min_electrons", "brem_max_electrons", "decision_growth",
    ]  # fmt: skip
    assert payload["observables"] == ("line", "brem")
    assert AdaptiveSampling.from_dict(payload) == policy


def test_positional_budget_raises_naming_sampling():
    with pytest.raises(TypeError, match=r"sampling=FixedSampling"):
        pr.Numerics(31, 17)


def test_signature_stays_introspectable():
    parameters = inspect.signature(pr.Numerics).parameters
    assert "sampling" in parameters
    assert not {"cxr_trials", "brem_trials", "precision"} & set(parameters)


@pytest.mark.parametrize("value", [None, {"mode": "fixed"}, Precision(0.1, 40, 80, 20)])
def test_sampling_requires_a_typed_policy(value):
    with pytest.raises(TypeError, match="sampling must be"):
        pr.Numerics(sampling=value)


def test_adaptive_sampling_rejects_the_lockstep_core():
    with pytest.raises(ValueError, match="per-electron or CUDA"):
        pr.Numerics(sampling=_policy(), transport_core="lockstep")
    with pytest.raises(ValueError, match="per-electron or CUDA"):
        replace(pr.Numerics(transport_core="lockstep"), sampling=_policy())


def test_replace_is_the_update_path_and_cannot_contradict():
    fixed = pr.Numerics(sampling=FixedSampling(31, 17))
    assert replace(fixed, max_dE_frac=0.01).sampling is fixed.sampling
    updated = replace(fixed, sampling=FixedSampling(31, 99))
    assert updated.sampling == FixedSampling(31, 99)
    # An explicit value equal to the old magic default is honored, never snapped back.
    assert replace(fixed, sampling=FixedSampling(450, 100)).sampling == FixedSampling(450, 100)
    adaptive = replace(fixed, sampling=_policy())
    assert adaptive.sampling == _policy()
    assert replace(adaptive, sampling=FixedSampling()).sampling == FixedSampling()
    assert {item.name for item in fields(adaptive)} >= {"sampling"}
    with pytest.raises(FrozenInstanceError):
        adaptive.sampling = FixedSampling()


def test_preset_values_build_the_single_policy():
    from pyrite._config_presets import numerics_from_values

    assert numerics_from_values({"n_electrons": 200}).sampling == FixedSampling(200, 100)
    assert numerics_from_values({}).sampling == FixedSampling()
    adaptive = numerics_from_values({"precision": _policy().to_dict()})
    assert adaptive.sampling == _policy()


def _legacy_payload():
    """A stored ``["cxr", "brem"]`` policy from before brem stopping mirrored CXR (#1398)."""
    return {
        "target_rse": 0.1,
        "min_electrons": 40,
        "max_electrons": 80,
        "observables": ["line", "brem"],
    }


def test_legacy_both_observables_resolve_to_explicit_independent_brem_stopping():
    from pyrite._precision import identity_policy, sampling_model

    legacy = _legacy_payload()
    with pytest.warns(DeprecationWarning, match="brem stopping without"):
        resolved = identity_policy(legacy)
    assert resolved["brem_target_rse"] == 0.1
    assert resolved["brem_max_electrons"] == 80
    assert resolved["brem_min_electrons"] == 40  # the minimum is inherited too (#1398)
    # The explicit spelling of the same request has the same identity: no stale reuse,
    # and no second identity for one rule.
    explicit = Precision.from_dict(
        {**legacy, "brem_target_rse": 0.1, "brem_min_electrons": 40, "brem_max_electrons": 80}
    ).to_dict()
    assert dict(resolved) == {**explicit, "observables": ["line", "brem"]}
    assert identity_policy(explicit) == explicit
    assert sampling_model(legacy) == sampling_model(explicit) == "decision-points-v5"


def test_legacy_policy_changes_case_identity_while_cxr_only_keeps_it():
    from pyrite._precision import _adaptive_sampling_markers

    fixed = {"Ne": 80, "Ne_brem": 80, "seed": 1}
    cxr_only = Precision(0.1, 40, 80).to_dict()
    # Pre-change bytes: no brem keys, the v4 marker.
    assert not any(key.startswith("brem_") for key in cxr_only)
    assert _adaptive_sampling_markers(cxr_only) == {"adaptive_sampling_model": "decision-points-v4"}
    cxr_key = case_content_key({**fixed, "adaptive_precision": cxr_only})
    with pytest.warns(DeprecationWarning):
        legacy_key = case_content_key({**fixed, "adaptive_precision": _legacy_payload()})
    assert legacy_key != cxr_key
    explicit = Precision(
        0.1,
        40,
        80,
        observables=("cxr", "brem"),
        brem_target_rse=0.1,
        brem_min_histories=40,
        brem_max_histories=80,
    )
    assert case_content_key({**fixed, "adaptive_precision": explicit.to_dict()}) == legacy_key
