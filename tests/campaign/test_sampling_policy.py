"""``Numerics.sampling`` is the only electron budget and keeps case identities (#680, #1007)."""

import inspect
import pickle
from dataclasses import FrozenInstanceError, fields, replace

import pytest

import pyrite as pr
from pyrite._sampling import adaptive_sampling
from pyrite.campaign.profiles import case_content_key
from pyrite.config import FixedSampling


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
        brem_max_electrons=60,
    )


def test_fixed_policy_lowers_to_counts_without_adaptive_identity():
    from pyrite.campaign.lowering import build_case

    numerics = pr.Numerics(sampling=FixedSampling(cxr_electrons=31, brem_electrons=17))
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
    assert case.Ne == case.Ne_brem == policy.max_electrons
    assert policy.brem_policy().max_electrons == 60
    assert pickle.loads(pickle.dumps(numerics)).sampling == policy
    described = numerics.describe_sampling()
    assert described["mode"] == "adaptive"
    assert described["effective"]["maximum_brem_electrons"] == 60


def test_catalog_precision_converts_to_equal_adaptive_policy():
    policy = _policy()
    stored = pr.config.Precision.from_dict(policy.to_dict())
    converted = adaptive_sampling(stored)
    assert converted == policy
    assert adaptive_sampling(policy) is policy


def test_default_budget_is_the_single_stored_policy():
    numerics = pr.Numerics()
    assert numerics.sampling == FixedSampling(cxr_electrons=450, brem_electrons=100)
    assert numerics.resolved_sampling is numerics.sampling
    assert numerics.describe_sampling() == {
        "mode": "fixed",
        "requested": {"cxr_electrons": 450, "brem_electrons": 100},
        "effective": {"cxr_electrons": 450, "brem_electrons": 100},
    }


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "5"])
def test_fixed_policy_validates_counts(value):
    with pytest.raises(ValueError, match="positive integer"):
        pr.config.FixedSampling(cxr_electrons=value)


@pytest.mark.parametrize(
    ("argument", "value", "replacement"),
    [
        ("cxr_trials", 7, r"sampling=FixedSampling\(cxr_electrons"),
        ("brem_trials", 7, r"sampling=FixedSampling\(brem_electrons"),
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


def test_positional_budget_raises_naming_sampling():
    with pytest.raises(TypeError, match=r"sampling=FixedSampling"):
        pr.Numerics(31, 17)


def test_signature_stays_introspectable():
    parameters = inspect.signature(pr.Numerics).parameters
    assert "sampling" in parameters
    assert not {"cxr_trials", "brem_trials", "precision"} & set(parameters)


@pytest.mark.parametrize("value", [None, {"mode": "fixed"}, pr.config.Precision(0.1, 40, 80, 20)])
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
