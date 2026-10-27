"""Public tagged sampling policies must lower to unchanged case identities (#680)."""

import pickle
from dataclasses import replace

import pytest

import pyrite as pr
from pyrite.campaign.profiles import case_content_key


def _scene():
    return pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "hopg", thickness_ang=10_000.0, tilt_deg=30.0, footprint=pr.geometry.Footprint(5.0, 5.0)
        ),
        pr.Detector(),
        physics=pr.Physics(bremsstrahlung_model="eedl"),
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


def test_fixed_policy_matches_legacy_case_payload_and_key():
    scene = _scene()
    from pyrite.campaign.lowering import build_case

    old = build_case(replace(scene, numerics=pr.Numerics(cxr_trials=31, brem_trials=17)))
    new = build_case(
        replace(
            scene,
            numerics=pr.Numerics(
                sampling=pr.config.FixedSampling(cxr_electrons=31, brem_electrons=17)
            ),
        )
    )
    assert pickle.dumps(old.to_dict(), protocol=5) == pickle.dumps(new.to_dict(), protocol=5)
    assert case_content_key(old) == case_content_key(new)
    assert "adaptive_precision" not in new
    assert new.Ne == 31 and new.Ne_brem == 17


def test_adaptive_policy_matches_legacy_identity_and_brem_prefix():
    from pyrite.campaign.lowering import build_case

    scene = _scene()
    policy = _policy()
    legacy = pr.config.Precision.from_dict(policy.to_dict())
    old = build_case(replace(scene, numerics=pr.Numerics(precision=legacy)))
    new_numerics = pr.Numerics(sampling=policy)
    new = build_case(replace(scene, numerics=new_numerics))
    assert pickle.dumps(old.to_dict(), protocol=5) == pickle.dumps(new.to_dict(), protocol=5)
    assert case_content_key(old) == case_content_key(new)
    assert new.adaptive_precision == policy.to_dict()
    assert new.adaptive_precision["observables"] == ("line", "brem")
    assert policy.brem_policy().max_electrons == 60
    assert pickle.loads(pickle.dumps(new_numerics)).sampling == policy
    assert new_numerics.describe_sampling()["mode"] == "adaptive"
    assert new_numerics.describe_sampling()["inactive_legacy_fields"] == {
        "cxr_trials": 450,
        "brem_trials": 100,
    }


def test_default_and_existing_legacy_path_are_unchanged():
    numerics = pr.Numerics()
    assert numerics.sampling is None
    assert numerics.precision is None
    assert numerics.resolved_sampling == pr.config.FixedSampling()
    assert numerics.describe_sampling()["source"] == "legacy"
    legacy = pr.Numerics(precision=pr.config.Precision(0.1, 40, 80, 20))
    assert legacy.sampling is None
    assert legacy.resolved_sampling.to_dict() == legacy.precision.to_dict()
    assert legacy.describe_sampling()["source"] == "legacy"


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "5"])
def test_fixed_policy_validates_counts(value):
    with pytest.raises(ValueError, match="positive integer"):
        pr.config.FixedSampling(cxr_electrons=value)


def test_explicit_conflicting_sources_are_rejected():
    policy = _policy()
    with pytest.raises(ValueError, match="cannot combine"):
        pr.Numerics(sampling=policy, cxr_trials=7)
    with pytest.raises(ValueError, match="cannot combine"):
        pr.Numerics(sampling=pr.config.FixedSampling(31, 17), brem_trials=23)
    with pytest.raises(ValueError, match="cannot combine"):
        pr.Numerics(
            sampling=pr.config.FixedSampling(), precision=pr.config.Precision(0.1, 40, 80, 20)
        )
    with pytest.raises(ValueError, match="conflicts"):
        pr.Numerics(sampling=policy, precision=pr.config.Precision(0.2, 40, 80, 20))
    with pytest.raises(TypeError, match="sampling must be"):
        pr.Numerics(sampling={"mode": "fixed"})
    with pytest.raises(ValueError, match="per-electron or CUDA"):
        pr.Numerics(sampling=policy, transport_core="lockstep")


def test_replace_preserves_sampling_selection_and_positional_legacy():
    original = pr.Numerics(sampling=pr.config.FixedSampling(31, 17))
    assert replace(original, max_dE_frac=0.01).resolved_sampling == original.sampling
    assert pr.Numerics(31, 17).resolved_sampling == original.sampling
    adaptive = pr.Numerics(sampling=_policy())
    assert replace(adaptive, max_dE_frac=0.01).precision.to_dict() == _policy().to_dict()
