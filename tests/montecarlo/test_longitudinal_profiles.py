from dataclasses import replace

import numpy as np
import pytest

from pyrite.campaign.config import material_sweep
from pyrite.campaign.longitudinal import (
    H_EV_FS,
    LongitudinalDistribution,
)
from pyrite.campaign.profiles import dataset_identity
from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.montecarlo.transport import C_ANG_PER_FS, _sample_bunch_offsets
from pyrite.results import Settings


def _campaign_cases(material: str, policy: LongitudinalDistribution):
    return build_cases(
        Sweep(
            material=material,
            beam=BeamSpec(
                energy_keV=(30.0, 100.0),
                transverse_fwhm_x_mm=0.1,
                transverse_fwhm_y_mm=0.1,
                bunch_charge_pc=1.0,
                rep_rate_hz=5000.0,
                longitudinal=policy,
            ),
            thickness_ang=(1000.0, 5000.0, 10000.0, 40000.0),
            tilt_deg=45.0,
            tilt_azim_deg=180.0,
        )
    )


@pytest.mark.parametrize(
    ("material", "energy_keV", "expected_energy_eV"),
    [
        ("hopg", 30.0, 857.9574287),
        ("hopg", 100.0, 1432.3516594),
        ("hbn", 30.0, 864.3975835),
        ("hbn", 100.0, 1443.1034359),
    ],
)
def test_target_timing_uses_pinned_basal_reflection_and_h_over_e(
    material, energy_keV, expected_energy_eV
):
    policy = LongitudinalDistribution("microtrain", envelope_rms_fs=200.0)
    case = next(case for case in _campaign_cases(material, policy) if case["E0_keV"] == energy_keV)
    resolved = case["longitudinal_distribution"]

    assert resolved["target_material"] == material
    assert resolved["target_crystal"] == material
    assert resolved["target_reflection"] == (0, 0, 2)
    assert resolved["target_energy_eV"] == pytest.approx(expected_energy_eV, rel=1e-9)
    assert resolved["target_period_fs"] == pytest.approx(H_EV_FS / resolved["target_energy_eV"])
    omega = 2.0 * np.pi / resolved["target_period_fs"]
    assert np.exp(-((omega * resolved["microbunch_rms_fs"]) ** 2)) == pytest.approx(0.9)
    assert resolved["spacing_fs"] == pytest.approx(resolved["target_period_fs"])
    assert "catalog-pinned" in resolved["provenance"]


def test_three_campaign_policies_have_16_cases_and_only_longitudinal_differences():
    policies = (
        LongitudinalDistribution("gaussian", envelope_rms_fs=200.0),
        LongitudinalDistribution("microtrain", envelope_rms_fs=200.0),
        LongitudinalDistribution("compressed"),
    )
    campaigns = [
        [case for material in ("hopg", "hbn") for case in _campaign_cases(material, policy)]
        for policy in policies
    ]

    assert [len(cases) for cases in campaigns] == [16, 16, 16]
    for group in zip(*campaigns, strict=True):
        common = [
            {key: value for key, value in case.items() if key != "longitudinal_distribution"}
            for case in group
        ]
        assert common[0] == common[1] == common[2]
        gaussian, train, compressed = (case["longitudinal_distribution"] for case in group)
        assert gaussian["rms_duration_fs"] == 200.0
        assert train["envelope_rms_fs"] == 200.0
        assert compressed["rms_duration_fs"] == pytest.approx(train["microbunch_rms_fs"])


def test_bundled_campaign_profiles_resolve_exact_shared_beam_and_case_grid(lab_catalog):
    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    names = (
        "hopg_hbn_gaussian_200fs",
        "hopg_hbn_microtrain_200fs",
        "hopg_hbn_compressed_microbunch",
    )
    assert all(lab_catalog.profile_materials(name) == ("hopg", "hbn") for name in names)
    for name in names:
        beam = material_sweep("hopg", profile=name).beam
        assert beam.transverse_fwhm_x_mm == 0.1
        assert beam.transverse_fwhm_y_mm == 0.1
        assert beam.bunch_charge_pc == 1.0
        assert beam.rep_rate_hz == 5000.0

    campaigns = [
        [
            case
            for material in ("hopg", "hbn")
            for case in build_cases(material_sweep(material, profile=name))
        ]
        for name in names
    ]
    assert [len(cases) for cases in campaigns] == [16, 16, 16]
    for group in zip(*campaigns, strict=True):
        common = [
            plain({key: value for key, value in case.items() if key != "longitudinal_distribution"})
            for case in group
        ]
        assert common[0] == common[1] == common[2]
        for case in group:
            assert case["beam_fwhm_mm"] == 0.1


def test_microtrain_sampling_is_deterministic_centered_and_has_target_bunching():
    policy = LongitudinalDistribution("microtrain", envelope_rms_fs=200.0)
    resolved = _campaign_cases("hopg", policy)[0]["longitudinal_distribution"]
    kwargs = dict(
        Ne=100_000,
        bunch_length_fs=None,
        long_shape="gaussian",
        long_offsets_fs=None,
        seed=314159,
        longitudinal_distribution=resolved,
    )

    first = _sample_bunch_offsets(**kwargs)
    second = _sample_bunch_offsets(**kwargs)
    dt_fs = first / C_ANG_PER_FS
    omega = 2.0 * np.pi / resolved["target_period_fs"]

    np.testing.assert_array_equal(first, second)
    assert dt_fs.mean() == pytest.approx(0.0, abs=1e-12)
    assert dt_fs.std() == pytest.approx(200.0, rel=0.01)
    bunching = abs(np.mean(np.exp(1j * omega * dt_fs))) ** 2
    assert bunching == pytest.approx(0.9, abs=0.01)


def test_longitudinal_policy_changes_identity_but_not_macro_particle_definition():
    base = Sweep(
        material="hopg",
        beam=BeamSpec(
            energy_keV=(30.0,),
            longitudinal=LongitudinalDistribution("microtrain", envelope_rms_fs=200.0),
        ),
        thickness_ang=(1000.0,),
        tilt_deg=45.0,
        tilt_azim_deg=180.0,
    )
    changed = replace(
        base,
        beam=replace(
            base.beam,
            longitudinal=replace(base.beam.longitudinal, retained_coherence=0.8),
        ),
    )
    settings = Settings(n_electrons=20, n_electrons_brem=10)

    identity = dataset_identity("hopg", "full", settings, base)
    changed_identity = dataset_identity("hopg", "full", settings, changed)

    assert identity["parameter_sha256"] != changed_identity["parameter_sha256"]
    assert "longitudinal_resolutions" in identity["resolved_parameters"]
    assert "Ne" not in identity["resolved_parameters"]["sweep"]["longitudinal"]


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (LongitudinalDistribution("gaussian", envelope_rms_fs=200.0), None),
        (None, "requires envelope_rms_fs"),
    ],
)
def test_policy_validation_smoke(policy, message):
    if message is None:
        assert policy is not None
        return
    with pytest.raises(ValueError, match=message):
        LongitudinalDistribution("microtrain")
