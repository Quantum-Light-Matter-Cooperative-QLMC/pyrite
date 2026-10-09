"""Runtime warning/provenance behavior for the central-ray Born screen."""

import json
import warnings

import numpy as np
import pytest

from pyrite import api
from pyrite.campaign.config import MATERIALS, material_sweep
from pyrite.campaign.kinematic_validity import (
    KinematicValidityWarning,
    case_kinematic_validity,
    warn_kinematic_validity,
)
from pyrite.campaign.sweep import BeamSpec, build_cases
from pyrite.checkpoints.persistence import _IncrementalManifest, _manifest_save
from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG


def _case(**changes):
    return {
        "name": "audit",
        "crystal": "silicon",
        "hkl_list": [(2, 2, 0)],
        "B_ang2": 0.0,
        "E0_keV": 100_000.0,
        "thickness_ang": 1e7,
        "theta_obs_rad": np.deg2rad(170),
        "tilt_deg": -5.0,
        "surface_hkl": (1, 1, 0),
        **changes,
    }


def test_warning_once_per_case_reflection_and_json_provenance(tmp_path):
    case = _case(hkl_list=[(2, 2, 0), (2, 2, 0), (-2, -2, 0)])
    with pytest.warns(KinematicValidityWarning) as emitted:
        warn_kinematic_validity(case)
    assert len(emitted) == 1
    audit = api.run_provenance(case, None)["kinematic_validity"]
    assert len(audit["reflections"]) == 2
    assert audit["reflections"][0]["extinction_reachable"]
    assert audit["reflections"][1]["status"] == "no_positive_resonance"
    json.dumps(audit, allow_nan=False)
    records = {"audit": {100_000.0: {"case": case}}}
    path = tmp_path / "checkpoint"
    _manifest_save(path, records)
    manifest = json.loads((path / "meta.json").read_text())
    assert manifest["kinematic_validity"][0]["audit"] == audit
    incremental = _IncrementalManifest()
    incremental.add(records)
    assert incremental.manifest()["kinematic_validity"] == manifest["kinematic_validity"]
    assert "kinematic_validity" not in case


def test_orientation_and_layer_thickness():
    base = _case()
    first = case_kinematic_validity(base)["reflections"][0]
    rotated = case_kinematic_validity(_case(tilt_deg=30))["reflections"][0]
    beta = np.sqrt(1 - (1 + 100_000_000 / 510_998.95) ** -2)
    g = 2 * np.pi * np.sqrt(8) / CRYSTALS["silicon"]["lattice"]["a"]
    expected = (
        HBARC_EV_ANG * beta * g * np.cos(np.deg2rad(30)) / (1 - beta * np.cos(np.deg2rad(170)))
    )
    assert rotated["photon_energy_eV"] == pytest.approx(expected)
    assert rotated["photon_energy_eV"] < first["photon_energy_eV"]
    layered = _case(
        layer_radiators=[None, base],
        abs_layers=[(0, 1e6, [("Si", 0.05)]), (1e6, 1e6 + 100, [("Si", 0.05)])],
    )
    reflection = case_kinematic_validity(layered)["reflections"][0]
    assert reflection["layer"] == 1
    assert reflection["thickness_ang"] == 100
    assert not reflection["extinction_reachable"]


def test_case_lowering_warns_for_thick_silicon():
    sweep = material_sweep(
        "silicon",
        thickness_ang=1e7,
        tilt_deg=-5,
        tilt_azim_deg=0,
        theta_obs_deg=170,
        beam=BeamSpec(energy_keV=100_000),
        beam_uvw=(1, 1, 0),
    )
    with pytest.warns(KinematicValidityWarning):
        cases = build_cases(
            sweep, inelastic_model="continuous", elastic_model="mott", bremsstrahlung_model="eedl"
        )
    assert cases
    assert "kinematic_validity" not in cases[0].to_dict()


@pytest.mark.slow
@pytest.mark.parametrize("material", MATERIALS)
def test_standard_profile_passes_screen(material):
    # Resolve all configured energies/orientations/thicknesses, no transport,
    # external tables or GPU probes. The acceptance is a catalog audit.
    sweep = material_sweep(material)
    with warnings.catch_warnings():
        warnings.simplefilter("error", KinematicValidityWarning)
        build_cases(
            sweep, inelastic_model="continuous", elastic_model="mott", bremsstrahlung_model="eedl"
        )


def test_extinction_alone_does_not_warn_off_bragg():
    case = _case(theta_obs_rad=np.deg2rad(119))
    audit = case_kinematic_validity(case)["reflections"][0]
    assert audit["extinction_reachable"] and not audit["photon_mixing"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", KinematicValidityWarning)
        warn_kinematic_validity(case)


def test_batched_optics_pair_each_reflection_with_its_resonance():
    from pyrite.materials.crystal import chi_g, crystal_absorption_length_ang

    audit = case_kinematic_validity(_case(hkl_list=[(1, 1, 1), (-1, -1, -1), (2, 2, 0), (4, 4, 0)]))
    for record in audit["reflections"]:
        if record["status"] != "evaluated":
            continue
        energy = record["photon_energy_eV"]
        # xraydb's local spline brackets depend on the energy-array span;
        # tolerate 1 ppm while rejecting row/energy mispairing by orders more.
        expected = abs(chi_g("silicon", tuple(record["hkl"]), energy, 0, True))
        assert record["chi_abs"] == pytest.approx(expected, rel=1e-6)
        expected_absorption = float(crystal_absorption_length_ang("silicon", energy))
        assert record["absorption_length_ang"] == pytest.approx(expected_absorption, rel=1e-6)


def test_unsupported_optical_energy_is_recorded():
    audit = case_kinematic_validity(_case(E0_keV=1e-6))
    assert audit["reflections"][0]["status"] == "outside_optical_data"
    json.dumps(audit, allow_nan=False)
