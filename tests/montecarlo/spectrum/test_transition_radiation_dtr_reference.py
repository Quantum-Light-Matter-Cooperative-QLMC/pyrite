"""Alternating-stack diagnostic reference, its interference and provenance."""

import json

import numpy as np
import pytest

from pyrite.materials.crystal import CRYSTALS, reciprocal_g_vector
from pyrite.validation.dtr_reference import REFERENCE_PROFILE, main, reference_spectrum


def test_reference_resolves_thin_alternating_stack_and_signed_accounting():
    columns, meta = reference_spectrum()
    config = meta["config"]
    assert [layer["material"] for layer in config["layers"]] == ["sio2", "mos2", "sio2"]
    assert [layer["thickness_ang"] for layer in config["layers"]] == [50.0, 100.0, 50.0]
    assert config["lattice_origin_ang"] == [0.0, 0.0, 50.0]
    assert len(columns["energy_eV"]) == config["energy_nodes"] == 1201
    assert meta["center_eV"] == pytest.approx(1008.53, abs=0.02)
    assert 0.25 < meta["abs_chi0"] / meta["gamma_inverse_squared"] < 4
    assert meta["max_born_coupling"] < 0.1
    assert meta["dtr_on_off_peak_fraction"] > 0.1
    assert np.max(np.abs(columns["pxr_dtr_cross"])) > 0
    np.testing.assert_allclose(
        columns["line"],
        columns["pxr"] + columns["dtr"] + columns["pxr_dtr_cross"],
        rtol=1e-12,
        atol=1e-20,
    )
    np.testing.assert_allclose(
        columns["total"],
        columns["line"] + columns["tr"] + columns["tr_line_cross"],
        rtol=1e-12,
        atol=1e-20,
    )
    again, repeated = reference_spectrum()
    assert meta["resolved_sha256"] == repeated["resolved_sha256"]
    for name in columns:
        np.testing.assert_array_equal(columns[name], again[name])


def test_reference_local_window_is_resolved(tmp_path):
    fine, _ = reference_spectrum()
    profile = tmp_path / "coarse.toml"
    profile.write_text(
        REFERENCE_PROFILE.read_text().replace("energy_nodes = 1201", "energy_nodes = 601")
    )
    coarse, _ = reference_spectrum(profile)
    for component in ("line", "pxr", "dtr", "total"):
        fine_integral = np.trapezoid(fine[component], fine["energy_eV"])
        coarse_integral = np.trapezoid(coarse[component], coarse["energy_eV"])
        assert coarse_integral == pytest.approx(fine_integral, rel=1e-5)


def test_reference_termination_changes_tr_cross_term_and_identity(tmp_path):
    original, meta = reference_spectrum()
    _, gmag = reciprocal_g_vector((0, 0, 2), CRYSTALS["mos2"]["lattice"])
    # Half a lattice period flips the diffracted field, leaving its self term.
    profile = tmp_path / "termination.toml"
    profile.write_text(
        REFERENCE_PROFILE.read_text().replace(
            "lattice_origin_ang = [0.0, 0.0, 50.0]",
            f"lattice_origin_ang = [0.0, 0.0, {50 + np.pi / gmag}]",
        )
    )
    shifted, changed = reference_spectrum(profile)
    assert changed["resolved_sha256"] != meta["resolved_sha256"]
    for component in ("pxr", "dtr", "line", "pxr_dtr_cross"):
        np.testing.assert_allclose(shifted[component], original[component], rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(
        shifted["tr_line_cross"], -original["tr_line_cross"], rtol=1e-12, atol=1e-20
    )


def test_reference_writes_inspectable_artifacts(tmp_path, capsys):
    main(["--output-dir", str(tmp_path)])
    result = json.loads(capsys.readouterr().out)
    assert result["output_dir"] == str(tmp_path)
    assert result["validation_status"] == "unverified"
    assert result["calculation_seconds"] > 0
    assert (tmp_path / "spectrum.png").read_bytes().startswith(b"\x89PNG")
    saved = json.loads((tmp_path / "reference.json").read_text())
    assert saved["resolved_sha256"] == result["resolved_sha256"]
    table = np.genfromtxt(tmp_path / "spectrum.csv", delimiter=",", names=True)
    np.testing.assert_allclose(
        table["line"], table["pxr"] + table["dtr"] + table["pxr_dtr_cross"], atol=1e-20
    )


def test_two_beam_reference_accounting_and_identity(tmp_path):
    profile = tmp_path / "small.toml"
    profile.write_text(
        REFERENCE_PROFILE.read_text().replace("energy_nodes = 1201", "energy_nodes = 61")
    )
    born, original = reference_spectrum(profile)
    columns, meta = reference_spectrum(profile, two_beam=True)
    assert meta["validation_status"] == "unverified"
    assert meta["model"] != original["model"]
    assert meta["resolved_sha256"] != original["resolved_sha256"]
    assert meta["profile_sha256"] == original["profile_sha256"]
    assert 0 < meta["two_beam_born_peak_fraction"] < 0.01
    assert "including diffraction extinction" in meta["two_beam_accounting"]
    for name, value in born.items():
        np.testing.assert_array_equal(columns[name], value)
    np.testing.assert_allclose(
        columns["two_beam_total"],
        columns["two_beam_transition"] + columns["two_beam_line"] + columns["two_beam_cross"],
        rtol=1e-12,
        atol=1e-20,
    )
    np.testing.assert_array_equal(
        columns["two_beam_minus_born_total"], columns["two_beam_total"] - columns["total"]
    )
    again, repeated = reference_spectrum(profile, two_beam=True)
    assert meta["resolved_sha256"] == repeated["resolved_sha256"]
    for name in columns:
        np.testing.assert_array_equal(columns[name], again[name])


def test_two_beam_reference_writes_comparison_artifacts(tmp_path, capsys):
    profile = tmp_path / "small.toml"
    profile.write_text(
        REFERENCE_PROFILE.read_text().replace("energy_nodes = 1201", "energy_nodes = 61")
    )
    output = tmp_path / "comparison"
    main(["--profile", str(profile), "--two-beam", "--output-dir", str(output)])
    result = json.loads(capsys.readouterr().out)
    saved = json.loads((output / "reference.json").read_text())
    assert result["resolved_sha256"] == saved["resolved_sha256"]
    assert result["two_beam_born_peak_fraction"] > 0
    assert (output / "spectrum.png").read_bytes().startswith(b"\x89PNG")
    table = np.genfromtxt(output / "spectrum.csv", delimiter=",", names=True)
    assert len(table) == 61
    np.testing.assert_allclose(
        table["two_beam_total"],
        table["two_beam_transition"] + table["two_beam_line"] + table["two_beam_cross"],
        atol=1e-20,
    )


def test_two_beam_thick_reference_departs_from_born(tmp_path):
    profile = tmp_path / "thick.toml"
    profile.write_text(
        REFERENCE_PROFILE.read_text()
        .replace("energy_nodes = 1201", "energy_nodes = 61")
        .replace("thickness_ang = 100.0", "thickness_ang = 10000.0")
    )
    columns, meta = reference_spectrum(profile, two_beam=True)
    assert meta["max_born_coupling"] > 1
    assert meta["two_beam_born_peak_fraction"] > 0.1
    assert all(np.all(np.isfinite(value)) for value in columns.values())
    assert np.all(columns["two_beam_total"] >= 0)
