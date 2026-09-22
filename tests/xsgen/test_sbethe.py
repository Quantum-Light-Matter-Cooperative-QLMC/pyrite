"""SBETHE input sequences, output parsing, and the material generation path.

No Fortran here: the generation test drives a fake binary. The anchor that
compiles and runs the real program lives in ``test_extern_codes.py``.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.xsgen._errors import DataFetchError
from pyrite.xsgen.sbethe import (
    SbetheDeck,
    catalog_material,
    generate_material,
    material_token,
    parse_integrated,
    parse_oscillator,
    parse_stopping,
    resolve_catalog_table,
    table_arrays,
)
from pyrite.xsgen.toolchain import Toolchain

STOPPING = """\
#  Stopping power, corrected Bethe formula.
#
#  Projectile particle: electron     rest energy = 5.10999E+05 eV
#                                         charge = -1 e
#
#  Material filename:   water
#  Electrons/molecule ........  1.00000E+01
#  Molecular weight ..........  1.80153E+01
#  Density ...................  1.00000E+00 g/cm^3
#  Mean excitation energy ....  7.50000E+01 eV
#
#   Energy          Stopping power          Without shell corr.    Stopping CS
#    (eV)        (eV/A)      (MeV/mtu)     (eV/A)      (MeV/mtu)    (eV*cm^2)
# ------------------------------------------------------------------------------
  1.00000E+03  1.13625E+00  1.13625E+02  1.18000E+00  1.18000E+02  3.39900E-15
  2.00000E+03  6.80000E-01  6.80000E+01  7.00000E-01  7.00000E+01  2.03400E-15
  1.00000E+09  2.39620E-02  2.39620E+00  2.40039E-02  2.40039E+00  7.16824E-17
"""

INTEGRATED = """\
#  Asymptotic formulas for integrated cross sections.
#
#  Material filename:   water
#  Electrons/molecule ...........  1.00000E+01
#  Molecular weight .............  1.80153E+01
#  Density ......................  1.00000E+00 g/cm^3
#
#   Energy       sigma^0      sigma^1      sigma^2        MFP       Stopping    Straggling
#    (eV)        (cm^2)      (eV*cm^2)   ((eV*cm)^2)     (mtu)      (MeV/mtu)   (MeV^2/mtu)
  1.00000E+01  2.56197E-15  1.00000E-14  1.00000E-08  1.00000E-08  1.00000E+02  1.00000E+01
  1.00000E+03  1.00000E-16  5.00000E-16  5.00000E-10  1.00000E-06  1.13625E+02  2.00000E+01
  1.00000E+09  2.14166E-18  1.02235E-16  1.67031E-09  1.39682E-05  3.41751E+00  5.58351E+01
"""

# The repeated abscissa at 5.38000E+02 is the oxygen K edge: the
# oscillator-strength density steps there, and SBETHE writes the step as two
# rows at one energy.
OSCILLATOR = """\
# Optical oscillator strength (DHFS model).
#
#  Material filename (.mat) ...  water
#  Electrons/molecule .........  1.0000E+01
#
#  f-sum ......................  1.0000E+00
#  Mean excitation energy .....  7.5000E+01 eV
#
#     W(eV)      OOS(1/eV)     cumulative
# -----------------------------------------
  1.000000E-03  1.386342E-09  0.000000E+00
  3.600000E-02  1.796670E-06  2.155961E-08
  5.380000E+02  4.000000E-04  9.000000E+00
  5.380000E+02  9.000000E-04  9.000000E+00
  2.600000E+06  1.447580E-15  1.000000E+01
"""


def test_catalog_material_derives_density_and_mean_excitation_from_number_density():
    silicon = catalog_material("silicon")

    assert silicon.key == "silicon"
    assert set(silicon.composition) == {14}
    assert silicon.density_g_cm3 == pytest.approx(2.33, rel=0.02)
    assert silicon.mean_excitation_eV == pytest.approx(173.0)
    assert silicon.band_gap_eV is None


def test_catalog_material_alias_uses_the_runnable_materials_film_crystal():
    assert catalog_material("mos2-on-sapphire").composition == catalog_material("mos2").composition


def test_catalog_material_rejects_an_unknown_key():
    with pytest.raises(ValueError, match="unknown catalog material"):
        catalog_material("unobtainium")


def test_every_runtime_catalog_target_has_a_pinned_full_domain_sbethe_table():
    keys = sorted(set(CATALOG.materials) | set(CATALOG.media))
    for key in keys:
        table = resolve_catalog_table(key)
        arrays = table.arrays()
        energy = arrays["stopping_energy_eV"]
        stopping = arrays["stopping_eV_per_angstrom"]

        assert table.manifest["code"] == "sbethe"
        assert table.manifest["quantity"] == "collision_stopping"
        assert energy[0] == pytest.approx(1.0e3)
        assert energy[-1] == pytest.approx(1.0e9)
        assert np.all(np.diff(energy) > 0.0)
        assert np.all(np.isfinite(stopping))
        assert np.all(stopping > 0.0)

    hopg = resolve_catalog_table("hopg").arrays()
    assert not np.allclose(
        hopg["stopping_eV_per_angstrom"],
        hopg["stopping_no_shell_eV_per_angstrom"],
    )


def test_the_deck_renders_the_sequence_the_program_prompts_for():
    deck = SbetheDeck(
        name="water",
        composition={1: 2, 8: 1},
        density_g_cm3=1.0,
        mean_excitation_eV=75.0,
    )
    lines = deck.render().splitlines()

    assert lines[0] == "water"
    assert lines[1] == "1"  # composition from the keyboard, never a pdcompos.pen ID
    assert lines[2] == "2"  # number of elements
    assert lines[3] == "1"  # stoichiometric formula
    assert lines[4].split()[0] == "1"
    assert lines[5].split()[0] == "8"
    assert float(lines[6]) == pytest.approx(1.0)
    assert lines[7] == "Y"  # always override the proposed I
    assert float(lines[8]) == pytest.approx(75.0)
    assert lines[9] == "N"  # conductor
    assert lines[10] == "1"  # electron
    assert len(lines) == 11


def test_a_single_element_skips_the_formula_branch():
    """The one-element branch asks only for the atomic number."""
    deck = SbetheDeck(
        name="gold", composition={79: 1}, density_g_cm3=19.3, mean_excitation_eV=790.0
    )
    lines = deck.render().splitlines()

    assert lines[2] == "1"  # number of elements
    assert lines[3] == "79"  # atomic number, with no stoichiometric-index answer
    assert float(lines[4]) == pytest.approx(19.3)


def test_an_insulator_answers_the_gap_prompt():
    conductor = SbetheDeck(
        name="al", composition={13: 1}, density_g_cm3=2.7, mean_excitation_eV=166.0
    )
    insulator = SbetheDeck(
        name="quartz",
        composition={14: 1, 8: 2},
        density_g_cm3=2.32,
        mean_excitation_eV=139.2,
        band_gap_eV=8.9,
    )

    assert conductor.render().splitlines()[-2] == "N"
    gap_lines = insulator.render().splitlines()
    assert gap_lines[-3] == "Y"
    assert float(gap_lines[-2]) == pytest.approx(8.9)


def test_the_gap_is_part_of_the_model_record():
    """A band gap changes the numbers, so it cannot be invisible to the key."""
    without = SbetheDeck(
        name="q", composition={14: 1, 8: 2}, density_g_cm3=2.32, mean_excitation_eV=139.2
    )
    with_gap = SbetheDeck(
        name="q",
        composition={14: 1, 8: 2},
        density_g_cm3=2.32,
        mean_excitation_eV=139.2,
        band_gap_eV=8.9,
    )

    assert without.model_record() != with_gap.model_record()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"composition": {}}, "at least one element"),
        ({"composition": {0: 1}}, "atomic number"),
        ({"composition": {104: 1}}, "atomic number"),
        ({"composition": {1: 0}}, "stoichiometric index"),
        ({"density_g_cm3": 0.0}, "density"),
        ({"mean_excitation_eV": 0.5}, "must exceed 1 eV"),
        ({"projectile": "neutron"}, "projectile"),
    ],
)
def test_invalid_decks_are_rejected(kwargs, message):
    base = {
        "name": "x",
        "composition": {1: 2, 8: 1},
        "density_g_cm3": 1.0,
        "mean_excitation_eV": 75.0,
    }
    with pytest.raises(ValueError, match=message):
        SbetheDeck(**{**base, **kwargs})


def test_the_material_name_is_reduced_to_what_sbethe_accepts():
    """It is both an answer with no blanks and the name of a .mat file."""
    assert material_token("liquid water") == "liquid-water"
    assert material_token("a" * 40) == "a" * 15
    assert material_token("../etc/passwd") == "etc-passwd"
    with pytest.raises(ValueError, match="no characters"):
        material_token("///")


def test_parsing_the_stopping_table_reads_columns_and_header_scalars():
    result = parse_stopping(STOPPING)

    assert result.energy_ev.shape == (3,)
    assert result.stopping_mev_cm2_per_g[0] == pytest.approx(113.625)
    assert result.stopping_no_shell_mev_cm2_per_g[0] == pytest.approx(118.0)
    assert result.stopping_cs_ev_cm2[-1] == pytest.approx(7.16824e-17)
    assert result.electrons_per_molecule == pytest.approx(10.0)
    assert result.molecular_weight_g_mol == pytest.approx(18.0153)
    assert result.density_g_cm3 == pytest.approx(1.0)
    assert result.mean_excitation_eV == pytest.approx(75.0)


def test_parsing_the_integrated_table_exposes_the_cross_section_moments():
    result = parse_integrated(INTEGRATED)

    assert result.sigma0_cm2[0] == pytest.approx(2.56197e-15)
    assert result.sigma1_ev_cm2[0] == pytest.approx(1.0e-14)
    assert result.sigma2_ev2_cm2[-1] == pytest.approx(1.67031e-09)
    assert result.straggling_mev2_cm2_per_g[-1] == pytest.approx(55.8351)


def test_the_oscillator_grid_may_repeat_an_energy_at_a_shell_edge():
    result = parse_oscillator(OSCILLATOR)

    assert result.energy_ev.shape == (5,)
    assert result.energy_ev[2] == result.energy_ev[3]
    assert result.oos_per_ev[2] != result.oos_per_ev[3]
    assert result.electrons_per_molecule == pytest.approx(10.0)


def test_a_stopping_grid_that_repeats_an_energy_is_rejected():
    """Only the oscillator abscissa is allowed to repeat."""
    doubled = STOPPING.replace(
        "  2.00000E+03  6.80000E-01",
        "  1.00000E+03  6.80000E-01",
    )
    with pytest.raises(ValueError, match="increasing"):
        parse_stopping(doubled)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        (
            lambda text: text.replace("  1.00000E+03  1.13625E+00", "  1.00000E+03  x"),
            "not numeric",
        ),
        (lambda text: text.replace("  3.39900E-15", ""), "columns"),
        (
            lambda text: text.replace("#  Mean excitation energy ....  7.50000E+01 eV", ""),
            "missing",
        ),
        (lambda text: text.replace("1.13625E+00", "-1.13625E+00"), "must be positive"),
    ],
)
def test_corrupt_stopping_output_is_rejected(corruption, message):
    with pytest.raises(ValueError, match=message):
        parse_stopping(corruption(STOPPING))


def test_table_arrays_keeps_each_output_on_its_own_grid():
    arrays = table_arrays(
        parse_stopping(STOPPING),
        parse_integrated(INTEGRATED),
        parse_oscillator(OSCILLATOR),
    )

    assert arrays["stopping_energy_eV"].shape == (3,)
    assert arrays["integrated_energy_eV"].shape == (3,)
    assert arrays["oos_energy_eV"].shape == (5,)
    # asymptotic.dat reaches below the corrected-Bethe floor that stp.dat starts at.
    assert arrays["integrated_energy_eV"][0] < arrays["stopping_energy_eV"][0]
    assert float(arrays["mean_excitation_eV"]) == pytest.approx(75.0)
    assert np.isclose(float(arrays["electrons_per_molecule"]), 10.0)


def test_outputs_that_disagree_on_the_material_are_rejected():
    """The three files describe one run; a mismatch means they do not."""
    other = OSCILLATOR.replace(
        "#  Electrons/molecule .........  1.0000E+01",
        "#  Electrons/molecule .........  2.0000E+01",
    )
    with pytest.raises(ValueError, match="electrons per molecule"):
        table_arrays(
            parse_stopping(STOPPING),
            parse_integrated(INTEGRATED),
            parse_oscillator(other),
        )


def _sbethe_tree(root):
    """Build a minimal SBETHE source tree with its fetched data directory."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "sbethe.f").write_text("      END\n", encoding="utf-8")
    (root / "sdbase").mkdir(exist_ok=True)
    return root


def _patch_build(monkeypatch, tmp_path, outputs):
    monkeypatch.setattr("pyrite.paths.user_data_path", lambda *a, **k: tmp_path / "data")
    monkeypatch.setattr("pyrite.xsgen.store.data_dir", lambda: tmp_path / "packaged")
    monkeypatch.setattr(
        "pyrite.xsgen.sbethe.generate.find_toolchain",
        lambda: Toolchain(Path("/fake/gfortran"), "GNU Fortran fake"),
    )
    monkeypatch.setattr("pyrite.xsgen.sbethe.generate.build", lambda *a, **k: Path("/fake/sbethe"))
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(outputs=dict(outputs))

    monkeypatch.setattr("pyrite.xsgen.sbethe.generate.run_program", fake_run)
    return calls


_OUTPUTS = {
    "stp.dat": STOPPING,
    "asymptotic.dat": INTEGRATED,
    "OOS.dat": OSCILLATOR,
}


def test_generate_reuses_the_content_addressed_table(monkeypatch, tmp_path):
    source = _sbethe_tree(tmp_path / "sbethe")
    calls = _patch_build(monkeypatch, tmp_path, _OUTPUTS)

    first = generate_material(
        "water", {1: 2, 8: 1}, density_g_cm3=1.0, mean_excitation_eV=75.0, source_path=source
    )
    second = generate_material(
        "water", {1: 2, 8: 1}, density_g_cm3=1.0, mean_excitation_eV=75.0, source_path=source
    )

    assert first.generated is True
    assert second.generated is False
    assert second.table.key == first.table.key
    assert len(calls) == 1
    assert first.table.arrays()["stopping_MeV_cm2_per_g"].shape == (3,)


def test_the_run_is_driven_by_the_rendered_deck(monkeypatch, tmp_path):
    source = _sbethe_tree(tmp_path / "sbethe")
    calls = _patch_build(monkeypatch, tmp_path, _OUTPUTS)

    generate_material(
        "water", {1: 2, 8: 1}, density_g_cm3=1.0, mean_excitation_eV=75.0, source_path=source
    )

    stdin = calls[0]["stdin_text"]
    assert stdin.splitlines()[0] == "water"
    assert "sdbase" in calls[0]["data_dirs"]
    assert set(calls[0]["outputs"]) == {"stp.dat", "asymptotic.dat", "OOS.dat"}


@pytest.mark.parametrize(
    "changed",
    [
        {"density_g_cm3": 1.1},
        {"mean_excitation_eV": 78.0},
        {"composition": {1: 2, 8: 2}},
        {"band_gap_eV": 8.9},
    ],
)
def test_a_different_material_or_deck_is_a_different_table(monkeypatch, tmp_path, changed):
    """Anything that changes the numbers has to change the key."""
    source = _sbethe_tree(tmp_path / "sbethe")
    _patch_build(monkeypatch, tmp_path, _OUTPUTS)
    base = {
        "composition": {1: 2, 8: 1},
        "density_g_cm3": 1.0,
        "mean_excitation_eV": 75.0,
        "source_path": source,
    }

    first = generate_material("water", **base)
    other = generate_material("water", **{**base, **changed})

    assert other.table.key != first.table.key


def test_absent_reference_data_names_the_command_that_installs_it(monkeypatch, tmp_path):
    source = tmp_path / "sbethe"
    source.mkdir(parents=True)
    (source / "sbethe.f").write_text("      END\n", encoding="utf-8")
    _patch_build(monkeypatch, tmp_path, _OUTPUTS)

    with pytest.raises(DataFetchError, match="pyrite tables fetch sbethe"):
        generate_material(
            "water", {1: 2, 8: 1}, density_g_cm3=1.0, mean_excitation_eV=75.0, source_path=source
        )
