from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from pyrite.paths import data_dir
from pyrite.xsgen.elsepa import (
    ElsepaDeck,
    generate_element,
    output_name,
    parse_dcs,
    table_arrays,
)
from pyrite.xsgen.toolchain import Toolchain


def _reference_output() -> bytes:
    return (data_dir() / "xsgen" / "elsepa" / "test-run-output" / "dcs_1p000e03.dat").read_bytes()


def test_free_atom_deck_is_explicit_and_predicts_fixed_output_names():
    deck = ElsepaDeck.free_atom(79, [100.0, 1_000.0])

    assert "IZ     79\n" in deck.render()
    assert "NELEC  79\n" in deck.render()
    assert "MUFFIN 0\n" in deck.render()
    assert deck.output_names == ("dcs_1p000e02.dat", "dcs_1p000e03.dat")
    assert deck.model_record()["energies_eV"] == [100.0, 1_000.0]


@pytest.mark.parametrize("energy", [0.0, 4.998, float("nan"), float("inf")])
def test_deck_rejects_energies_elsepa_cannot_run(energy):
    with pytest.raises(ValueError, match="energy"):
        ElsepaDeck.free_atom(79, [energy])


def test_deck_rejects_output_filename_collisions():
    with pytest.raises(ValueError, match="collide"):
        ElsepaDeck.free_atom(79, [1_000.0, 1_000.1])


def test_deck_rejects_a_fractional_atomic_number():
    with pytest.raises(ValueError, match="atomic number"):
        ElsepaDeck.free_atom(cast(Any, 79.5), [1_000.0])


def test_output_name_matches_the_fortran_format():
    assert output_name(1.0e3) == "dcs_1p000e03.dat"
    assert output_name(2.34567e6) == "dcs_2p345e06.dat"


def test_parser_reads_the_committed_vendor_reference_output():
    result = parse_dcs(_reference_output())

    assert result.energy_ev == pytest.approx(1_000.0)
    assert result.total_elastic_cm2 == pytest.approx(2.47469e-16)
    assert result.transport1_cm2 == pytest.approx(5.54005e-17)
    assert result.transport2_cm2 == pytest.approx(5.07959e-17)
    assert result.absorption_cm2 == pytest.approx(9.36109e-17)
    assert result.theta_deg.shape == (606,)
    assert result.theta_deg[[0, -1]].tolist() == [0.0, 180.0]
    assert result.mu[[0, -1]].tolist() == [0.0, 1.0]
    assert np.all(result.dcs_cm2_sr > 0)


def test_table_builds_the_normalized_cdf_on_the_native_grid():
    panel = parse_dcs(_reference_output())
    arrays = table_arrays([panel], energies_ev=[1_000.0])

    cdf = arrays["angular_cdf"][0]
    assert cdf[[0, -1]].tolist() == [0.0, 1.0]
    assert np.all(np.diff(cdf) >= 0)
    angular_integral = 4.0 * np.pi * np.trapezoid(panel.dcs_cm2_sr, panel.mu)
    assert angular_integral == pytest.approx(panel.total_elastic_cm2, rel=3e-4)


def test_parser_rejects_a_truncated_non_table():
    with pytest.raises(ValueError, match="fewer than two"):
        parse_dcs("# Kinetic energy = 1.0E3 eV\n")


def test_generate_reuses_the_content_addressed_table(monkeypatch, tmp_path):
    source = tmp_path / "elsepa"
    (source / "database").mkdir(parents=True)
    for name in ("elscata.f", "elsepa2020.f", "radial.f"):
        (source / name).write_text(name, encoding="utf-8")
    monkeypatch.setattr("pyrite.paths.user_data_path", lambda *a, **k: tmp_path / "data")
    monkeypatch.setattr("pyrite.xsgen.store.data_dir", lambda: tmp_path / "packaged")
    monkeypatch.setattr(
        "pyrite.xsgen.elsepa.generate.find_toolchain",
        lambda: Toolchain(Path("/fake/gfortran"), "GNU Fortran fake"),
    )
    monkeypatch.setattr("pyrite.xsgen.elsepa.generate.build", lambda *a, **k: Path("/fake/elscata"))
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(outputs={"dcs_1p000e03.dat": _reference_output()})

    monkeypatch.setattr("pyrite.xsgen.elsepa.generate.run_program", fake_run)

    first = generate_element(80, [1_000.0], source_path=source)
    second = generate_element(80, [1_000.0], source_path=source)

    assert first.generated is True
    assert second.generated is False
    assert second.table.key == first.table.key
    assert len(calls) == 1
    assert first.table.arrays()["dcs_cm2_sr"].shape == (1, 606)
