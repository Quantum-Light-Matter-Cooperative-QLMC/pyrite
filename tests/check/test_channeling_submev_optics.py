"""Deterministic checks of the sub-MeV optics record (#585)."""

import pytest

from checks import channeling_submev_optics as opt


def test_escape_limits():
    thin = opt.escape(1e9)
    assert all(v == pytest.approx(1.0, rel=1e-6) for v in thin.values())
    thick = opt.escape(1.0)
    # t >> L: mean transmission tends to L / t.
    assert thick["2000nm"] == pytest.approx(1.0 / 2000.0, rel=1e-6)


def test_medium_line_matches_constant_index(monkeypatch):
    delta = 1e-3
    monkeypatch.setattr(opt, "medium", lambda material, energy: {"delta": delta})
    beta = 0.5
    vacuum = 400.0
    gap = vacuum * (1.0 - beta)
    line = opt.medium_line("tungsten", vacuum, beta)
    assert line["line_eV"] == pytest.approx(gap / (1.0 - beta * (1.0 - delta)), rel=1e-12)
    assert line["shift_eV"] < 0.0


def test_medium_line_refuses_strong_refraction(monkeypatch):
    monkeypatch.setattr(opt, "medium", lambda material, energy: {"delta": 0.5})
    assert opt.medium_line("silicon", 20.0, 0.9) is None


def test_tungsten_line_record():
    row = next(r for r in opt.LINES if r[0] == "tungsten" and r[1] == "<001>" and r[3] == 30)
    end = opt.evaluate(row)["ends"][0]
    assert -3.0 < end["medium_line"]["shift_eV"] < -1.0
    assert not end["flags"]["tables_disagree"]
    assert not end["flags"]["below_3_plasma"]
    assert 20.0 < end["L_abs_nm"] < 60.0


def test_no_medium_line_below_three_plasma_energies():
    # Si(111) at 500 keV converges under the cap, but lies below 3 hbar omega_p.
    row = next(r for r in opt.LINES if r[0] == "silicon" and r[1] == "(111)" and r[3] == 500)
    for end in opt.evaluate(row)["ends"]:
        assert end["flags"]["below_3_plasma"]
        assert end["refraction_parameter"] < opt.MAX_REFRACTION
        assert end["medium_line"] is None
