"""Atomic shell source validation and the explicit EEDL label join."""

from types import SimpleNamespace

import pytest
import xraydb

from pyrite.montecarlo import shell_configuration as config


def _fixture(tmp_path, rows: str):
    path = tmp_path / "pdatconf.p14"
    path.write_text(rows, encoding="ascii")
    return path


def test_atomic_shells_preserve_occupation_and_energy(tmp_path):
    path = _fixture(
        tmp_path,
        "# shell records\n"
        "2 1 K 1s1/2 2 24.59 0.535 0 0\n"
        "3 1 K 1s1/2 2 58.0 0.329 0 0\n"
        "3 2 L1 2s1/2 1 5.392 1.935 0 0\n",
    )
    shells = config.load_atomic_shells(path)
    assert tuple(shell.occupation for shell in shells[3]) == (2, 1)
    assert shells[3][1].ionization_energy_eV == pytest.approx(5.392)
    assert shells[3][1].label == "L1"


def test_atomic_shell_cache_preserves_fresh_mappings_and_changed_bytes(tmp_path):
    path = _fixture(tmp_path, "2 1 K 1s1/2 2 24.59 0.535 0 0\n")
    shells = config.load_atomic_shells(path)
    original = shells[2][0]
    shells.clear()
    assert config.load_atomic_shells(path)[2][0] == original
    path.write_text("2 1 K 1s1/2 2 25.00 0.535 0 0\n", encoding="ascii")
    assert config.load_atomic_shells(path)[2][0].ionization_energy_eV == 25.0
    path.write_text("2 1 K 1s1/2 1 25.00 0.535 0 0\n", encoding="ascii")
    with pytest.raises(ValueError, match="occupations"):
        config.load_atomic_shells(path)


def test_cached_default_shells_still_verify_checksum(tmp_path, monkeypatch):
    import hashlib

    path = _fixture(tmp_path, "2 1 K 1s1/2 2 24.59 0.535 0 0\n")
    monkeypatch.setattr(config, "_default_path", lambda: path)
    monkeypatch.setattr(config, "PDATCONF_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    assert config.load_atomic_shells()[2][0].ionization_energy_eV == 24.59
    path.write_text("2 1 K 1s1/2 2 25.00 0.535 0 0\n", encoding="ascii")
    with pytest.raises(ValueError, match="checksum mismatch"):
        config.load_atomic_shells()


@pytest.mark.parametrize(
    "rows, error",
    [
        ("3 1 K 1s1/2 2 58 0.3 0 0\n", "occupations"),
        (
            "2 1 K 1s1/2 1 20 0.3 0 0\n2 1 K 1s1/2 1 20 0.3 0 0\n",
            "duplicate",
        ),
        ("2 1 K 1s1/2 2 nan 0.3 0 0\n", "bounds"),
        (
            "2 1 K 1s1/2 1 20 0.3 0 0\n2 2 K 1s1/2 1 20 0.3 0 0\n",
            "duplicate shell label",
        ),
        ("2 1 K 2s1/2 2 20 0.3 0 0\n", "disagrees"),
    ],
)
def test_atomic_shells_fail_closed(tmp_path, rows, error):
    with pytest.raises(ValueError, match=error):
        config.load_atomic_shells(_fixture(tmp_path, rows))


def _eedl(monkeypatch, *channels):
    monkeypatch.setattr(
        config,
        "load_eedl_shell_ionization",
        lambda element: tuple(
            SimpleNamespace(shell_designator=d, binding_energy_eV=e) for d, e in channels
        ),
    )


def test_eedl_match_keeps_source_binding_difference(tmp_path, monkeypatch):
    shell = config.load_atomic_shells(_fixture(tmp_path, "2 1 K 1s1/2 2 24.59 0.535 0 0\n"))[2]
    _eedl(monkeypatch, (1, 25.0))
    match = config.match_eedl_shells("He", 2, shell)[0]
    assert match.shell == shell[0]
    assert match.eedl_labels == ("K",)
    assert match.binding_differences_eV == pytest.approx((0.41,))


def test_eedl_spin_orbit_partner_joins_filled_nl_shell(tmp_path, monkeypatch):
    shells = config.load_atomic_shells(
        _fixture(
            tmp_path,
            "5 1 K 1s1/2 2 192 0.19 0 0\n5 2 L1 2s1/2 2 11.4 1.0 0 0\n5 3 L2 2p1/2 1 8.3 0.8 0 0\n",
        )
    )[5]
    _eedl(monkeypatch, (1, 192.0), (2, 11.4), (3, 8.3), (4, 5.2))
    matches = config.match_eedl_shells("B", 5, shells)
    assert [match.eedl_labels for match in matches] == [("K",), ("L1",), ("L2", "L3")]
    assert matches[2].binding_differences_eV == pytest.approx((0.0, -3.1))


def test_eedl_designators_follow_endf_not_penelope_codes(tmp_path, monkeypatch):
    # PENELOPE code 24 is P1; ENDF inserts O8/O9, so EEDL P1 is designator 26.
    rows = "".join(
        f"55 {code} {label} {orbital} {occ} {energy} 0.1 0 0\n"
        for code, label, orbital, occ, energy in (
            (1, "K", "1s1/2", 2, 35985),
            (2, "L1", "2s1/2", 2, 5714),
            (3, "L2", "2p1/2", 2, 5359),
            (4, "L3", "2p3/2", 4, 5012),
            (5, "M1", "3s1/2", 2, 1211),
            (6, "M2", "3p1/2", 2, 1071),
            (7, "M3", "3p3/2", 4, 1003),
            (8, "M4", "3d3/2", 4, 740),
            (9, "M5", "3d5/2", 6, 726),
            (10, "N1", "4s1/2", 2, 232),
            (11, "N2", "4p1/2", 2, 172),
            (12, "N3", "4p3/2", 4, 162),
            (13, "N4", "4d3/2", 4, 79),
            (14, "N5", "4d5/2", 6, 77),
            (17, "O1", "5s1/2", 2, 23),
            (18, "O2", "5p1/2", 2, 13),
            (19, "O3", "5p3/2", 4, 12),
            (24, "P1", "6s1/2", 1, 3.9),
        )
    )
    shells = config.load_atomic_shells(_fixture(tmp_path, rows))[55]
    _eedl(monkeypatch, (26, 3.9))
    (match,) = config.match_eedl_shells("Cs", 55, shells)
    assert match.shell.designator == 24
    assert match.eedl_labels == ("P1",)


def test_eedl_shell_without_sbethe_nl_fails(tmp_path, monkeypatch):
    shell = config.load_atomic_shells(_fixture(tmp_path, "2 1 K 1s1/2 2 24.59 0.535 0 0\n"))[2]
    _eedl(monkeypatch, (2, 25.0))
    with pytest.raises(ValueError, match="L1 \\(2s\\) has no unique SBETHE"):
        config.match_eedl_shells("He", 2, shell)


def test_fetched_source_joins_every_eedl_element():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    source = config.load_atomic_shells()
    assert len(source) == 99
    assert sum(shell.occupation for shell in source[14]) == 14
    si = config.match_eedl_shells("Si", 14, source[14])
    assert [match.eedl_labels for match in si][-1] == ("M2", "M3")
    assert max(abs(d) for m in si for d in m.binding_differences_eV) < 0.01
    cs = config.match_eedl_shells("Cs", 55, source[55])
    assert cs[-1].shell.label == "P1" and cs[-1].eedl_labels == ("P1",)
    for z, shells in source.items():
        symbol = xraydb.atomic_symbol(z)
        assert config.match_eedl_shells(symbol, z, shells)
