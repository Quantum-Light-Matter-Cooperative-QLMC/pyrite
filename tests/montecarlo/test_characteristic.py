"""EEDL characteristic-radiation parser, units, and runner integration."""

import hashlib

import numpy as np
import pytest
import xraydb

from pyrite._backend import REAL, xp
from pyrite._spectral_components import line_spectrum
from pyrite.montecarlo.spectrum import characteristic
from tests.helpers import scaled_rtol


def _zero_mu(_composition, energy):
    """Transparent stand-in for ``_mu_total_inv_ang``.

    The real coefficient returns on the device of its input, so the stub has to
    as well: a NumPy-returning lambda blows up under ``PYRITE_TEST_BACKEND=cuda``
    the moment it is handed a CuPy energy array.
    """
    return xp.zeros_like(xp.asarray(energy, dtype=REAL))


def _carbon_segments(lengths: list[float]) -> dict[str, object]:
    lengths_array = np.asarray(lengths, dtype=float)
    z_mid = np.cumsum(lengths_array) - 0.5 * lengths_array
    return {
        "r_mid": np.column_stack((np.zeros_like(z_mid), np.zeros_like(z_mid), z_mid)),
        "v_hat": np.tile([0.0, 0.0, 1.0], (lengths_array.size, 1)),
        "L_ang": lengths_array,
        "E_keV": np.full(lengths_array.size, 30.0),
        "elec_id": np.zeros(lengths_array.size, dtype=int),
        "layer": np.zeros(lengths_array.size, dtype=int),
        "Ne": 1,
        "thickness_ang": float(lengths_array.sum()),
    }


def test_packaged_eedl_bytes_match_pinned_checksum():
    """Guard the pin against line-ending renormalization.

    Upstream EEDL ships 75-column CRLF records. The repository normalizes text
    to LF (`.gitattributes`), which rewrites these bytes and breaks the pin
    unless the file stays marked `-text`, so assert the shipped bytes directly
    rather than only through the loader.
    """
    path = characteristic.CHARACTERISTIC_DATA_DIR / characteristic.CHARACTERISTIC_EEDL_FILENAME
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert digest == characteristic.CHARACTERISTIC_EEDL_SHA256
    with path.open("rb") as stream:
        assert stream.readline().endswith(b"\r\n"), "CRLF lost to eol normalization"


def test_packaged_eadl_bytes_match_published_file():
    """Keep the cascade's source file byte-for-byte reproducible."""
    path = characteristic.CHARACTERISTIC_DATA_DIR / characteristic.CHARACTERISTIC_EADL_FILENAME

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "78ccf8a4e07c1c120a2e3d94ff051aab2180d151f35e8bc3406d52df5af5e88c"
    )
    assert (
        characteristic.CHARACTERISTIC_EADL_SHA256 == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with path.open("rb") as stream:
        assert stream.readline().endswith(b"\r\n"), "CRLF lost to eol normalization"


def test_packaged_carbon_eedl_values_and_relaxation_join():
    table = characteristic.load_characteristic_cross_sections("C")

    assert table.atomic_number == 6
    assert table.ionization_shell_labels == ("K",)
    assert table.shell_binding_energy_eV[0] == 288.0
    sigma_k_30kev = np.interp(
        30_000.0,
        table.projectile_energy_eV_by_shell[0],
        table.ionization_cross_sections_cm2_by_shell[0],
    )
    # 2025 EEDL MF=23/MT=534, linearly interpolated at 30 keV. Every check
    # here passes ``atol=0.0``: the default 1e-8 absolute tolerance dwarfs a
    # ~5.4e-20 cm^2 cross section, so even zero would satisfy it.
    assert np.isclose(sigma_k_30kev, 5.4137330932220697e-20, rtol=1.0e-13, atol=0.0)
    assert np.isclose(table.line_yield_per_vacancy[0].sum(), 0.001682088, rtol=1e-12, atol=0.0)
    assert set(table.line_energy_eV) == {277.0}
    ka1 = table.line_labels.index("Ka1")
    # A Lorentzian transition width is the sum of the initial- and final-hole
    # widths: C K (0.0868 eV) + C L3 (0.0045 eV).
    assert np.isclose(table.line_fwhm_eV[ka1], 0.0913, atol=0.0)
    assert not table.shell_binding_energy_eV.flags.writeable
    assert not table.line_fwhm_eV.flags.writeable


def test_host_shell_ionization_rates_do_not_require_relaxation(monkeypatch):
    def unexpected_relaxation(*_args, **_kwargs):
        raise AssertionError("host shell rates must not consult xraydb")

    monkeypatch.setattr(characteristic.xraydb, "xray_edge", unexpected_relaxation)
    shells = characteristic.load_eedl_shell_ionization("C")

    assert [shell.shell_designator for shell in shells[:4]] == [1, 2, 3, 4]
    assert shells[0].binding_energy_eV == 288.0
    sigma_k = np.interp(30_000.0, shells[0].projectile_energy_eV, shells[0].cross_section_cm2)
    assert np.isclose(sigma_k, 5.4137330932220697e-20, rtol=1e-13, atol=0.0)
    assert not shells[0].cross_section_cm2.flags.writeable


def test_characteristic_single_track_matches_n_l_sigma_omega_over_four_pi(monkeypatch):
    monkeypatch.setattr(characteristic, "_mu_total_inv_ang", _zero_mu)
    energy = np.arange(250.0, 291.0, 1.0)
    density_ang3 = 0.1
    length_ang = 100.0

    spectrum = characteristic.mc_characteristic_spectrum(
        _carbon_segments([length_ang]),
        energy,
        composition=[("C", density_ang3)],
        electron_limit=1,
    )

    table = characteristic.load_characteristic_cross_sections("C")
    sigma_k = np.interp(
        30_000.0,
        table.projectile_energy_eV_by_shell[0],
        table.ionization_cross_sections_cm2_by_shell[0],
    )
    edges, _widths = characteristic._energy_bin_edges_and_widths(energy)
    captured_yield = sum(
        table.line_yield_per_vacancy[0, line_index]
        * characteristic._lorentzian_bin_weights(edges, line_energy, line_fwhm).sum()
        for line_index, (line_energy, line_fwhm) in enumerate(
            zip(table.line_energy_eV, table.line_fwhm_eV, strict=True)
        )
        if line_energy > table.recommended_cutoff_eV
    )
    # Each source branch is multiplied by the independently evaluated physical
    # Lorentzian mass captured by this finite window. Branch intensities and
    # omitted tails are deliberately not renormalized.
    expected = (
        density_ang3 * 1.0e24 * length_ang * 1.0e-8 * sigma_k * captured_yield / (4.0 * np.pi)
    )
    # Uniform 1 eV bins: the density sum is the bin-integrated photon yield.
    # The 41-bin reduction runs at the backend's REAL, so fp32 justifies the
    # bin count as the amplification; fp64 keeps the original bound.
    assert np.isclose(
        spectrum.sum(),
        expected,
        rtol=scaled_rtol(2.0e-13, eps_multiple=energy.size),
        atol=0.0,
    )


def test_constant_energy_segment_subdivision_preserves_characteristic_yield(monkeypatch):
    monkeypatch.setattr(characteristic, "_mu_total_inv_ang", _zero_mu)
    energy = np.arange(250.0, 291.0, 1.0)
    common = dict(E_grid_eV=energy, composition=[("C", 0.1)], electron_limit=1)

    whole = characteristic.mc_characteristic_spectrum(_carbon_segments([100.0]), **common)
    split = characteristic.mc_characteristic_spectrum(_carbon_segments([50.0, 50.0]), **common)

    np.testing.assert_allclose(split, whole, rtol=2.0e-13, atol=0.0)


def test_lorentzian_bin_weights_preserve_physical_window_mass():
    narrow_edges = np.array([9.0, 10.0, 11.0])
    wide_edges = np.array([8.0, 9.0, 10.0, 11.0, 12.0])

    narrow = characteristic._lorentzian_bin_weights(narrow_edges, 10.0, 2.0)
    wide = characteristic._lorentzian_bin_weights(wide_edges, 10.0, 2.0)

    np.testing.assert_allclose(narrow, [0.25, 0.25], rtol=0.0, atol=1.0e-15)
    np.testing.assert_array_equal(narrow, wide[1:3])
    assert narrow.sum() == 0.5
    assert narrow.sum() < wide.sum() < 1.0


def test_characteristic_line_is_bin_integrated_physically_truncated_lorentzian(monkeypatch):
    monkeypatch.setattr(characteristic, "_mu_total_inv_ang", _zero_mu)
    energy = np.arange(275.0, 279.0001, 0.02)
    density_ang3 = 0.1
    length_ang = 100.0

    spectrum = characteristic.mc_characteristic_spectrum(
        _carbon_segments([length_ang]),
        energy,
        composition=[("C", density_ang3)],
        electron_limit=1,
    )

    table = characteristic.load_characteristic_cross_sections("C")
    sigma_k = np.interp(
        30_000.0,
        table.projectile_energy_eV_by_shell[0],
        table.ionization_cross_sections_cm2_by_shell[0],
    )
    edges, widths = characteristic._energy_bin_edges_and_widths(energy)
    captured_yield = sum(
        table.line_yield_per_vacancy[0, line_index]
        * characteristic._lorentzian_bin_weights(edges, line_energy, line_fwhm).sum()
        for line_index, (line_energy, line_fwhm) in enumerate(
            zip(table.line_energy_eV, table.line_fwhm_eV, strict=True)
        )
        if line_energy > table.recommended_cutoff_eV
    )
    expected = (
        density_ang3 * 1.0e24 * length_ang * 1.0e-8 * sigma_k * captured_yield / (4.0 * np.pi)
    )

    assert np.count_nonzero(spectrum) == spectrum.size
    # Same reduction-length argument as the 1 eV-bin case above, over the 201
    # bins of this finer grid.
    assert np.isclose(
        np.sum(spectrum * widths),
        expected,
        rtol=scaled_rtol(2.0e-13, eps_multiple=energy.size),
        atol=0.0,
    )
    peak = int(np.argmax(spectrum))
    # Mirrored bins differ only by the rounding of equal bin weights, so a few
    # ulps of REAL is the whole budget here.
    np.testing.assert_allclose(
        spectrum[peak - 5 : peak],
        spectrum[peak + 1 : peak + 6][::-1],
        rtol=scaled_rtol(1.0e-9, eps_multiple=4.0),
        atol=0.0,
    )


def test_off_grid_line_contributes_only_its_physical_tail(monkeypatch):
    monkeypatch.setattr(characteristic, "_mu_total_inv_ang", _zero_mu)
    energy = np.arange(280.0, 284.0001, 0.02)

    spectrum = characteristic.mc_characteristic_spectrum(
        _carbon_segments([100.0]),
        energy,
        composition=[("C", 0.1)],
        electron_limit=1,
    )

    table = characteristic.load_characteristic_cross_sections("C")
    assert np.all((table.line_energy_eV < energy[0]) | (table.line_energy_eV > energy[-1]))
    assert np.all(spectrum > 0.0)
    edges, widths = characteristic._energy_bin_edges_and_widths(energy)
    wide_edges = np.array([250.0, 300.0])
    narrow_mass = characteristic._lorentzian_bin_weights(
        edges, table.line_energy_eV[-1], table.line_fwhm_eV[-1]
    ).sum()
    wide_mass = characteristic._lorentzian_bin_weights(
        wide_edges, table.line_energy_eV[-1], table.line_fwhm_eV[-1]
    ).sum()
    assert 0.0 < narrow_mass < wide_mass < 1.0
    assert np.sum(spectrum * widths) > 0.0


def test_energy_bin_edges_clamp_negative_first_edge_to_physical_floor():
    # First spacing (99) exceeds the start (1.0), so the mirrored reflection
    # 1.0 - 0.5*99 = -48.5 would be an unphysical negative photon energy.
    grid = np.array([1.0, 100.0, 101.0])

    edges, widths = characteristic._energy_bin_edges_and_widths(grid)

    assert edges[0] == 0.0
    assert widths[0] == edges[1] - 0.0
    np.testing.assert_array_equal(edges[1:], [50.5, 100.5, 101.5])
    assert np.all(widths > 0.0)


def test_energy_bin_edges_leave_ordinary_first_edge_unclamped():
    grid = np.arange(250.0, 291.0, 1.0)

    edges, _widths = characteristic._energy_bin_edges_and_widths(grid)

    assert edges[0] == grid[0] - 0.5 * (grid[1] - grid[0])
    assert edges[0] > 0.0


def test_characteristic_line_window_mass_conserves_captured_plus_truncated():
    energy = np.arange(275.0, 279.0001, 0.02)

    report = characteristic.characteristic_line_window_mass(energy, "C")

    table = characteristic.load_characteristic_cross_sections("C")
    physically_relevant = {
        label
        for label, line_energy in zip(table.line_labels, table.line_energy_eV, strict=True)
        if line_energy > table.recommended_cutoff_eV
    }
    assert set(report) == physically_relevant
    for captured, truncated in report.values():
        assert 0.0 <= captured <= 1.0
        assert np.isclose(captured + truncated, 1.0, rtol=0.0, atol=1.0e-12)
    # The K-alpha lines sit inside this narrow window: most of their mass is
    # captured, not truncated.
    ka1_captured, ka1_truncated = report["Ka1"]
    assert ka1_captured > 0.9
    assert ka1_truncated < 0.1


def test_characteristic_line_window_mass_reports_full_truncation_off_grid():
    # Far below every carbon line: captured mass rounds to zero and the
    # truncated tail is (numerically) the whole line, i.e. nothing is
    # silently redistributed into a window that cannot see the line at all.
    report = characteristic.characteristic_line_window_mass(np.array([1.0, 2.0]), "C")

    for captured, truncated in report.values():
        assert captured == pytest.approx(0.0, abs=1.0e-4)
        assert truncated == pytest.approx(1.0, abs=1.0e-4)


def test_severely_truncated_window_warns_instead_of_silently_dropping_mass(monkeypatch):
    monkeypatch.setattr(characteristic, "_mu_total_inv_ang", _zero_mu)
    # The K-alpha centroid (277.0 eV) sits inside this window, but the window
    # is only ~0.9 FWHM wide, so most of the physical Lorentzian mass is
    # truncated at its edges rather than captured.
    energy = np.array([276.98, 277.02])

    with pytest.warns(RuntimeWarning, match="truncates"):
        characteristic.mc_characteristic_spectrum(
            _carbon_segments([100.0]),
            energy,
            composition=[("C", 0.1)],
            electron_limit=1,
        )


def test_well_covered_window_does_not_warn_about_truncation(monkeypatch, recwarn):
    monkeypatch.setattr(characteristic, "_mu_total_inv_ang", _zero_mu)
    energy = np.arange(275.0, 279.0001, 0.02)

    characteristic.mc_characteristic_spectrum(
        _carbon_segments([100.0]),
        energy,
        composition=[("C", 0.1)],
        electron_limit=1,
    )

    assert not any(issubclass(w.category, RuntimeWarning) for w in recwarn.list)


def test_characteristic_enforces_one_keV_transport_validity_floor(monkeypatch):
    monkeypatch.setattr(characteristic, "_mu_total_inv_ang", _zero_mu)

    with np.testing.assert_raises_regex(ValueError, "requires E_cut_keV >= 1 keV"):
        characteristic.mc_characteristic_spectrum(
            _carbon_segments([100.0]),
            np.arange(250.0, 291.0),
            composition=[("C", 0.1)],
            E_cut_keV=0.5,
        )

    sub_floor = _carbon_segments([100.0])
    sub_floor["E_keV"] = np.array([0.75])
    spectrum = characteristic.mc_characteristic_spectrum(
        sub_floor,
        np.arange(250.0, 291.0),
        composition=[("C", 0.1)],
    )
    np.testing.assert_array_equal(spectrum, 0.0)


def test_absorber_elements_do_not_load_unused_ionization_tables(monkeypatch):
    loaded = []
    real_load = characteristic.load_characteristic_cross_sections

    def tracked_load(element, **kwargs):
        loaded.append(element)
        return real_load(element, **kwargs)

    monkeypatch.setattr(characteristic, "load_characteristic_cross_sections", tracked_load)
    monkeypatch.setattr(characteristic, "_mu_total_inv_ang", _zero_mu)
    characteristic.mc_characteristic_spectrum(
        _carbon_segments([100.0]),
        np.arange(250.0, 291.0),
        composition=[("C", 0.1)],
        layers=[(0.0, 100.0, [("C", 0.1)]), (100.0, 200.0, [("W", 0.05)])],
        electron_limit=1,
    )

    assert loaded == ["C"]


def test_runner_keeps_characteristic_separate_from_both_line_modes(monkeypatch):
    from pyrite.montecarlo import runner

    energy = np.array([100.0, 200.0, 300.0])
    brem_energy = np.array([100.0, 150.0, 250.0, 300.0])
    characteristic_line = np.array([0.1, 0.3, 0.5])
    segments = {
        "L_ang": np.array([1.0]),
        "Ne": 1,
        "n_backscattered": 0,
        "n_missed": 0,
    }
    monkeypatch.setattr(runner, "_segments_on_device", lambda value: value)
    monkeypatch.setattr(
        runner,
        "_lines_for_segments",
        lambda *_args, coherent, **_kwargs: np.full(energy.shape, 2.0 if coherent else 1.0),
    )
    monkeypatch.setattr(
        runner,
        "_characteristic_from_segments",
        lambda _segments, grid, *_args, **_kwargs: (
            np.testing.assert_array_equal(grid, energy) or characteristic_line
        ),
    )
    monkeypatch.setattr(
        runner,
        "_brem_wide_from_segments",
        lambda *_args, **_kwargs: np.zeros(brem_energy.shape),
    )

    output = runner._spectrum_case_impl(
        {
            "name": "case",
            "crystal": "hopg",
            "E0_keV": 30.0,
            "composition": [("C", 0.1)],
            "coherent_emission": True,
        },
        {
            "E_grid": energy,
            "E_brem": brem_energy,
            "n_hat": np.array([0.0, 0.0, -1.0]),
            "segs": segments,
            "Ne_lines": 1,
            "Ne_brem": 1,
            "groove": None,
        },
    )

    np.testing.assert_array_equal(output["spec_characteristic"], characteristic_line)
    np.testing.assert_array_equal(output["spec"], np.full(energy.shape, 1.0))
    np.testing.assert_array_equal(output["spec_coherent"], np.full(energy.shape, 2.0))
    np.testing.assert_array_equal(line_spectrum(output), 1.0 + characteristic_line)
    np.testing.assert_array_equal(line_spectrum(output, coherent=True), 2.0 + characteristic_line)


def test_carbon_l_subshells_bound_below_the_cutoff_are_not_primaries():
    """C L1-L3 (< 20 eV) can never radiate above the 50 eV floor."""
    table = characteristic.load_characteristic_cross_sections("C")

    assert table.ionization_shell_labels == ("K",)
    assert table.relaxation_shell_labels == ("K", "L1", "L2", "L3")
    assert table.vacancy_transfer.shape == (1, 4)
    assert not table.vacancy_transfer.flags.writeable


def test_carbon_k_emission_is_the_eadl_radiative_branch():
    """Source values: EADL2025 Z=6 K radiative FTR (K-L2, K-L3)."""
    table = characteristic.load_characteristic_cross_sections("C")
    elam = characteristic.load_characteristic_cross_sections("C", fluorescence_yields="elam")

    np.testing.assert_allclose(
        table.line_yield_per_vacancy[0], [0.000561488, 0.0011206], rtol=1e-15, atol=0
    )
    assert table.shell_fluorescence_yield[0] == pytest.approx(0.001682088, rel=1e-12)
    # Elam/Krause C omega_K is 0.0014; the flag rescales the radiative branch only.
    assert elam.line_yield_per_vacancy[0].sum() == pytest.approx(0.0014, rel=1e-12)
    assert elam.line_yield_per_vacancy[0, 1] / elam.line_yield_per_vacancy[0, 0] == pytest.approx(
        0.0011206 / 0.000561488, rel=1e-12
    )


def test_copper_lines_join_xraydb_energies_and_keep_eadl_only_transitions():
    table = characteristic.load_characteristic_cross_sections("Cu")
    lines = dict(
        zip(
            table.line_labels,
            zip(table.line_source, table.line_energy_eV, strict=True),
            strict=True,
        )
    )

    assert lines["Ka1"] == ("xraydb", 8046.3)
    assert lines["Kb5"][0] == "xraydb"  # xraydb K-M4,5 claims both EADL K-M4 and K-M5
    assert "Ka3" not in lines  # dipole-forbidden K-L1: no EADL radiative branch
    assert lines["L3-N1"] == ("eadl", 930.38)
    kb5 = table.line_labels.index("Kb5")
    k_row = table.relaxation_shell_labels.index("K")
    k = table.relaxation.subshells[k_row]
    k_m45 = k.radiative_probability[np.isin(k.radiative_final, [8, 9])].sum()
    assert table.radiative_yield_per_decay[k_row, kb5] == pytest.approx(k_m45, rel=1e-15)


def test_cascade_visits_are_the_exact_inverse_of_i_minus_d():
    """Forward substitution against an independent dense solve."""
    for element in ("Si", "Cu", "Au"):
        relaxation = characteristic.load_characteristic_cross_sections(element).relaxation
        daughters, visits = characteristic.vacancy_cascade(relaxation, 50.0)

        assert not np.tril(daughters).any(), "cascade order must make D strictly upper-triangular"
        np.testing.assert_allclose(
            visits, np.linalg.inv(np.eye(len(daughters)) - daughters), rtol=1e-12, atol=1e-12
        )
        n = len(daughters)
        assert not np.linalg.matrix_power(daughters, n).any(), "D must be nilpotent"


def test_cascade_cutoff_above_every_binding_is_the_direct_vacancy_limit():
    relaxation = characteristic.load_characteristic_cross_sections("Cu").relaxation
    daughters, visits = characteristic.vacancy_cascade(relaxation, 1.0e6)

    assert not daughters.any()
    np.testing.assert_array_equal(visits, np.eye(len(visits)))


def test_copper_k_vacancy_feeds_l_emission_and_conserves_probability():
    """Each decay row sums to one plus the number of extra holes it creates."""
    table = characteristic.load_characteristic_cross_sections("Cu")
    relaxation = table.relaxation
    daughters, visits = characteristic.vacancy_cascade(relaxation, 50.0)
    for row, shell in enumerate(relaxation.subshells):
        if shell.binding_energy_eV <= 50.0:
            continue
        radiative = shell.radiative_probability.sum()
        auger = shell.auger_probability.sum()
        assert radiative + auger == pytest.approx(1.0, abs=1e-5)
        assert daughters[row].sum() == pytest.approx(radiative + 2.0 * auger, rel=1e-12)

    k = table.ionization_shell_labels.index("K")
    l_lines = [i for i, shell in enumerate(table.line_initial_shell) if shell.startswith("L")]
    assert table.line_yield_per_vacancy[k, l_lines].sum() > 0.0
    assert visits[0, table.relaxation_shell_labels.index("L3")] == pytest.approx(0.902, abs=5e-4)


def test_relaxation_cutoff_bounds_vacancy_propagation_by_binding_energy():
    """EADL binds Cu L1 at 1103 eV: a 1.2 keV cutoff stops every L decay, not K."""
    table = characteristic.load_characteristic_cross_sections("Cu")
    transfer, yields = characteristic._cascade_line_yields(table, 1200.0)
    k = table.ionization_shell_labels.index("K")
    initial = np.asarray(table.line_initial_shell)

    assert not yields[:, initial != "K"].any()
    k_lines = initial == "K"
    np.testing.assert_allclose(
        yields[k, k_lines], table.radiative_yield_per_decay[0, k_lines], rtol=0, atol=0
    )
    assert transfer[k, table.relaxation_shell_labels.index("L3")] > 0.0


def test_elam_flag_imposes_xraydb_omega_and_keeps_probability():
    table = characteristic.load_characteristic_cross_sections("Cu", fluorescence_yields="elam")
    for label in ("K", "L1", "L2", "L3"):
        index = table.ionization_shell_labels.index(label)
        assert table.shell_fluorescence_yield[index] == pytest.approx(
            xraydb.xray_edge("Cu", label).fyield, rel=1e-12
        )
    daughters, _visits = characteristic.vacancy_cascade(
        table.relaxation, 50.0, fluorescence_yields=table.relaxation_fluorescence_yields
    )
    k = table.relaxation.subshells[0]
    omega = xraydb.xray_edge("Cu", "K").fyield
    assert daughters[0].sum() == pytest.approx(omega + 2.0 * (1.0 - omega), rel=1e-12)
    assert k.fluorescence_yield != pytest.approx(omega, rel=1e-3)


def test_characteristic_rejects_an_unknown_yield_source():
    with pytest.raises(ValueError, match="fluorescence_yields"):
        characteristic.load_characteristic_cross_sections("Cu", fluorescence_yields="krause")


def test_characteristic_model_marker_records_the_eadl_cascade():
    marker = characteristic.CHARACTERISTIC_MODEL

    assert f"eadl-2025-{characteristic.CHARACTERISTIC_EADL_SHA256[:12]}" in marker
    assert marker.endswith("eadl-cascade-eadl-yields-lorentzian-segment-escape-v7")
    assert characteristic.characteristic_model_marker("elam") != marker


def test_light_elements_without_subshells_above_the_cutoff_load_empty_tables():
    """No H or He subshell is bound above 50 eV: nothing can radiate, nothing raises."""
    for element in ("H", "He"):
        table = characteristic.load_characteristic_cross_sections(element)

        assert table.ionization_shell_labels == ()
        assert table.line_labels == ()
        assert table.line_yield_per_vacancy.shape == (0, 0)
