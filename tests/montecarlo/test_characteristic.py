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


def test_packaged_carbon_eedl_values_and_relaxation_join():
    table = characteristic.load_characteristic_cross_sections("C")

    assert table.atomic_number == 6
    assert table.ionization_shell_labels[:4] == ("K", "L1", "L2", "L3")
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
    assert np.isclose(table.shell_fluorescence_yield[0], 0.0014, atol=0.0)
    assert np.isclose(table.line_yield_per_vacancy[0].sum(), 0.0014, atol=0.0)
    assert {266.2, 277.0} <= set(table.line_energy_eV)
    ka1 = table.line_labels.index("Ka1")
    # A Lorentzian transition width is the sum of the initial- and final-hole
    # widths: C K (0.0868 eV) + C L3 (0.0045 eV).
    assert np.isclose(table.line_fwhm_eV[ka1], 0.0913, atol=0.0)
    assert not table.shell_binding_energy_eV.flags.writeable
    assert not table.line_fwhm_eV.flags.writeable


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

    def tracked_load(element, *, data_dir=None):
        loaded.append(element)
        return real_load(element, data_dir=data_dir)

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


def _l_shell_ck(element: str) -> tuple[float, float, float]:
    return (
        xraydb.ck_probability(element, "L1", "L2"),
        xraydb.ck_probability(element, "L1", "L3"),
        xraydb.ck_probability(element, "L2", "L3"),
    )


def test_ck_free_light_element_keeps_an_identity_vacancy_transfer():
    """Elements without tabulated L Coster--Kronig reduce to the direct product.

    xraydb reports no L Coster--Kronig for Z <= 11, so carbon's transfer must be
    the identity and its line yields must stay at the pre-cascade
    ``omega_i * I_il``. This is the limiting case that keeps the v4 carbon
    anchors above valid.
    """
    table = characteristic.load_characteristic_cross_sections("C")

    assert _l_shell_ck("C") == (0.0, 0.0, 0.0)
    assert np.array_equal(table.vacancy_transfer, np.eye(len(table.ionization_shell_labels)))
    assert np.isclose(table.line_yield_per_vacancy[0].sum(), 0.0014, atol=0.0)
    assert not table.vacancy_transfer.flags.writeable


def test_copper_l_shell_coster_kronig_redistributes_primary_vacancies():
    """Row ``i`` holds where one primary vacancy in ``i`` ends up.

    Coster--Kronig moves the L hole outward without creating a second L hole,
    so every row sums to one. xraydb's ``f13`` is a *total* probability that
    already contains the L1 -> L2 -> L3 route, which is why it is fed from the
    primary L1 population and ``f23`` is applied only to the primary L2
    population: routing ``f12 * N_L1`` through ``f23`` as well would count that
    path twice.
    """
    table = characteristic.load_characteristic_cross_sections("Cu")
    index = {label: i for i, label in enumerate(table.ionization_shell_labels)}
    f12, f13, f23 = _l_shell_ck("Cu")
    transfer = table.vacancy_transfer

    assert (f12, f13, f23) == (0.3, 0.681, 0.47)
    assert transfer[index["L1"], index["L1"]] == pytest.approx(1.0 - f12 - f13)
    assert transfer[index["L1"], index["L2"]] == pytest.approx(f12)
    assert transfer[index["L1"], index["L3"]] == pytest.approx(f13)
    assert transfer[index["L1"], index["L3"]] != pytest.approx(f13 + f12 * f23)
    assert transfer[index["L2"], index["L2"]] == pytest.approx(1.0 - f23)
    assert transfer[index["L2"], index["L3"]] == pytest.approx(f23)
    assert transfer[index["L3"], index["L3"]] == 1.0
    assert np.allclose(transfer.sum(axis=1), 1.0)
    # Decay fills a hole from a less-bound shell, so vacancies only ever move
    # to higher indices: the transfer is upper triangular.
    assert not np.tril(transfer, -1).any()


def test_m_and_k_shell_rows_are_left_as_identity():
    """Only the L shell is redistributed by this slice.

    xraydb's M-shell Coster--Kronig values are not a probability distribution --
    the finals of Cr M1 sum to 3.82 and 134 (Z, initial) pairs exceed one -- so
    they are deliberately excluded until EADL supplies a normalized topology.
    A K vacancy's own transfer stays the identity because its Auger daughters
    are not propagated by this slice either.
    """
    table = characteristic.load_characteristic_cross_sections("Cu")
    index = {label: i for i, label in enumerate(table.ionization_shell_labels)}
    identity = np.eye(len(table.ionization_shell_labels))

    for label in ("K", "M1", "M2", "M3", "M4", "M5"):
        assert np.array_equal(table.vacancy_transfer[index[label]], identity[index[label]])


def test_copper_l_emission_gains_the_independently_computed_ck_factor():
    """Total L emission against an expectation built only from source tables.

    Expected total is ``sum_i n_i omega_i`` with ``n`` from the
    Krause/Elam Coster--Kronig factors and ``N`` the EEDL primary populations;
    the pre-cascade value is ``sum_i N_i omega_i``. For Cu at 30 keV that
    ratio is 1.2347 -- the L1 hole is the one being moved, and omega_L3 is
    6.9x omega_L1.
    """
    table = characteristic.load_characteristic_cross_sections("Cu")
    index = {label: i for i, label in enumerate(table.ionization_shell_labels)}
    f12, f13, f23 = _l_shell_ck("Cu")
    sigma = {
        label: float(
            np.interp(
                30_000.0,
                table.projectile_energy_eV_by_shell[index[label]],
                table.ionization_cross_sections_cm2_by_shell[index[label]],
            )
        )
        for label in ("L1", "L2", "L3")
    }
    omega = {
        label: float(table.shell_fluorescence_yield[index[label]]) for label in ("L1", "L2", "L3")
    }
    populated = {
        "L1": sigma["L1"] * (1.0 - f12 - f13),
        "L2": sigma["L2"] * (1.0 - f23) + f12 * sigma["L1"],
        "L3": sigma["L3"] + f23 * sigma["L2"] + f13 * sigma["L1"],
    }
    expected = sum(populated[label] * omega[label] for label in populated)
    direct = sum(sigma[label] * omega[label] for label in sigma)

    emitted = sum(
        sigma[label] * float(table.line_yield_per_vacancy[index[label]].sum()) for label in sigma
    )

    assert emitted == pytest.approx(expected, rel=1.0e-12)
    assert emitted / direct == pytest.approx(1.2347, abs=5.0e-4)


def test_l_shell_transfer_rejects_coster_kronig_probabilities_over_one(monkeypatch):
    """A CK table whose L1 finals exceed unity is a data error, not a rescale."""
    monkeypatch.setattr(characteristic.xraydb, "ck_probability", lambda *_a, **_k: 0.6)

    with pytest.raises(ValueError, match="Coster--Kronig"):
        characteristic._l_shell_vacancy_transfer("Cu", ("K", "L1", "L2", "L3"))


def test_l_shell_transfer_skips_absent_subshells():
    """Transfer is only wired between subshells the EEDL table actually carries."""
    transfer = characteristic._l_shell_vacancy_transfer("Cu", ("K", "L1", "L2"))
    f12, _f13, _f23 = _l_shell_ck("Cu")

    assert transfer[1, 2] == pytest.approx(f12)
    assert transfer[1, 1] == pytest.approx(1.0 - f12)
    assert np.allclose(transfer.sum(axis=1), 1.0)


def test_characteristic_model_marker_records_the_ck_relaxation():
    assert characteristic.CHARACTERISTIC_MODEL.endswith("l-shell-ck-lorentzian-v5")
