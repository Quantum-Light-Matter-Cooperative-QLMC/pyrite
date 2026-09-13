"""EEDL characteristic-radiation parser, units, and runner integration."""

from __future__ import annotations

import hashlib

import numpy as np

from pyrite.montecarlo.spectrum import characteristic


def _carbon_segments(lengths: list[float]) -> dict[str, object]:
    lengths_array = np.asarray(lengths, dtype=float)
    z_mid = np.cumsum(lengths_array) - 0.5 * lengths_array
    return {
        "r_mid": np.column_stack((np.zeros_like(z_mid), np.zeros_like(z_mid), z_mid)),
        "v_hat": np.tile([0.0, 0.0, 1.0], (lengths_array.size, 1)),
        "L_ang": lengths_array,
        "E_keV": np.full(lengths_array.size, 30.0),
        "elec_id": np.zeros(lengths_array.size, dtype=int),
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
    monkeypatch.setattr(
        characteristic,
        "_mu_total_inv_ang",
        lambda _composition, energy: np.zeros(np.asarray(energy).shape),
    )
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
    in_window = (table.line_energy_eV >= energy[0]) & (table.line_energy_eV <= energy[-1])
    # The emitted photons are the summed per-line yields, not the edge yield
    # alone: the source branch intensities sum to 1 + 9.7e-8 for carbon and are
    # deliberately not renormalized, so omega_K differs from their sum at that
    # relative level. The default 1e-8 absolute tolerance is larger than these
    # ~6e-7 densities and hid the difference, hence ``atol=0.0``.
    expected = (
        density_ang3
        * 1.0e24
        * length_ang
        * 1.0e-8
        * sigma_k
        * table.line_yield_per_vacancy[0, in_window].sum()
        / (4.0 * np.pi)
    )
    # Uniform 1 eV bins: the density sum is the bin-integrated photon yield.
    assert np.isclose(spectrum.sum(), expected, rtol=2.0e-13, atol=0.0)


def test_constant_energy_segment_subdivision_preserves_characteristic_yield(monkeypatch):
    monkeypatch.setattr(
        characteristic,
        "_mu_total_inv_ang",
        lambda _composition, energy: np.zeros(np.asarray(energy).shape),
    )
    energy = np.arange(250.0, 291.0, 1.0)
    common = dict(E_grid_eV=energy, composition=[("C", 0.1)], electron_limit=1)

    whole = characteristic.mc_characteristic_spectrum(_carbon_segments([100.0]), **common)
    split = characteristic.mc_characteristic_spectrum(_carbon_segments([50.0, 50.0]), **common)

    np.testing.assert_allclose(split, whole, rtol=2.0e-13, atol=0.0)


def test_characteristic_line_is_bin_integrated_lorentzian_with_preserved_yield(monkeypatch):
    monkeypatch.setattr(
        characteristic,
        "_mu_total_inv_ang",
        lambda _composition, energy: np.zeros(np.asarray(energy).shape),
    )
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
    in_window = (table.line_energy_eV >= energy[0]) & (table.line_energy_eV <= energy[-1])
    expected = (
        density_ang3
        * 1.0e24
        * length_ang
        * 1.0e-8
        * sigma_k
        * table.line_yield_per_vacancy[0, in_window].sum()
        / (4.0 * np.pi)
    )
    edges, widths = characteristic._energy_bin_edges_and_widths(energy)

    assert np.count_nonzero(spectrum) == spectrum.size
    assert np.isclose(np.sum(spectrum * widths), expected, rtol=2.0e-13, atol=0.0)
    peak = int(np.argmax(spectrum))
    np.testing.assert_allclose(
        spectrum[peak - 5 : peak],
        spectrum[peak + 1 : peak + 6][::-1],
        rtol=1.0e-9,
        atol=0.0,
    )


def test_absorber_elements_do_not_load_unused_ionization_tables(monkeypatch):
    loaded = []
    real_load = characteristic.load_characteristic_cross_sections

    def tracked_load(element, *, data_dir=None):
        loaded.append(element)
        return real_load(element, data_dir=data_dir)

    monkeypatch.setattr(characteristic, "load_characteristic_cross_sections", tracked_load)
    monkeypatch.setattr(
        characteristic,
        "_mu_total_inv_ang",
        lambda _composition, energy: np.zeros(np.asarray(energy).shape),
    )
    characteristic.mc_characteristic_spectrum(
        _carbon_segments([100.0]),
        np.arange(250.0, 291.0),
        composition=[("C", 0.1)],
        layers=[(0.0, 100.0, [("C", 0.1)]), (100.0, 200.0, [("W", 0.05)])],
        electron_limit=1,
    )

    assert loaded == ["C"]


def test_runner_adds_one_characteristic_component_to_both_line_modes(monkeypatch):
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
    np.testing.assert_array_equal(output["spec"], 1.0 + characteristic_line)
    np.testing.assert_array_equal(output["spec_coherent"], 2.0 + characteristic_line)
