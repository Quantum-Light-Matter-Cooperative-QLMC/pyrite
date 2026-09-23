"""Physical invariants of the SBETHE optical-response partition."""

import numpy as np
import pytest
from scipy.constants import N_A

from pyrite.montecarlo.transport.inelastic import (
    _oscillator_quadrature,
    build_gos_partition,
    hard_transfer_cdf,
    sample_hard_collision,
)
from pyrite.xsgen.sbethe import resolve_catalog_table


@pytest.mark.parametrize("material", ["silicon", "mos2", "sio2"])
def test_shipped_optical_strength_obeys_f_sum_and_mean_excitation(material):
    arrays = resolve_catalog_table(material).arrays()
    energy = arrays["oos_energy_eV"]
    density = arrays["oos_per_eV"]
    total = np.trapezoid(density, energy)
    geometric_mean = np.exp(np.trapezoid(density * np.log(energy), energy) / total)

    assert total == pytest.approx(float(arrays["electrons_per_molecule"]), rel=1e-3)
    assert geometric_mean == pytest.approx(float(arrays["mean_excitation_eV"]), rel=1e-3)


def test_silicon_oos_core_edge_preserves_strength_and_raw_transfer_spectrum():
    arrays = resolve_catalog_table("silicon").arrays()
    edge_ev = 102.2154  # First positive shell-edge jump in the shipped OOS.dat.
    w, strength = _oscillator_quadrature(arrays)
    valence_strength = np.sum(strength[w < edge_ev])
    core_strength = np.sum(strength[w >= edge_ev])
    assert 3.5 < valence_strength < 4.0
    assert 10.0 < core_strength < 10.5
    assert valence_strength + core_strength == pytest.approx(np.sum(strength))

    partitions = [
        build_gos_partition(arrays, 3000.0, cutoff, core_edge_ev=edge_ev)
        for cutoff in (1.0, edge_ev, 150.0, 3000.0)
    ]
    full = build_gos_partition(arrays, 3000.0, 1.0)
    reference = partitions[0]
    assert reference.calibration == 1.0
    assert reference.core_edge_ev == edge_ev
    assert reference.raw_sigma1_ev_cm2 < full.raw_sigma1_ev_cm2
    assert np.min(reference.lower_ev) >= edge_ev
    assert hard_transfer_cdf(reference, np.asarray([edge_ev]))[0] == 0.0
    for partition in partitions:
        assert partition.raw_sigma0_cm2 == pytest.approx(reference.raw_sigma0_cm2, rel=1e-13)
        assert partition.raw_sigma1_ev_cm2 == pytest.approx(reference.raw_sigma1_ev_cm2, rel=1e-13)
        assert partition.soft_stopping_ev_cm2 + partition.hard_stopping_ev_cm2 == pytest.approx(
            partition.raw_sigma1_ev_cm2, rel=1e-13
        )
    assert partitions[-1].hard_sigma_cm2 == 0.0

    rng = np.random.default_rng(93)
    sampled = [sample_hard_collision(reference, *u) for u in rng.random((128, 3))]
    assert min(event.transfer_ev for event in sampled) >= edge_ev
    assert all(event.secondary_energy_ev is None for event in sampled)


def test_core_gos_requires_a_positive_native_shell_edge_jump():
    arrays = resolve_catalog_table("silicon").arrays()
    with pytest.raises(ValueError, match="shell-edge jump"):
        build_gos_partition(arrays, 3000.0, 1.0, core_edge_ev=100.0)


@pytest.mark.parametrize("threshold_ev", [1.0, 50.0, 500.0, 10_000.0, 100_000.0])
def test_partition_keeps_corrected_mean_without_a_loss_gap(threshold_ev):
    arrays = resolve_catalog_table("silicon").arrays()
    energy_ev = 100_000.0
    partition = build_gos_partition(arrays, energy_ev, threshold_ev)
    expected = np.exp(
        np.interp(
            np.log(energy_ev),
            np.log(arrays["stopping_energy_eV"]),
            np.log(arrays["stopping_cs_eV_cm2"]),
        )
    )

    assert partition.soft_stopping_ev_cm2 >= 0.0
    assert partition.hard_stopping_ev_cm2 >= 0.0
    assert partition.soft_stopping_ev_cm2 + partition.hard_stopping_ev_cm2 == pytest.approx(
        expected, rel=5e-12
    )
    assert partition.hard_sigma_cm2 >= 0.0
    if threshold_ev >= energy_ev:
        assert partition.hard_sigma_cm2 == 0.0


@pytest.mark.parametrize("material", ["silicon", "mos2", "sio2"])
def test_cutoff_partitions_one_fixed_transfer_spectrum(material):
    arrays = resolve_catalog_table(material).arrays()
    thresholds = (1.0, 50.0, 503.21, 777.123, 5000.0, 100_000.0)
    partitions = [build_gos_partition(arrays, 100_000.0, wc) for wc in thresholds]
    reference = partitions[0]

    for partition in partitions[1:]:
        assert partition.raw_sigma0_cm2 == pytest.approx(reference.raw_sigma0_cm2, rel=1e-14)
        assert partition.raw_sigma1_ev_cm2 == pytest.approx(reference.raw_sigma1_ev_cm2, rel=1e-14)
        assert partition.raw_sigma2_ev2_cm2 == pytest.approx(
            reference.raw_sigma2_ev2_cm2, rel=1e-14
        )
        assert partition.calibration == pytest.approx(reference.calibration, rel=1e-14)
        assert partition.total_sigma_cm2 == pytest.approx(reference.total_sigma_cm2, rel=1e-14)
    assert [p.hard_sigma_cm2 for p in partitions] == sorted(
        (p.hard_sigma_cm2 for p in partitions), reverse=True
    )
    assert [p.hard_stopping_ev_cm2 for p in partitions] == sorted(
        (p.hard_stopping_ev_cm2 for p in partitions), reverse=True
    )
    assert partitions[-1].hard_sigma_cm2 == 0.0


def test_interior_cutoff_clips_close_bins_and_samples_above_cutoff():
    arrays = resolve_catalog_table("silicon").arrays()
    threshold_ev = 777.123
    partition = build_gos_partition(arrays, 100_000.0, threshold_ev)
    close = partition.kind == 2
    assert np.any(close)
    assert np.min(partition.lower_ev[close]) == pytest.approx(threshold_ev)
    index = int(np.flatnonzero(close)[0])
    previous = partition.cumulative_hard_cm2[index - 1] if index else 0.0
    event = (previous + partition.cumulative_hard_cm2[index]) / (2.0 * partition.hard_sigma_cm2)
    for u in (0.0, 0.25, 0.5, 0.999999):
        # Select first clipped close component by its cross-section midpoint.
        collision = sample_hard_collision(partition, float(event), u, 0.5)
        assert threshold_ev <= collision.transfer_ev <= partition.upper_ev[index]


def test_hard_transfer_spectrum_matches_seeded_event_samples():
    arrays = resolve_catalog_table("silicon").arrays()
    partition = build_gos_partition(arrays, 3000.0, 10.0)
    edges = np.array([10.0, 15.0, 27.0, 100.0, 1000.0, 3000.0])
    cdf = hard_transfer_cdf(partition, edges)
    assert np.all(np.diff(cdf) >= 0.0)
    assert cdf[0] >= 0.0
    assert cdf[-1] == pytest.approx(1.0)

    rng = np.random.default_rng(93)
    uniforms = rng.random((10_000, 3))
    sampled = np.array([sample_hard_collision(partition, *u).transfer_ev for u in uniforms])
    observed = np.array([np.mean(sampled <= edge) for edge in edges])
    assert np.all(np.abs(observed - cdf) < 0.015)


@pytest.mark.parametrize("energy_ev", [1000.0, 5000.0, 50_000.0])
def test_silicon_loss_mode_against_independent_penn_bethe_fano_shapes(energy_ev):
    """KESS thesis, Fig. 4.5a: dominant Si loss lies near 17 ± 10 eV."""
    arrays = resolve_catalog_table("silicon").arrays()
    partition = build_gos_partition(arrays, energy_ev, 1.0)
    edges = np.arange(0.0, 101.0)
    bin_probability = np.diff(hard_transfer_cdf(partition, edges))
    mode_ev = edges[np.argmax(bin_probability)]
    assert 7.0 <= mode_ev <= 27.0
    assert bin_probability[7:27].sum() > 0.5


@pytest.mark.xfail(strict=True, reason="Si REELS peak height exposes unresolved GOS shape deficit")
@pytest.mark.parametrize("energy_ev", [1000.0, 3000.0])
def test_silicon_loss_peak_height_against_reels(energy_ev):
    """Werner, Phys. Rev. B 74, 075421, Fig. 3b: bulk peak exceeds 0.08/eV."""
    arrays = resolve_catalog_table("silicon").arrays()
    partition = build_gos_partition(arrays, energy_ev, 1.0)
    edges = np.arange(0.0, 51.0)
    cdf = hard_transfer_cdf(partition, edges)
    # Even if the plotted curve were normalized only over its 0–50 eV
    # window, the model peak remains below the conservative figure bound.
    density_per_ev = np.diff(cdf) / cdf[-1]
    assert np.max(density_per_ev) >= 0.08


@pytest.mark.xfail(
    strict=True, reason="Si REELS plasmon probability exposes unresolved GOS shape deficit"
)
@pytest.mark.parametrize("energy_ev", [1000.0, 3000.0])
def test_silicon_plasmon_window_probability_against_reels(energy_ev):
    """Werner, Phys. Rev. B 74, 075421, Fig. 3b: 14–20 eV area exceeds 0.35."""
    arrays = resolve_catalog_table("silicon").arrays()
    partition = build_gos_partition(arrays, energy_ev, 1.0)
    cdf = hard_transfer_cdf(partition, np.array([14.0, 20.0, 50.0]))
    # Figure-read bound allows retrieval artifacts and axis-reading error.
    # Normalizing to the plotted window is deliberately favorable to this model.
    assert (cdf[1] - cdf[0]) / cdf[2] >= 0.35


def test_hard_sampling_respects_threshold_and_recoil_limits():
    arrays = resolve_catalog_table("silicon").arrays()
    energy_ev = 100_000.0
    threshold_ev = 500.0
    partition = build_gos_partition(arrays, energy_ev, threshold_ev)
    assert partition.hard_sigma_cm2 > 0.0

    for u in np.linspace(0.001, 0.999, 71):
        collision = sample_hard_collision(partition, float(u), 0.37, 0.61)
        assert threshold_ev <= collision.transfer_ev <= energy_ev
        assert -1.0 <= collision.cos_primary <= 1.0
        assert np.isfinite(collision.cos_primary)
        if collision.kind == "close":
            assert collision.transfer_ev <= energy_ev / 2.0
            assert collision.secondary_energy_ev == pytest.approx(collision.transfer_ev)
            assert 0.0 <= collision.cos_secondary <= 1.0
        else:
            assert collision.secondary_energy_ev is None
            assert collision.cos_secondary is None

    close_index = np.flatnonzero(partition.kind == 2)[0]
    previous = partition.cumulative_hard_cm2[close_index - 1] if close_index else 0.0
    u_close = float(
        (previous + partition.cumulative_hard_cm2[close_index]) / (2.0 * partition.hard_sigma_cm2)
    )
    suppressed = sample_hard_collision(
        partition, u_close, 0.5, 0.5, production_threshold_ev=energy_ev
    )
    assert suppressed.kind == "close"
    assert suppressed.secondary_energy_ev is None
    assert suppressed.cos_secondary is None


def test_partition_rejects_energies_outside_corrected_stopping_domain():
    arrays = resolve_catalog_table("silicon").arrays()
    with pytest.raises(ValueError, match="stopping table"):
        build_gos_partition(arrays, 999.0, 100.0)


def test_equal_to_incident_energy_is_soft_even_for_a_delta_resonance():
    """An optical resonance at W=E must not survive Wc=E as a hard event."""
    arrays = {
        "stopping_energy_eV": np.array([500.0, 2000.0]),
        "stopping_cs_eV_cm2": np.array([1e-16, 1e-16]),
        "oos_energy_eV": np.array([999.0, 1001.0]),
        "oos_per_eV": np.array([0.5, 0.5]),
    }
    partition = build_gos_partition(arrays, 1000.0, 1000.0)
    assert partition.hard_sigma_cm2 == 0.0
    assert partition.hard_stopping_ev_cm2 == 0.0
    assert partition.soft_stopping_ev_cm2 == pytest.approx(1e-16)


@pytest.mark.parametrize(
    ("material", "valence", "band_gap_ev"),
    [("silicon", 4, 1.1), ("sio2", 16, 9.0)],
)
@pytest.mark.parametrize("energy_ev", [1000.0, 2000.0])
def test_total_imfp_against_independent_nist_tpp2m(material, valence, band_gap_ev, energy_ev):
    """NIST SRD 71 User Guide, Appendix A, Eqs. A.1–A.6; 50–2000 eV domain."""
    arrays = resolve_catalog_table(material).arrays()
    density = float(arrays["density_g_cm3"])
    molar_mass = float(arrays["molecular_weight_g_mol"])
    valence_density = valence * density / molar_mass
    plasma_ev = 28.8 * np.sqrt(valence_density)
    beta = -0.10 + 0.944 / np.sqrt(plasma_ev**2 + band_gap_ev**2) + 0.069 * density**0.1
    gamma = 0.191 * density**-0.5
    cc = 1.97 - 0.91 * valence_density
    dd = 53.4 - 20.8 * valence_density
    nist_imfp_ang = energy_ev / (
        plasma_ev**2 * (beta * np.log(gamma * energy_ev) - cc / energy_ev + dd / energy_ev**2)
    )

    partition = build_gos_partition(arrays, energy_ev, energy_ev)
    molecular_density_cm3 = density * N_A / molar_mass
    model_imfp_ang = 1e8 / (molecular_density_cm3 * partition.total_sigma_cm2)

    # NIST estimates ~20.5% absolute standard uncertainty for TPP-2M;
    # allow 25% for that and this OOS-bin approximation. Si is within 5%.
    assert model_imfp_ang == pytest.approx(nist_imfp_ang, rel=0.25)
