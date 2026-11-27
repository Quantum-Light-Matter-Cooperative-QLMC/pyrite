"""Automatic line grids and the truncation audit for full-fiber CXR cases (#1489).

Validation: cxr-fiber-intensity-average
Validation: line-grid-resonance-bandwidth
"""

import numpy as np
import pytest

from pyrite import Scene
from pyrite._cxr_grid_policy import (
    RESONANCE_BANDWIDTH_POLICY,
    CxrGridToleranceError,
    kinematic_cxr_stop_eV,
)
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.campaign.lowering import lower_simulation
from pyrite.campaign.reflection_selection import FIBER_DOMAIN_BANDWIDTH_MODEL
from pyrite.campaign.resolution import resolve_simulation
from pyrite.config import CxrReflections, FiberAverage, FixedSampling, Numerics, Physics
from pyrite.detectors import Detector, EnergyBins
from pyrite.geometry import Footprint, Slab
from pyrite.materials.crystal import CRYSTALS, reciprocal_g_vector
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import fiber_emission
from pyrite.montecarlo.runner.cxr_grid import (
    _policy_cache_key,
    check_cxr_truncation,
    cxr_truncation_audit,
)
from pyrite.montecarlo.spectrum.cxr.fiber import FiberReflectionSet

PHYSICS = Physics(bremsstrahlung_model="bremslib", radiative_model="uncoupled")
AVERAGE = FiberAverage(cutoff_invang=4.0, azimuth_count=2)


def _lower(energy_keV=30.0, average=AVERAGE, *, histories=2, grid=None):
    scene = Scene(
        Beam(energy_keV, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", 5000.0, tilt_deg=15.0, footprint=Footprint(5.0, 5.0)),
        Detector(),
    )
    numerics = Numerics(
        sampling=FixedSampling(cxr_histories=histories, brem_histories=histories),
        cxr_reflections=CxrReflections(fiber=average),
        energy_bins=None if grid is None else EnergyBins(cxr=grid),
    )
    return lower_simulation(resolve_simulation(scene, physics=PHYSICS, numerics=numerics))


@pytest.mark.parametrize("energy_keV", [30.0, 300.0])
def test_fiber_case_gets_a_bin_mean_policy_bounded_by_its_domain(energy_keV):
    case = _lower(energy_keV)
    policy = case["line_grid_policy"]
    sources = dict(policy["sources"])
    assert policy["bandwidth"]["policy"] == RESONANCE_BANDWIDTH_POLICY
    assert policy["quadrature"] == "bin-mean" and case["line_quadrature"] == "bin-mean"
    assert sources["bandwidth_reflections"] == FIBER_DOMAIN_BANDWIDTH_MODEL
    assert sources["quadrature"] == FIBER_DOMAIN_BANDWIDTH_MODEL
    # The ceiling is the closed-form bound over the domain's rows (|g| is roll
    # invariant), not the basal catalog families or the selection cap.
    lattice = CRYSTALS["hopg"]["lattice"]
    hkls = fiber_emission.fiber_domain(case).hkls
    bound = kinematic_cxr_stop_eV([reciprocal_g_vector(h, lattice)[1] for h in hkls], energy_keV)
    assert policy["bandwidth"]["stop_eV"] == min(bound, energy_keV * 1e3)


def test_explicit_grid_keeps_the_1483_payload():
    grid = np.arange(100.0, 4000.0, 20.0)
    case = _lower(grid=grid)
    assert case.get("line_grid_policy") is None and case["line_quadrature"] == "bin-mean"


def test_fiber_sweep_refuses_feature_windows():
    """Window seeds read the basal ``hkl_list``, so they would miss domain rows."""
    from types import SimpleNamespace

    from pyrite.campaign.reflection_selection import fiber_line_axis

    sweep = SimpleNamespace(cxr_fiber=AVERAGE, cxr_grid_policy={"windows": True})
    with pytest.raises(ValueError, match="feature windows"):
        fiber_line_axis(sweep)


def test_measured_cache_key_reads_the_fiber_domain():
    case = dict(_lower())
    payload = case["line_grid_policy"]
    key, _ = _policy_cache_key(payload, case, (), observation=False)
    other = {**case, "cxr_fiber": {**case["cxr_fiber"], "azimuth_count": 3}}
    assert _policy_cache_key(payload, other, (), observation=False)[0] != key
    # Execution-only batch size shares the entry.
    batched = {**case, "cxr_fiber_batch_size": 7}
    assert _policy_cache_key(payload, batched, (), observation=False)[0] == key


def test_audit_folds_every_roll_call_at_its_quadrature_weight(monkeypatch):
    """Sums, per-electron masses and collected coefficients carry 1 (basal) or 1/N."""
    case = dict(_lower(average=FiberAverage(cutoff_invang=4.0, azimuth_count=3)))
    domain = fiber_emission.fiber_domain(case)
    original = runner._cxr_for_segments

    def spy(segs, grid, pilot, *args, truncation_audit=None, **options):
        if pilot.get("cxr_fiber") is not None:
            return original(segs, grid, pilot, *args, truncation_audit=truncation_audit, **options)
        audit = truncation_audit
        assert audit is not None and audit.get("attribute") is None
        n = float(len(pilot["hkl_list"]))
        if "collect" in audit:
            audit["collect"].append((np.array([1.0]), np.array([2.0]), np.array([n])))
        else:
            audit["line_mass"] = audit.get("line_mass", 0.0) + n
            audit["mass_above"] = n / 10
            audit["mass_above_extension"] = np.array([n, 2 * n])
            audit["electron_mass"] = np.array([n, 0.0])
        return np.zeros(grid.shape)

    monkeypatch.setattr(runner, "_cxr_for_segments", spy)
    grid = np.array([100.0, 200.0])
    basal = sum(h[:2] == (0, 0) for h in domain.hkls)
    # Basal once at weight 1; the rest at 3 rolls of weight 1/3: the
    # roll-averaged row count is the domain's size.
    expected = basal + (len(domain.hkls) - basal)
    audit = {
        "start_eV": 100.0,
        "stop_eV": 200.0,
        "ceiling_eV": 300.0,
        "extension_stops_eV": [250.0, 300.0],
        "n_electrons": 2,
    }
    runner._cxr_for_segments({}, grid, case, None, None, None, Ne=2, truncation_audit=audit)
    assert audit["line_mass"] == pytest.approx(expected)
    assert audit["mass_above"] == pytest.approx(expected / 10)
    np.testing.assert_allclose(audit["mass_above_extension"], [expected, 2 * expected])
    np.testing.assert_allclose(audit["electron_mass"], [expected, 0.0])
    assert {"start_eV", "stop_eV", "ceiling_eV", "n_electrons"} <= set(audit)
    collect = {"start_eV": 100.0, "stop_eV": 200.0, "collect": [], "device_bytes": 1 << 30}
    runner._cxr_for_segments({}, grid, case, None, None, None, Ne=2, truncation_audit=collect)
    assert collect["device_bytes"] is None
    assert sum(float(c[2].sum()) for c in collect["collect"]) == pytest.approx(expected)
    with pytest.raises(ValueError, match="attribution"):
        runner._cxr_for_segments(
            {}, grid, case, None, None, None, truncation_audit={**audit, "attribute": {}}
        )


def _nonbasal(domain):
    rows = [(h, b) for h, b in zip(domain.hkls, domain.b_ang2, strict=True) if h[:2] != (0, 0)]
    return FiberReflectionSet(domain.model, *map(tuple, zip(*rows, strict=True)))


def _population(case, tp, start, stop):
    audit = {"start_eV": start, "stop_eV": stop, "collect": []}
    runner._cxr_for_segments(
        runner._segments_on_device(tp["segs"]),
        np.array([start, stop]),
        case,
        tp["n_hat"],
        None,
        tp.get("groove"),
        coherent=False,
        Ne=int(tp["Ne_lines"]),
        truncation_audit=audit,
    )
    energy = np.concatenate([np.asarray(c[0], dtype=float) for c in audit["collect"]])
    mass = np.concatenate([np.asarray(c[1] * c[2], dtype=float) for c in audit["collect"]])
    return energy, mass


@pytest.mark.slow
@pytest.mark.parametrize("energy_keV", [30.0, 300.0])
def test_automatic_fiber_grid_covers_the_roll_averaged_lines(energy_keV, monkeypatch):
    """Default policy: audited upper share within tolerance for the roll average.

    The resolved axis also holds the non-basal rows' own roll-averaged lines
    (the dominant mass at 300 keV, #1480), and the same audit refuses a band
    that cuts them.
    """
    case = _lower(energy_keV, histories=20)
    tp = runner._transport_case(case, transport_core="per-electron")
    out = runner._spectrum_case(case, tp)
    grid = np.asarray(out["E_grid"], dtype=float)
    record = out["line_grid_resolved"]["truncation_audit"]
    limit = record["truncation_limit"]
    assert record["upper_fraction_bound"] <= limit or record["capped_at_ceiling"]
    bandwidth = case["line_grid_policy"]["bandwidth"]
    start, ceiling = float(bandwidth["start_eV"]), float(bandwidth["stop_eV"])

    domain = fiber_emission.fiber_domain(case)
    monkeypatch.setattr(fiber_emission, "fiber_domain", lambda _case: _nonbasal(domain))
    energy, mass = _population(dict(case), tp, start, ceiling)
    assert mass.sum() > 0.0
    order = np.argsort(energy)
    share = np.cumsum(mass[order]) / mass.sum()
    assert energy[order][np.searchsorted(share, 1.0 - limit)] <= grid[-1]
    monkeypatch.undo()

    # A band ending at the roll-averaged population's mass median cuts lines.
    energy, mass = _population(dict(case), tp, start, ceiling)
    order = np.argsort(energy)
    median = energy[order][np.searchsorted(np.cumsum(mass[order]) / mass.sum(), 0.5)]
    truncated = np.linspace(grid[0], median, 2000)
    audit = cxr_truncation_audit(case, truncated, n_electrons=int(tp["Ne_lines"]))
    runner._cxr_for_segments(
        runner._segments_on_device(tp["segs"]),
        truncated,
        case,
        tp["n_hat"],
        None,
        tp.get("groove"),
        coherent=False,
        Ne=int(tp["Ne_lines"]),
        truncation_audit=audit,
    )
    with pytest.raises(CxrGridToleranceError) as refused:
        check_cxr_truncation(case, audit)
    assert refused.value.truncation_record["upper_fraction_bound"] > 0.1
