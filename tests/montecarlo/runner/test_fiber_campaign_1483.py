"""Full-fiber CXR as a campaign option: identity, lowering, refusals, runner seams (#1483).

Validation: cxr-fiber-intensity-average
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite import Scene
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.campaign.identity_options import _CONTENT_KEY_DENYLIST
from pyrite.campaign.lowering import lower_simulation
from pyrite.campaign.resolution import resolve_simulation
from pyrite.config import CxrReflections, FiberAverage, FixedSampling, Numerics, Physics
from pyrite.detectors import Detector, EnergyBins
from pyrite.geometry import Footprint, Slab
from pyrite.montecarlo.runner import fiber_emission
from pyrite.montecarlo.runner import lattice_share as ls
from pyrite.montecarlo.spectrum.cxr import fiber
from pyrite.montecarlo.trajectories import case_digest

GRID = np.arange(100.0, 4000.0, 20.0)
PHYSICS = Physics(bremsstrahlung_model="bremslib", radiative_model="uncoupled")


def _scene():
    return Scene(
        Beam(30.0, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", 5000.0, tilt_deg=15.0, footprint=Footprint(5.0, 5.0)),
        Detector(),
    )


def _numerics(fiber_average=None, *, grid=GRID, **extra):
    return Numerics(
        sampling=FixedSampling(cxr_histories=2, brem_histories=2),
        cxr_reflections=CxrReflections(fiber=fiber_average),
        energy_bins=None if grid is None else EnergyBins(cxr=grid),
        **extra,
    )


def _lower(numerics, physics=PHYSICS):
    return lower_simulation(resolve_simulation(_scene(), physics=physics, numerics=numerics))


def test_fiber_case_adds_identity_without_touching_basal_identity():
    basal = _lower(_numerics())
    fiber_case = _lower(_numerics(FiberAverage(cutoff_invang=6.0, azimuth_count=8)))
    assert "cxr_fiber" not in basal and "cxr_fiber_batch_size" not in basal
    record = fiber_case["cxr_fiber"]
    assert record["model"] == fiber.FIBER_CXR_MODEL
    assert (record["cutoff_invang"], record["azimuth_count"]) == (6.0, 8)
    assert "cxr_fiber_batch_size" not in fiber_case  # default batch adds nothing
    # Everything else is the basal case, minus its automatic selection: the
    # same catalog families still name the DBR/Kossel rows.
    assert basal["reflection_selection"]["automatic"] is True
    assert fiber_case["line_quadrature"] == "bin-mean" and "line_quadrature" not in basal
    rest = {k: v for k, v in fiber_case.items() if k not in ("cxr_fiber", "line_quadrature")}
    unselected = {k: v for k, v in basal.items() if k != "reflection_selection"}
    assert case_digest(rest) == case_digest(unselected)
    assert case_digest(fiber_case) != case_digest(basal)
    other = _lower(_numerics(FiberAverage(cutoff_invang=6.0, azimuth_count=16)))
    assert case_digest(other) != case_digest(fiber_case)


def test_batch_size_is_execution_only():
    case = _lower(_numerics(FiberAverage(cutoff_invang=6.0, reflection_batch_size=64)))
    assert case["cxr_fiber_batch_size"] == 64
    assert "cxr_fiber_batch_size" in _CONTENT_KEY_DENYLIST
    assert "cxr_fiber" not in _CONTENT_KEY_DENYLIST
    domain = fiber_emission.fiber_domain(case)
    assert domain.model.reflection_batch_size == 64


def test_case_record_round_trips_and_refuses_foreign_records():
    model = fiber.FiberCXRModel.from_numerics(FiberAverage(cutoff_invang=6.0))
    record = model.case_record()
    assert fiber.FiberCXRModel.from_case_record(record) == model
    for bad in (
        {**record, "model": "hopg-fiber-v0"},
        {**record, "thermal_source": "elsewhere"},
        {k: v for k, v in record.items() if k != "temperature_K"},
        {**record, "extra": 1},
    ):
        with pytest.raises(ValueError):
            fiber.FiberCXRModel.from_case_record(bad)


def test_fiber_owns_the_reflection_domain():
    average = FiberAverage(cutoff_invang=6.0)
    for kwargs in ({"n_families": 2}, {"max_reflections": 5}, {"line_yield_tolerance": 1e-3}):
        with pytest.raises(ValueError, match="owns the reflection domain"):
            CxrReflections(fiber=average, **kwargs)
    case = _lower(_numerics(average))
    assert case.get("reflection_selection") is None


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (dict(bremsstrahlung_model=None), "BremsLib"),
        (dict(coherent_emission=True), "incoherent"),
        (dict(temporal_profile=True), "temporal"),
        (dict(mosaic_mc_sampling="history"), "mosaic"),
        (dict(mosaic_fwhm_rad=0.014), "mosaic_route='mc'"),
        (dict(cxr_source="none"), "CXR lines"),
        (dict(line_quadrature=None), "bin-mean"),
        (dict(line_grid_policy={"windows": {"providers": ["kinematic"]}}), "feature windows"),
        (dict(surface_hkl=(1, 0, 0)), "c axis"),
    ],
)
def test_case_refuses_routes_outside_the_envelope(change, reason):
    case = dict(_lower(_numerics(FiberAverage(cutoff_invang=6.0))))
    case.update(change)
    case = {k: v for k, v in case.items() if v is not None}
    with pytest.raises(ValueError, match=reason):
        fiber_emission.validate_fiber_case(case)


def test_case_schema_validates_fiber_fields():
    case = _lower(_numerics(FiberAverage(cutoff_invang=6.0)))
    with pytest.raises(ValueError, match="BremsLib"):
        replace(case, bremsstrahlung_model=_absent(case))
    basal = _lower(_numerics())
    with pytest.raises(ValueError, match="requires cxr_fiber"):
        replace(basal, cxr_fiber_batch_size=8)
    with pytest.raises(ValueError):
        replace(case, cxr_fiber_batch_size=0)


def _absent(case):
    from pyrite.montecarlo.case import _ABSENT

    return _ABSENT


def test_streaming_refuses_fiber_cases():
    from pyrite.montecarlo.runner.streamed_scoring import streaming_refusal

    case = _lower(_numerics(FiberAverage(cutoff_invang=6.0)))
    assert "cxr_fiber" in streaming_refusal(case)


def test_line_seam_routes_rolls_through_the_basal_scorer(monkeypatch):
    from pyrite.montecarlo import runner

    case = dict(_lower(_numerics(FiberAverage(cutoff_invang=4.0, azimuth_count=3))))
    domain = fiber_emission.fiber_domain(case)
    calls = []
    original = runner._cxr_for_segments

    def spy(segs, grid, pilot, *args, **options):
        if pilot.get("cxr_fiber") is not None:
            return original(segs, grid, pilot, *args, **options)
        assert options["table_cache"] is None and options["coherent"] is False
        assert "cxr_fiber_batch_size" not in pilot
        for hkl in pilot["hkl_list"]:
            assert pilot["B_ang2"](hkl) == domain._thermal_lookup()[hkl]
        calls.append((tuple(pilot["hkl_list"]), pilot["azimuth_rad"]))
        return np.full(grid.shape, float(len(pilot["hkl_list"])))

    monkeypatch.setattr(runner, "_cxr_for_segments", spy)
    grid = np.array([100.0, 200.0])
    out = runner._cxr_for_segments({}, grid, case, None, None, None, Ne=2)
    basal = sum(h[:2] == (0, 0) for h in domain.hkls)
    # Basal rows once; every other row once per roll, averaged over the rolls.
    np.testing.assert_allclose(out, len(domain.hkls))
    assert sum(len(h) for h, _ in calls) == basal + 3 * (len(domain.hkls) - basal)
    with pytest.raises(ValueError, match="temporal"):
        runner._cxr_for_segments({}, grid, case, None, None, None, temporal={})


def test_continuum_seam_uses_the_fiber_domain(monkeypatch):
    monkeypatch.setattr(ls, "_CACHE", {})
    base = {"C": SimpleNamespace(key="base", digest="0")}
    case = dict(_lower(_numerics(FiberAverage(cutoff_invang=4.0))))
    domain = fiber_emission.fiber_domain(case)
    assert ls.scored_reflections(case, None) == list(domain.hkls)
    assert ls.lattice_share_applies(case)
    seen = {}
    calls = []

    def corrected(self, tables, composition, **options):
        calls.append(options)
        seen.update(options, hkls=self.hkls)
        return {"C": "corrected"}, {"model": fiber.FIBER_CXR_MODEL, "shells": 1}

    monkeypatch.setattr(fiber.FiberReflectionSet, "corrected_bremslib_tables", corrected)
    tables, record = ls.layer_emission_tables(case, None, [("C", 0.1)], base)
    assert tables == {"C": "corrected"} and record["corrected"] is True
    assert ls.layer_emission_tables(case, None, [("C", 0.1)], base)[1] == record
    assert len(calls) == 1  # one table build per domain, composition, tables and energy
    assert record["reflections"] == len(domain.hkls)
    assert seen == dict(
        E_cut_keV=float(case.get("E_cut_lines_keV", 5.0)),
        max_energy_keV=float(case["E0_keV"]),
        hkls=domain.hkls,
    )
    with pytest.raises(ValueError, match="BremsLib"):
        ls.layer_isotropic_share(case, None, [("C", 0.1)])


def test_paired_scorer_accepts_a_campaign_fiber_case(monkeypatch):
    """``score_fiber_case`` on a lowered fiber case must not recurse into the fiber seam."""
    from pyrite.montecarlo.spectrum.cxr import fiber_case

    case = dict(_lower(_numerics(FiberAverage(cutoff_invang=4.0))))
    pilot = fiber_case._single_radiator(case, {"n_layers": 1})
    assert "cxr_fiber" not in pilot and "cxr_fiber_batch_size" not in pilot


@pytest.mark.slow
def test_campaign_spectrum_matches_the_paired_scorer_on_one_transport():
    """Live lines and continuum equal :func:`score_fiber_case` on the same ``tp``."""
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.spectrum.cxr.fiber_case import score_fiber_case

    average = FiberAverage(cutoff_invang=4.0, azimuth_count=2, reflection_batch_size=16)
    case = _lower(_numerics(average, grid=np.arange(200.0, 3000.0, 40.0)))
    tp = runner._transport_case(case, transport_core="per-electron")
    out = runner._spectrum_case(case, tp)
    model = fiber.FiberCXRModel.from_numerics(average)
    paired = score_fiber_case(model, case, tp, out["E_grid"])
    assert np.max(paired["cxr"]) > 0.0
    np.testing.assert_allclose(out["spec"] - out.get("spec_dbr", 0.0), paired["cxr"], rtol=1e-12)
    brem = np.interp(out["E_grid"], out["E_grid_brem"], out["brem_wide"])
    record = out["brem_lattice_share"]["layers"][0]
    assert record["model"] == fiber.FIBER_CXR_MODEL
    assert record["reflections"] == len(fiber_emission.fiber_domain(case).hkls)
    # Same matched subtraction: removed fraction and clamps per element.
    assert record["elements"] == paired["continuum_correction"]["elements"]
    assert record["shells"] == paired["continuum_correction"]["shells"]
    assert np.all(np.isfinite(brem))


def _fiber_profile_catalog(monkeypatch, table):
    from types import MappingProxyType

    from pyrite.campaign import config
    from pyrite.materials import CATALOG

    custom = replace(
        CATALOG,
        profile_transport_numerics=MappingProxyType(
            {
                **CATALOG.profile_transport_numerics,
                "standard": MappingProxyType({"cxr_fiber": table}),
            }
        ),
    )
    monkeypatch.setattr(config, "CATALOG", custom)
    return config


def test_profile_selects_the_fiber_domain_and_forks_dataset_identity(monkeypatch):
    from pyrite.campaign.config import default_settings, material_sweep
    from pyrite.campaign.profiles import dataset_identity
    from pyrite.campaign.sweep import build_cases

    grid = np.arange(200.0, 3000.0, 40.0)
    baseline = material_sweep("hopg", energy_bins=EnergyBins(cxr=grid))
    config = _fiber_profile_catalog(monkeypatch, {"cutoff_invang": 6.0, "azimuth_count": 8})
    sweep = config.material_sweep("hopg", energy_bins=EnergyBins(cxr=grid))
    assert sweep.cxr_fiber == FiberAverage(cutoff_invang=6.0, azimuth_count=8)
    settings = default_settings()
    fiber_id = dataset_identity("hopg", settings, sweep)["parameter_sha256"]
    basal_id = dataset_identity("hopg", settings, baseline)["parameter_sha256"]
    assert fiber_id != basal_id
    # A basal sweep's payload carries no fiber key at all (historical digest).
    assert "cxr_fiber" not in json_payload(baseline, settings)
    cases = build_cases(sweep, n_electrons=1, n_electrons_brem=1)
    assert cases and all(c["cxr_fiber"]["cutoff_invang"] == 6.0 for c in cases)
    assert all(c.get("reflection_selection") is None for c in cases)
    # Without an explicit line grid the stored basal rows are skipped: the
    # case resolves automatically over the fiber domain (#1489).
    stored = config.material_sweep("hopg", energy_bins=EnergyBins(cxr_by_energy={30.0: grid}))
    assert stored.energy_bins.cxr_by_energy
    automatic = build_cases(stored, n_electrons=1, n_electrons_brem=1)
    assert automatic and all(c.get("line_grid_policy") is not None for c in automatic)


def test_dataset_identity_hashes_the_case_record_not_the_execution_batch(monkeypatch):
    from pyrite.campaign.config import default_settings, material_sweep
    from pyrite.campaign.profiles import dataset_identity

    grid = EnergyBins(cxr=np.arange(200.0, 3000.0, 40.0))
    base = material_sweep("hopg", energy_bins=grid)
    settings = default_settings()

    def identity(average):
        return dataset_identity("hopg", settings, replace(base, n_families=4, cxr_fiber=average))

    first = identity(FiberAverage(cutoff_invang=6.0))
    same = identity(FiberAverage(cutoff_invang=6, reflection_batch_size=32))
    assert first["parameter_sha256"] == same["parameter_sha256"]
    record = first["resolved_parameters"]["sweep"]["cxr_fiber"]
    assert record == fiber.FiberCXRModel.from_numerics(base_average()).case_record()
    monkeypatch.setattr(fiber, "FIBER_CXR_MODEL", "hopg-fiber-next")
    assert (
        identity(FiberAverage(cutoff_invang=6.0))["parameter_sha256"] != first["parameter_sha256"]
    )


def base_average():
    return FiberAverage(cutoff_invang=6.0)


def test_auto_bremsstrahlung_resolves_before_the_fiber_case_is_built(monkeypatch):
    from pyrite.xsgen.bremslib import tables

    monkeypatch.setattr(tables, "load_bremsstrahlung_tables", lambda elements: {})
    physics = Physics(bremsstrahlung_model="auto", radiative_model="uncoupled")
    average = FiberAverage(cutoff_invang=6.0)
    assert _lower(_numerics(average), physics) == _lower(_numerics(average))


def test_converted_fiber_sweep_keeps_and_can_drop_the_domain():
    from pyrite.campaign.config import default_settings, material_sweep
    from pyrite.campaign.legacy import adapt_configured_sweep

    average = FiberAverage(cutoff_invang=6.0)
    old = material_sweep("hopg", energy_bins=EnergyBins(cxr=GRID))
    old = replace(old, n_families=4, cxr_fiber=average)
    converted = adapt_configured_sweep(old, default_settings())
    numerics = converted.base.numerics
    assert numerics.cxr_reflections.fiber == average
    kept = converted.cases(replace(numerics, seed=5))
    assert kept and all(case["cxr_fiber"]["cutoff_invang"] == 6.0 for case in kept)
    eight = replace(numerics, cxr_reflections=CxrReflections(n_families=8))
    dropped = converted.cases(eight)
    assert dropped and all("cxr_fiber" not in case for case in dropped)


def test_pixel_scene_keeps_the_profile_fiber_domain():
    from types import MappingProxyType

    from pyrite.campaign.single_scene import resolve_pixel_scene
    from pyrite.materials import load_material_catalog

    name = "pixelated_detector_demo"
    catalog = load_material_catalog(profile=name)
    assert resolve_pixel_scene(catalog, "hopg", name)[6].cxr_reflections.fiber is None
    existing = catalog.profile_transport_numerics.get(name, {})
    table = {**existing, "cxr_fiber": {"cutoff_invang": 6.0}}
    custom = replace(
        catalog,
        profile_transport_numerics=MappingProxyType(
            {**catalog.profile_transport_numerics, name: MappingProxyType(table)}
        ),
    )
    numerics = resolve_pixel_scene(custom, "hopg", name)[6]
    assert numerics.cxr_reflections.fiber == FiberAverage(cutoff_invang=6.0)


def json_payload(sweep, settings):
    from pyrite.campaign.profiles import dataset_identity

    return json.dumps(dataset_identity("hopg", settings, sweep))


def test_profile_numerics_validate_the_fiber_table():
    from pyrite._numerics import validate_profile_numerics

    validate_profile_numerics({"cxr_fiber": {"cutoff_invang": 6.0}})
    with pytest.raises(TypeError):
        validate_profile_numerics({"cxr_fiber": {"cutoff": 6.0}})
    with pytest.raises(ValueError, match="owns the reflection domain"):
        validate_profile_numerics({"cxr_fiber": {"cutoff_invang": 6.0}, "n_families": 3})
