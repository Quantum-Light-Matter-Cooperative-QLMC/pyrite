"""Beam holds one sub-object per kind of property and lowers to the flat record."""

import numpy as np
import pytest

import pyrite as pr
from pyrite.campaign.beam import BunchLength, GdfSource, Offsets, PointBunch, PointSpot, Spot
from pyrite.campaign.longitudinal import LongitudinalDistribution
from pyrite.campaign.sweep import BeamSpec, build_cases
from pyrite.campaign.sweep import Sweep as LegacySweep
from pyrite.montecarlo.transverse import TransverseDistribution

_CS = TransverseDistribution(normalized_emittance_x_mm_mrad=1.0, beta_twiss_x_m=0.1)


def _case(beam):
    (case,) = build_cases(
        LegacySweep(material="hopg", beam=beam, thickness_ang=1e4, tilt_deg=30.0),
        n_electrons=1,
        n_electrons_brem=1,
        bremsstrahlung_model="eedl",
    )
    return case


@pytest.mark.parametrize(
    ("beam", "spec"),
    [
        (
            pr.Beam(energy_keV=30.0, transverse=Spot(0.5, 0.25), longitudinal=PointBunch()),
            BeamSpec(energy_keV=30.0, transverse_fwhm_x_mm=0.5, transverse_fwhm_y_mm=0.25),
        ),
        (
            pr.Beam(energy_keV=30.0, transverse=PointSpot(), longitudinal=PointBunch()),
            BeamSpec(energy_keV=30.0, transverse_fwhm_x_mm=None, transverse_fwhm_y_mm=None),
        ),
        (
            pr.Beam(energy_keV=30.0, transverse=Spot(), longitudinal=BunchLength(100.0)),
            BeamSpec(energy_keV=30.0, bunch_length_fs=100.0),
        ),
        (
            pr.Beam(energy_keV=30.0, transverse=Spot(), longitudinal=Offsets((1.0, -1.0))),
            BeamSpec(energy_keV=30.0, long_offsets_fs=(1.0, -1.0)),
        ),
        (
            pr.Beam(energy_keV=30.0, transverse=_CS, longitudinal=PointBunch()),
            BeamSpec.with_transverse(_CS, energy_keV=30.0),
        ),
        (
            pr.Beam(
                energy_keV=30.0,
                transverse=Spot(),
                longitudinal=LongitudinalDistribution("gaussian", 50.0),
            ),
            BeamSpec(energy_keV=30.0, longitudinal=LongitudinalDistribution("gaussian", 50.0)),
        ),
    ],
)
def test_lowering_matches_the_flat_record_and_round_trips(beam, spec):
    assert beam.to_spec() == spec
    assert pr.Beam.from_spec(spec) == beam
    assert _case(beam)["beam_fwhm_mm"] == _case(spec)["beam_fwhm_mm"]
    assert _case(beam) == _case(spec)


def test_wrong_types_name_the_allowed_options():
    with pytest.raises(TypeError, match="Spot, PointSpot, or TransverseDistribution"):
        pr.Beam(transverse=1.0, longitudinal=PointBunch())
    with pytest.raises(TypeError, match="PointBunch, BunchLength"):
        pr.Beam(transverse=Spot(), longitudinal=100.0)
    with pytest.raises(TypeError):
        pr.Beam(
            transverse_fwhm_x_mm=1.0, transverse=pr.TransverseSpot(), longitudinal=pr.PointBunch()
        )  # flat spellings are gone
    with pytest.raises(TypeError):
        pr.Beam(divergence_mrad=1.0, transverse=pr.TransverseSpot(), longitudinal=pr.PointBunch())


def test_sub_object_validation():
    with pytest.raises(ValueError, match="fwhm_x_mm"):
        Spot(0.0)
    with pytest.raises(ValueError, match="rms_fs"):
        BunchLength(-1.0)
    with pytest.raises(ValueError, match="empty"):
        Offsets(())
    with pytest.raises(ValueError, match="normalization"):
        GdfSource("a.gdf", 0.0, normalization="x")


def test_gdf_source_excludes_analytic_phase_space():
    beam = pr.Beam.from_gdf("a.gdf", z_origin_m=0.0, time_s=1e-9, energy_keV=30.0)
    assert isinstance(beam.transverse, PointSpot) and beam.gdf is not None
    spec = beam.to_spec()
    assert spec.source == "gpt_gdf" and spec.gdf_z_origin_m == 0.0
    assert pr.Beam.from_spec(spec) == beam
    gdf = GdfSource("a.gdf", 0.0, time_s=1e-9)
    with pytest.raises(ValueError, match="carries its own phase space"):
        pr.Beam(gdf=gdf, transverse=Spot(), longitudinal=PointBunch())
    with pytest.raises(ValueError, match="carries its own phase space"):
        pr.Beam(gdf=gdf, transverse=PointSpot(), longitudinal=PointBunch(), energy_spread_frac=0.01)
    with pytest.raises(ValueError, match="rep_rate_hz"):
        pr.Beam.from_gdf("a.gdf", z_origin_m=0.0, normalization="gdf_charge", rep_rate_hz=0.0)


def test_derived_divergence_is_available_on_the_public_beam():
    beam = pr.Beam(energy_keV=30.0, transverse=_CS, longitudinal=PointBunch())
    low, high = beam.derived_divergence_mrad(30.0), beam.derived_divergence_mrad(3000.0)
    assert low is not None and high is not None and high[0] < low[0]
    assert (
        pr.Beam(
            energy_keV=30.0, transverse=Spot(), longitudinal=PointBunch()
        ).derived_divergence_mrad(30.0)
        is None
    )


def test_scene_and_sweep_axes_use_sub_objects():
    beam = pr.Beam(energy_keV=30.0, transverse=Spot(), longitudinal=PointBunch())
    scene = pr.Scene(beam=beam, target=pr.Slab("hopg", footprint=pr.Footprint(5.0, 5.0)))
    sweep = pr.Sweep(scene, {"beam.transverse": [Spot(0.5), Spot(1.0), PointSpot()]})
    assert [s.beam.transverse for _, s in sweep.expand()] == [Spot(0.5), Spot(1.0), PointSpot()]
    with pytest.raises(ValueError, match="no field"):
        pr.Sweep(scene, {"beam.transverse_fwhm_x_mm": [1.0]})
    assert np.isfinite(scene.beam.energy_keV)


def test_required_beam_distributions_and_removed_point_alias():
    import pyrite.campaign.beam as beam_module

    with pytest.raises(TypeError, match="transverse.*longitudinal"):
        pr.Beam(energy_keV=30.0)
    with pytest.raises(TypeError, match="longitudinal"):
        pr.Beam(energy_keV=30.0, transverse=PointSpot())
    with pytest.raises(TypeError, match="transverse"):
        pr.Beam(energy_keV=30.0, longitudinal=PointBunch())
    assert not hasattr(pr, "Point")
    assert not hasattr(beam_module, "Point")
