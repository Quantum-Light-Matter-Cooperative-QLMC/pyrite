"""Public beam-front descriptor, canonical spot name and case identity."""

from dataclasses import fields

import numpy as np
import pytest

import pyrite as pr
from pyrite.api import build_case
from pyrite.campaign.beam import Spot, TransverseSpot
from pyrite.campaign.sweep import BeamSpec


def _beam(front=None, **kwargs):
    return pr.Beam(
        energy_keV=30.0,
        transverse=pr.beams.TransverseSpot(0.001),
        longitudinal=pr.beams.LongitudinalDistribution("gaussian", 0.001),
        bunch_front_tilt=front,
        **kwargs,
    )


def test_canonical_spot_name_preserves_old_construction():
    assert pr.beams.TransverseSpot is TransverseSpot is Spot is pr.beams.Spot
    assert type(pr.beams.Spot(0.1)).__name__ == "TransverseSpot"
    assert pr.beams.TransverseSpot(0.1, 0.2).plane_y_mm == 0.2


@pytest.mark.parametrize("angle", [-90, 90, 91, -91, np.inf, np.nan])
def test_tilt_rejects_singular_or_nonfinite_angles(angle):
    with pytest.raises(ValueError, match="strictly between"):
        pr.beams.BunchFrontTilt(angle)


def test_tilt_round_trip_and_zero_canonicalization():
    beam = _beam(pr.beams.BunchFrontTilt(25.0, 90.0))
    assert pr.Beam.from_spec(beam.to_spec()) == beam
    spec = beam.to_spec()
    assert BeamSpec(**{f.name: getattr(spec, f.name) for f in fields(spec)}) == spec
    assert _beam(pr.beams.BunchFrontTilt(0.0, 90.0)) == _beam()
    with pytest.raises(ValueError, match="azimuth"):
        pr.beams.BunchFrontTilt(10, np.nan)


@pytest.mark.parametrize("energy", [0.0, -2000.0, np.inf, np.nan])
def test_timing_gradient_requires_physical_kinetic_energy(energy):
    with pytest.raises(ValueError, match="finite positive beam energy"):
        pr.beams.BunchFrontTilt(10).gradient(energy)


def test_profile_field_parser_and_flat_mapping_round_trip():
    from pyrite.materials._beam_detector_parse import _parse_profile_beam
    from pyrite.materials._catalog_decode import _Errors

    errors = _Errors()
    fields = _parse_profile_beam(
        {"bunch_length_fs": 0.001, "bunch_front_tilt": {"tilt_deg": -30.0, "azimuth_deg": 90.0}},
        "profiles.test.beam",
        errors,
    )
    assert not errors.items and fields is not None
    spec = BeamSpec(energy_keV=30.0, **fields)
    assert pr.Beam.from_spec(spec).bunch_front_tilt == pr.beams.BunchFrontTilt(-30.0, 90.0)
    bad = _Errors()
    _parse_profile_beam({"bunch_front_tilt": {"tilt_deg": 90.0}}, "beam", bad)
    assert bad.items and "tilt_deg" in bad.items[0]
    zero_errors = _Errors()
    zero = _parse_profile_beam({"bunch_front_tilt": {"tilt_deg": 0.0}}, "beam", zero_errors)
    assert zero is not None and not zero_errors.items
    assert BeamSpec(**zero).bunch_front_tilt is None


def test_flat_gdf_spec_cannot_silently_drop_an_imposed_front():
    spec = BeamSpec(
        energy_keV=30.0,
        transverse_fwhm_x_mm=None,
        transverse_fwhm_y_mm=None,
        source="gpt_gdf",
        gdf_path="absent.gdf",
        gdf_z_origin_m=0.0,
        bunch_front_tilt=pr.beams.BunchFrontTilt(10),
    )
    with pytest.raises(ValueError, match="incompatible"):
        pr.Beam.from_spec(spec)


def test_case_records_nonzero_tilt_and_keeps_zero_identity():
    def lower(front):
        return build_case(
            pr.Scene(
                beam=_beam(front),
                target=pr.geometry.Slab("hopg", thickness_ang=100.0, tilt_deg=45.0, footprint=None),
                numerics=pr.Numerics(cxr_trials=10, brem_trials=2),
            )
        )

    plain = lower(None)
    assert lower(pr.beams.BunchFrontTilt(0)) == plain
    tilted = lower(pr.beams.BunchFrontTilt(10.0, 90.0))
    assert tilted["bunch_front_tilt"] == {"tilt_deg": 10.0, "azimuth_deg": 90.0}
    assert tilted != plain


@pytest.mark.parametrize("kind", ["spread", "offsets", "uniform", "twiss", "gdf"])
def test_unsupported_source_combinations_fail_explicitly(kind):
    fields = dict(
        energy_keV=30.0,
        transverse=pr.beams.TransverseSpot(0.01),
        longitudinal=pr.beams.PointBunch(),
        bunch_front_tilt=pr.beams.BunchFrontTilt(10),
    )
    if kind == "spread":
        fields["energy_spread_frac"] = 0.01
    elif kind == "offsets":
        fields["longitudinal"] = pr.beams.Offsets((0.0, 1.0))
    elif kind == "uniform":
        fields["longitudinal"] = pr.beams.LongitudinalDistribution("gaussian", 1.0, shape="uniform")
    elif kind == "twiss":
        from pyrite.montecarlo.transverse import TransverseDistribution

        fields["transverse"] = TransverseDistribution(
            normalized_emittance_x_mm_mrad=1.0, beta_twiss_x_m=0.1
        )
    else:
        from pyrite.campaign.beam import GdfSource

        fields.update(gdf=GdfSource("absent.gdf", 0.0), transverse=pr.beams.PointSpot())
    with pytest.raises(ValueError, match="bunch-front tilt supports only"):
        pr.Beam(**fields)
