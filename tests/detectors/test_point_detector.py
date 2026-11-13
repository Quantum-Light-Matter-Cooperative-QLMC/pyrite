"""PointDetector: one observation direction, zero angular acceptance (#994)."""

import pickle
from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite.campaign.lowering import build_case
from pyrite.config import FixedSampling
from pyrite.detectors import AngularAcceptance, Detector, EnergyBins, PointDetector
from pyrite.units import Q_


def test_defaults_on_axis_with_zero_acceptance():
    point = PointDetector()
    assert isinstance(point, Detector)
    assert point.observation_angle_deg == 0.0
    assert (point.polar_acceptance_deg, point.azimuthal_acceptance_deg) == (0.0, 0.0)
    assert point.solid_angle_sr == 0.0
    assert point.acceptance is None
    assert PointDetector(observation_angle=Q_(0.5, "rad")).observation_angle_deg == pytest.approx(
        np.rad2deg(0.5)
    )


def test_copies_stay_point_detectors():
    point = PointDetector(30.0)
    assert pickle.loads(pickle.dumps(point)) == point
    assert type(pickle.loads(pickle.dumps(point))) is PointDetector
    moved = replace(point, observation_angle_deg=45.0)
    assert type(moved) is PointDetector and moved.solid_angle_sr == 0.0
    turned = point.with_observation_angle(Q_(10.0, "degree"))
    assert type(turned) is PointDetector and turned.observation_angle_deg == 10.0
    assert point != Detector(30.0)


def test_an_acceptance_is_refused():
    with pytest.raises(TypeError, match="no angular acceptance"):
        PointDetector(acceptance=AngularAcceptance(polar_deg=1.0, azimuthal_deg=1.0))
    with pytest.raises(TypeError, match="no angular acceptance"):
        PointDetector(_legacy_acceptance=(1.0, None))


def test_lowers_to_a_zero_aperture_case():
    beam = pr.Beam(energy_keV=100.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch())
    target = pr.geometry.Slab(
        "silicon", thickness_ang=5000.0, tilt_deg=30.0, width=None, height=None
    )
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=2, brem_histories=2),
        energy_bins=EnergyBins(
            cxr=np.linspace(1500.0, 3000.0, 16), brem=np.array([1000.0, 4000.0])
        ),
    )
    case = build_case(pr.Scene(beam, target, PointDetector(20.0)), numerics)
    assert case["theta_obs_rad"] == pytest.approx(np.deg2rad(20.0))
    assert case["dtheta_obs_rad"] == 0.0
    assert case["domega_sr"] == 0.0


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"kind": "line"}, "must be 'point'"),
        ({"kind": "point", "polar_acceptance_deg": 1.0}, "no angular acceptance"),
        ({"kind": "point", "solid_angle_sr": 1e-3}, "no angular acceptance"),
        ({"kind": "point", "shape": [2, 2]}, "pixel detector acceptance"),
    ],
)
def test_profile_point_kind_is_validated_at_load(tmp_path, row, message):
    """#624: a catalog ``kind = "point"`` excludes every acceptance field."""
    import tomlkit

    from pyrite._catalog_layout import bundled_catalog, read_text
    from pyrite.materials import MaterialConfigError
    from pyrite.materials.catalog import load_material_catalog

    document = tomlkit.parse(read_text(bundled_catalog()))
    document["detectors"]["bad"] = row
    path = tmp_path / "catalog.toml"
    path.write_text(tomlkit.dumps(document))
    with pytest.raises(MaterialConfigError, match=message):
        load_material_catalog(path)


def test_packaged_point_detector_resolves_to_point_detector():
    from pyrite.campaign.config import catalog_detector
    from pyrite.materials import CATALOG

    assert CATALOG.detectors["point"] == {"kind": "point", "observation_angle_deg": 0.0}
    assert type(catalog_detector("channeling_demo")) is Detector
    spec = dict(CATALOG.detectors["point"])
    from pyrite.campaign.profile_detectors import _scalar_detector

    assert _scalar_detector(spec) == PointDetector()
