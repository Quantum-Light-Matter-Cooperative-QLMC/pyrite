"""Unit-bearing physical configuration inputs normalize at their constructors."""

import subprocess
import sys

import numpy as np
import pint
import pytest

from pyrite.beams import Beam, BunchFrontTilt, BunchLength, PointBunch, Spot
from pyrite.config import Numerics, Physics
from pyrite.detectors import AngularAcceptance, Detector, EnergyBins, PixelGrid, PlanarPose
from pyrite.geometry import RectangularBody, SubstanceRef, Target
from pyrite.observations import Acquisition
from pyrite.units import Q_, ureg


def test_shared_registry_converts_beam_and_accepts_external_registry() -> None:
    legacy = Beam(energy_keV=60.0, transverse=Spot(), longitudinal=PointBunch())
    quantity = Beam(energy_keV=0.06 * ureg.MeV, transverse=Spot(), longitudinal=PointBunch())
    assert quantity.energy_keV == legacy.energy_keV
    assert quantity.to_spec().energy_keV == legacy.to_spec().energy_keV

    external = pint.UnitRegistry()
    external_quantity = Beam(
        energy_keV=external.Quantity(60_000, "eV"),
        transverse=Spot(),
        longitudinal=PointBunch(),
    )
    assert external_quantity.energy_keV == legacy.energy_keV
    assert Q_(1, ureg.pC).to("pC").magnitude == 1


def test_pint_is_not_loaded_by_plain_import_and_missing_extra_has_guidance() -> None:
    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pyrite; assert 'pint' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_check.returncode == 0, import_check.stderr

    missing_extra = subprocess.run(
        [
            sys.executable,
            "-c",
            "import builtins; original=builtins.__import__; "
            "builtins.__import__=lambda name,*a,**k: "
            "(_ for _ in ()).throw(ModuleNotFoundError('pint')) if name=='pint' "
            "else original(name,*a,**k); "
            "from pyrite.units import ureg",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_extra.returncode != 0
    assert "pyrite-mc[units]" in missing_extra.stderr


def test_quantity_arrays_and_compound_geometry_normalize_to_plain_values() -> None:
    beam = Beam(
        energy_keV=np.array([0.03, 0.045, 0.06]) * ureg.MeV,
        transverse=Spot(),
        longitudinal=PointBunch(),
    )
    assert np.array_equal(beam.energy_keV, np.array([30.0, 45.0, 60.0]))
    assert not hasattr(beam.energy_keV, "units")

    pose = PlanarPose(center_mm=(0 * ureg.mm, 0 * ureg.mm, 10 * ureg.cm))
    assert pose.center_mm == (0.0, 0.0, 100.0)
    pixels = PixelGrid(shape=(256, 256), pitch_mm=(55 * ureg.um, 55 * ureg.um))
    assert pixels.pitch_mm == (0.055, 0.055)


def test_acquisition_and_angles_normalize_once() -> None:
    legacy = Acquisition(0.5, (0.0, 100.0, 200.0))
    quantity = Acquisition(
        exposure_s=500 * ureg.ms,
        measured_edges_eV=(0 * ureg.eV, 0.1 * ureg.keV, 0.2 * ureg.keV),
    )
    assert quantity == legacy
    assert isinstance(quantity.exposure_s, float)
    assert quantity.measured_edges_eV == legacy.measured_edges_eV

    assert Detector(0.5 * ureg.radian).observation_angle_deg == pytest.approx(np.rad2deg(0.5))
    acceptance = AngularAcceptance(0.1 * ureg.radian, 0.2 * ureg.radian)
    assert acceptance.polar_deg == pytest.approx(np.rad2deg(0.1))
    assert acceptance.azimuthal_deg == pytest.approx(np.rad2deg(0.2))


def test_incompatible_dimensions_and_plain_dimensionless_angles_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"Beam\.energy_keV.*millimeter.*energy.*keV"):
        Beam(energy_keV=1 * ureg.mm, transverse=Spot(), longitudinal=PointBunch())

    with pytest.raises(ValueError, match=r"Detector\.observation_angle_deg.*dimensionless.*angle"):
        Detector(1 * ureg.dimensionless)

    with pytest.raises(ValueError, match=r"Acquisition\.exposure_s.*electron_volt.*time.*'s'"):
        Acquisition(1 * ureg.eV, (0.0, 1.0))


def test_target_beam_timing_physics_and_numerics_normalize_units() -> None:
    body = RectangularBody(
        SubstanceRef(kind="crystal", key="mos2"),
        size_mm=(1 * ureg.cm, 1 * ureg.cm),
        thickness_mm=100 * ureg.um,
    )
    target = Target(body=body, tilt_deg=0.5 * ureg.radian)
    assert target.body.size_mm == (10.0, 10.0)
    assert target.body.thickness_mm == 0.1
    assert target.tilt_deg == pytest.approx(np.rad2deg(0.5))

    beam = Beam(
        energy_keV=60,
        transverse=Spot(fwhm_x_mm=500 * ureg.um),
        longitudinal=BunchLength(0.2 * ureg.ps),
        bunch_front_tilt=BunchFrontTilt(0.1 * ureg.radian),
    )
    assert beam.transverse.fwhm_x_mm == 0.5
    assert beam.longitudinal.rms_fs == 200.0
    assert beam.bunch_front_tilt.tilt_deg == pytest.approx(np.rad2deg(0.1))

    bins = EnergyBins(cxr=np.array([0.1, 0.2]) * ureg.keV)
    assert np.array_equal(bins.cxr, np.array([100.0, 200.0]))
    physics = Physics(inelastic_cutoff_eV=0.5 * ureg.keV)
    numerics = Numerics(channeling_dz_ang=0.025 * ureg.um)
    assert physics.inelastic_cutoff_eV == 500.0
    assert numerics.channeling_dz_ang == 250.0
