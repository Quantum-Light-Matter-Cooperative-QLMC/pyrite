"""Unit-bearing physical configuration inputs normalize at their constructors."""

import pickle
import subprocess
import sys
from dataclasses import asdict, fields, replace

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


def test_numeric_beam_paths_do_not_import_pint() -> None:
    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import builtins, sys; original=builtins.__import__; "
            "builtins.__import__=lambda name,*a,**k: "
            "(_ for _ in ()).throw(AssertionError('Pint imported on numeric path')) "
            "if name.split('.')[0]=='pint' else original(name,*a,**k); "
            "import pyrite; from dataclasses import replace, asdict; "
            "from pyrite.beams import Beam, Spot, PointBunch; "
            "beam=Beam(60, transverse=Spot(), longitudinal=PointBunch()); "
            "assert replace(beam).energy_keV == 60; "
            "assert Beam.from_spec(beam.to_spec()).energy_keV == 60; "
            "assert asdict(beam)['energy_keV'] == 60; "
            "assert 'pint' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_check.returncode == 0, import_check.stderr


def test_semantic_beam_energy_preserves_numeric_records_and_replacement() -> None:
    options = dict(transverse=Spot(), longitudinal=PointBunch())
    numeric = Beam(60, **options)
    quantity = Beam(kinetic_energy=0.06 * ureg.MeV, **options)
    assert quantity == numeric
    assert asdict(quantity) == asdict(numeric)
    assert "kinetic_energy" not in {item.name for item in fields(quantity)}
    assert quantity.to_spec() == numeric.to_spec()
    assert Beam.from_spec(quantity.to_spec()) == numeric
    assert pickle.loads(pickle.dumps(quantity)) == numeric
    assert quantity.kinetic_energy.to("eV").magnitude == 60_000
    assert replace(quantity) == numeric
    assert replace(quantity, rep_rate_hz=100).energy_keV == 60
    assert replace(quantity, energy_keV=80).kinetic_energy == 80 * ureg.keV
    changed = quantity.with_kinetic_energy(0.08 * ureg.MeV)
    assert changed == replace(numeric, energy_keV=80)
    assert quantity.energy_keV == 60
    assert Beam(**options).energy_keV == (30.0, 45.0, 60.0)
    positional = Beam(60, 0.1, 2000, 2, None, None, **options)
    assert positional.energy_spread_frac == 0.1
    assert positional.rep_rate_hz == 2000
    assert positional.bunch_charge_pc == 2


def test_semantic_energy_arrays_and_external_registry_have_independent_readback() -> None:
    external = pint.UnitRegistry()
    source = external.Quantity(np.array([0.03, 0.045, 0.06]), "MeV")
    beam = Beam(kinetic_energy=source, transverse=Spot(), longitudinal=PointBunch())
    assert np.allclose(beam.energy_keV, [30, 45, 60])
    readback = beam.kinetic_energy
    assert readback._REGISTRY is ureg
    assert readback.units == ureg.keV
    source.magnitude[0] = 1
    readback.magnitude[1] = 1
    assert np.allclose(beam.energy_keV, [30, 45, 60])
    assert np.allclose(beam.to_spec().energy_keV, [30, 45, 60])
    changed = beam.with_kinetic_energy(Q_([60, 80], "keV"))
    assert np.array_equal(changed.energy_keV, [60, 80])
    numeric = Beam(energy_keV=[30, 45], transverse=Spot(), longitudinal=PointBunch())
    numeric.kinetic_energy.magnitude[0] = 0
    assert numeric.energy_keV == [30, 45]


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (60, TypeError),
        ("60 keV", TypeError),
        (np.array([30, 60]), TypeError),
        ([30 * ureg.keV, 60 * ureg.keV], TypeError),
        (ureg.Quantity(1, "dimensionless"), ValueError),
        (1 * ureg.mm, ValueError),
        (Q_([[30, 60]], "keV"), TypeError),
        (Q_(np.array([True]), "keV"), TypeError),
        (Q_(1 + 1j, "keV"), TypeError),
        (Q_([], "keV"), ValueError),
        (Q_([30, np.nan], "keV"), ValueError),
        (np.inf * ureg.keV, ValueError),
        (0 * ureg.keV, ValueError),
        (-1 * ureg.keV, ValueError),
    ],
)
def test_semantic_energy_rejects_invalid_inputs(value: object, error: type[Exception]) -> None:
    beam = Beam(60, transverse=Spot(), longitudinal=PointBunch())
    with pytest.raises(error, match="kinetic_energy"):
        Beam(kinetic_energy=value, transverse=Spot(), longitudinal=PointBunch())
    with pytest.raises(error, match="kinetic_energy"):
        beam.with_kinetic_energy(value)


def test_semantic_and_numeric_energy_arguments_conflict_even_when_equivalent() -> None:
    options = dict(transverse=Spot(), longitudinal=PointBunch())
    with pytest.raises(TypeError, match="energy_keV.*kinetic_energy"):
        Beam(energy_keV=60, kinetic_energy=60 * ureg.keV, **options)
    with pytest.raises(TypeError, match="energy_keV.*kinetic_energy"):
        Beam(60, kinetic_energy=60 * ureg.keV, **options)
    beam = Beam(kinetic_energy=60 * ureg.keV, **options)
    with pytest.raises(TypeError, match="energy_keV.*kinetic_energy"):
        replace(beam, kinetic_energy=80 * ureg.keV)


@pytest.mark.parametrize("magnitude", ["1e400", "1e-400"])
def test_semantic_scalar_energy_must_remain_finite_positive_as_python_float(
    magnitude: str,
) -> None:
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip("longdouble has no wider exponent range than Python float")
    value = np.longdouble(magnitude)
    assert np.isfinite(value) and value > 0
    quantity = Q_(value, "keV")
    options = dict(transverse=Spot(), longitudinal=PointBunch())
    with pytest.raises(ValueError, match="kinetic_energy.*finite.*positive"):
        Beam(kinetic_energy=quantity, **options)
    with pytest.raises(ValueError, match="kinetic_energy.*finite.*positive"):
        Beam(60, **options).with_kinetic_energy(quantity)


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


@pytest.mark.parametrize(
    ("quantity", "kelvin"),
    [
        (ureg.Quantity(293.15, "kelvin"), 293.15),
        (ureg.Quantity(20, "degC"), 293.15),
        (ureg.Quantity(68, "degF"), 293.15),
        (ureg.Quantity(527.67, "degR"), 293.15),
        (ureg.Quantity(20, "millikelvin"), 0.02),
    ],
)
def test_absolute_temperature_kind_converts_offsets(quantity, kelvin):
    from pyrite._units import normalize_unit

    got = normalize_unit(quantity, "kelvin", field="T", expected="temperature", kind="temperature")
    assert got == pytest.approx(kelvin)
    assert (
        normalize_unit(300, "kelvin", field="T", expected="temperature", kind="temperature") == 300
    )


@pytest.mark.parametrize("unit", ["delta_degC", "delta_degF"])
def test_absolute_temperature_kind_refuses_temperature_differences(unit):
    from pyrite._units import normalize_unit, normalize_unit_input

    with pytest.raises(ValueError, match="T received temperature difference"):
        normalize_unit(
            ureg.Quantity(20, unit), "kelvin", field="T", expected="temperature", kind="temperature"
        )
    with pytest.raises(ValueError, match="T received temperature difference"):
        normalize_unit_input(
            f"20 {unit}", "kelvin", field="T", expected="temperature", kind="temperature"
        )
    assert normalize_unit_input(
        "20 degC", "kelvin", field="T", expected="temperature", kind="temperature"
    ) == pytest.approx(293.15)
