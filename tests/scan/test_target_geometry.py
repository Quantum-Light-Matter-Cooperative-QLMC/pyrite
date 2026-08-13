"""Target geometry objects: construction-time validity and lowering to case keys.

The equivalence tests pin ``lower()`` against what ``build_cases`` produces
today, so slice C is a rewire rather than a behaviour change.
"""

import numpy as np
import pytest

from pyrite.campaign.geometry import (
    BlazedGrooves,
    Footprint,
    Layer,
    Slab,
    Stack,
    crystal_params,
)
from pyrite.campaign.sweep import MATERIAL_LABELS, BeamSpec, Sweep, build_cases
from pyrite.detectors import DetectorSpec
from pyrite.materials import LayerSpec

MATERIAL = "mose2"
LABEL = MATERIAL_LABELS[MATERIAL]


def _lower(target, material=MATERIAL, n_families=4):
    cp = crystal_params(material, n_families)
    return target.lower(cp, label=MATERIAL_LABELS[material], beam_uvw=cp["beam_uvw"])


def _geometry_of(case):
    """The geometry half of a built Case, as ``LoweredTarget.case_keys`` reports it."""
    keys = dict(
        thickness_ang=case["thickness_ang"],
        crystal_width_mm=case["crystal_width_mm"],
        crystal_height_mm=case["crystal_height_mm"],
        tilt_deg=case["tilt_deg"],
        tilt_azim_deg=case["tilt_azim_deg"],
        abs_layers=case["abs_layers"],
        layer_radiators=case["layer_radiators"],
    )
    if case.get("groove_spacing_ang") is not None:
        keys["groove_spacing_ang"] = case["groove_spacing_ang"]
    return keys


# ---- construction-time validity ---------------------------------------------


def test_footprint_makes_the_pairing_rule_unrepresentable():
    with pytest.raises(TypeError):
        Footprint(5.0)  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_footprint_rejects_nonpositive_and_nonfinite(bad):
    with pytest.raises(ValueError, match="Footprint.width_mm must be finite and positive"):
        Footprint(bad, 5.0)
    with pytest.raises(ValueError, match="Footprint.height_mm must be finite and positive"):
        Footprint(5.0, bad)


def test_footprint_accepts_swept_dimensions():
    assert Footprint([4.0, 5.0], 5.0)._pairs() == [(4.0, 5.0), (5.0, 5.0)]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf")])
def test_blazed_grooves_reject_nonpositive_spacing(bad):
    with pytest.raises(ValueError, match="spacing_ang must be finite and positive"):
        BlazedGrooves(bad)


def test_blazed_grooves_are_not_sweepable():
    with pytest.raises(TypeError, match="spacing_ang must be a single number"):
        BlazedGrooves([1e4, 2e4])  # type: ignore[arg-type]


def test_slab_rejects_empty_material_and_bad_thickness():
    with pytest.raises(ValueError, match="Slab.material must be a non-empty material key"):
        Slab("  ")
    with pytest.raises(ValueError, match="Slab.thickness_ang must be finite and positive"):
        Slab(MATERIAL, thickness_ang=0.0)


def test_slab_bans_the_degenerate_angles():
    with pytest.raises(ValueError, match="polar tilt_deg == 0 is disallowed"):
        Slab(MATERIAL, tilt_deg=0.0)
    with pytest.raises(ValueError, match="tilt_azim_deg == 90 is disallowed"):
        Slab(MATERIAL, tilt_azim_deg=90.0)


def test_slab_allows_normal_incidence_for_transport_only_studies():
    assert Slab(MATERIAL, tilt_deg=0.0, allow_normal_incidence=True).tilt_deg == 0.0
    with pytest.raises(ValueError, match="tilt_azim_deg == 90 is disallowed"):
        Slab(MATERIAL, tilt_azim_deg=90.0, allow_normal_incidence=True)


def test_angles_are_quantized_before_they_are_validated():
    """0.2 deg quantizes to 0.0, so the tilt ban must fire on the quantized value."""
    with pytest.raises(ValueError, match="polar tilt_deg == 0 is disallowed"):
        Slab(MATERIAL, tilt_deg=0.2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tilt_azim_deg": 0.0}, "tilt_azim_deg == 180"),
        ({"tilt_azim_deg": 180.0, "tilt_deg": 95.0}, "0 < tilt_deg < 90"),
    ],
)
def test_grooves_require_their_angle_conjunction(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Slab(MATERIAL, entrance_face=BlazedGrooves(2.0e4), **kwargs)


def test_grooves_are_compatible_with_a_finite_footprint():
    target = Slab(
        MATERIAL,
        tilt_deg=45.0,
        tilt_azim_deg=180.0,
        entrance_face=BlazedGrooves(2.0e4),
        footprint=Footprint(5.0, 5.0),
    )
    lowered = _lower(target)[0]
    assert lowered.crystal_width_mm == 5.0
    assert lowered.groove_spacing_ang == 2.0e4


def test_grooves_forbid_a_stack_by_construction():
    """Constraint 8 is a type constraint: Stack has no entrance_face field."""
    with pytest.raises(TypeError):
        Stack(  # type: ignore[call-arg]
            layers=(Layer(MATERIAL, 2e4), Layer("sio2", 5e6)),
            entrance_face=BlazedGrooves(2.0e4),
        )


def test_grooves_require_a_90_degree_observation_angle():
    target = Slab(MATERIAL, tilt_deg=45.0, tilt_azim_deg=180.0, entrance_face=BlazedGrooves(2.0e4))
    target.validate_against(DetectorSpec(observation_angle_deg=90.0))
    with pytest.raises(ValueError, match="theta_obs_deg == 90"):
        target.validate_against(DetectorSpec(observation_angle_deg=45.0))


def test_ungrooved_target_accepts_any_observation_angle():
    Slab(MATERIAL).validate_against(DetectorSpec(observation_angle_deg=45.0))


def test_layer_validates_material_and_thickness():
    with pytest.raises(ValueError, match="Layer.material must be a non-empty material key"):
        Layer("", 5e6)
    with pytest.raises(ValueError, match=r"Layer.thickness_ang \('sio2'\)"):
        Layer("sio2", -1.0)
    with pytest.raises(ValueError, match="3-component"):
        Layer("silicon", 5e6, beam_uvw=(1, 1))  # type: ignore[arg-type]


def test_stack_needs_a_film_plus_something_beneath_it():
    with pytest.raises(ValueError, match="use Slab for a free-standing film"):
        Stack(layers=(Layer(MATERIAL, 2e4),))


def test_only_the_film_may_sweep_thickness():
    with pytest.raises(ValueError, match="only the film may sweep thickness_ang"):
        Stack(layers=(Layer(MATERIAL, 2e4), Layer("sio2", [1e6, 5e6])))
    Stack(layers=(Layer(MATERIAL, [1e4, 2e4]), Layer("sio2", 5e6)))


def test_stack_exposes_the_film_material():
    assert Stack(layers=(Layer(MATERIAL, 2e4), Layer("sio2", 5e6))).material == MATERIAL


# ---- lowering equivalence with build_cases ----------------------------------


def _sweep(**kwargs):
    base = dict(material=MATERIAL, beam=BeamSpec(energy_keV=30.0))
    return Sweep(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("sweep_kwargs", "target"),
    [
        (
            {},
            Slab(MATERIAL),
        ),
        (
            {"crystal_width_mm": None, "crystal_height_mm": None},
            Slab(MATERIAL, footprint=None),
        ),
        (
            {
                "thickness_ang": [1e4, 2e4],
                "tilt_deg": [15.0, 30.0],
                "tilt_azim_deg": [0.0, 45.0],
                "crystal_width_mm": [4.0, 5.0],
                "crystal_height_mm": 6.0,
            },
            Slab(
                MATERIAL,
                thickness_ang=[1e4, 2e4],
                tilt_deg=[15.0, 30.0],
                tilt_azim_deg=[0.0, 45.0],
                footprint=Footprint([4.0, 5.0], 6.0),
            ),
        ),
        (
            {"substrate": "sio2"},
            Stack(layers=(Layer(MATERIAL, 2e4), Layer("sio2", 5e6))),
        ),
        (
            {"substrate": "silicon", "substrate_thickness_ang": 1e6},
            Stack(layers=(Layer(MATERIAL, 2e4), Layer("silicon", 1e6))),
        ),
        (
            {
                "thickness_ang": [1e4, 2e4],
                "stack": (
                    LayerSpec("sio2", 2850.0),
                    LayerSpec("silicon", 5e6, beam_uvw=(1, 1, 1), azimuth_deg=30.0),
                ),
            },
            Stack(
                layers=(
                    Layer(MATERIAL, [1e4, 2e4]),
                    Layer("sio2", 2850.0),
                    Layer("silicon", 5e6, beam_uvw=(1, 1, 1), azimuth_deg=30.0),
                )
            ),
        ),
        (
            {
                "tilt_deg": 45.0,
                "tilt_azim_deg": 180.0,
                "groove_spacing_ang": 2.0e4,
            },
            Slab(
                MATERIAL,
                tilt_deg=45.0,
                tilt_azim_deg=180.0,
                entrance_face=BlazedGrooves(2.0e4),
            ),
        ),
    ],
    ids=[
        "default",
        "infinite-slab",
        "full-product",
        "substrate",
        "crystalline-substrate",
        "multilayer-stack",
        "grooved",
    ],
)
def test_lower_reproduces_the_build_cases_geometry(sweep_kwargs, target):
    cases = build_cases(_sweep(**sweep_kwargs))
    lowered = _lower(target)

    assert len(lowered) == len(cases)  # one beam energy, so the products align 1:1
    for case, geometry in zip(cases, lowered, strict=True):
        assert geometry.name == case["name"]
        assert geometry.case_keys() == _geometry_of(case)


def test_lower_omits_the_groove_key_for_an_ungrooved_target():
    """The absent key -- not a None -- is what keeps existing case payloads bit-for-bit."""
    assert "groove_spacing_ang" not in _lower(Slab(MATERIAL))[0].case_keys()


def test_lower_preserves_the_build_cases_product_order():
    target = Slab(
        MATERIAL,
        thickness_ang=[1e4, 2e4],
        tilt_deg=[15.0, 30.0],
        tilt_azim_deg=[0.0, 45.0],
        footprint=Footprint([4.0, 5.0], 6.0),
    )
    axes = [
        (g.thickness_ang, g.tilt_deg, g.tilt_azim_deg, g.crystal_width_mm) for g in _lower(target)
    ]
    assert axes[0] == (1e4, 15.0, 0.0, 4.0)
    assert axes[1] == (1e4, 15.0, 0.0, 5.0)
    assert axes[2] == (1e4, 15.0, 45.0, 4.0)
    assert axes[4] == (1e4, 30.0, 0.0, 4.0)
    assert axes[8] == (2e4, 15.0, 0.0, 4.0)


def test_lower_deduplicates_quantized_angles_like_build_cases():
    """0.1 and 0.2 both quantize to 0.0; stable-unique collapses them to one case."""
    target = Slab(MATERIAL, tilt_deg=[29.9, 30.1], allow_normal_incidence=False)
    assert [g.tilt_deg for g in _lower(target)] == [30.0]


def test_stack_radiators_align_with_the_absorber_stack():
    target = Stack(
        layers=(Layer(MATERIAL, 2e4), Layer("sio2", 2850.0), Layer("silicon", 5e6)),
    )
    geometry = _lower(target)[0]
    assert len(geometry.abs_layers) == 3
    assert len(geometry.layer_radiators) == 3
    assert geometry.layer_radiators[1] is None  # amorphous sio2 only absorbs
    assert geometry.layer_radiators[2]["crystal"] == "silicon"
    assert np.isclose(geometry.abs_layers[0][1], 2e4)
