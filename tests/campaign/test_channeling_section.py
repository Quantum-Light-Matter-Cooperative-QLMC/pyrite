"""The public ``Channeling`` section and its private flat lowering (#1017)."""

import warnings
from dataclasses import FrozenInstanceError, replace

import pytest

import pyrite
from pyrite import Numerics, Physics, quantum
from pyrite.campaign import profiles
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.quantum import Channeling, ChannelingNumerics
from pyrite.quantum._admission import AxialAdmissionError
from pyrite.quantum._channeling import (
    CHANNELING_OFF_FIELDS,
    channeling_fields,
    channeling_from_fields,
    channeling_table_values,
    validate_channeling_run,
)
from pyrite.units import ureg

PLANAR = Channeling("planar", plane=(1, 1, 0))
UNCOUPLED = Physics(radiative_model="uncoupled")


def test_planar_section_lowers_to_the_stored_flat_record():
    assert channeling_fields(PLANAR) == {
        **CHANNELING_OFF_FIELDS,
        "channeling_model": "planar",
        "channeling_plane": (1, 1, 0),
    }
    assert channeling_fields(None) == CHANNELING_OFF_FIELDS
    custom = Channeling(
        "planar",
        plane=[1, -1, 0],
        numerics=ChannelingNumerics(12, 8, ureg.Quantity(10, "nm"), 2000.0),
        admission="accept-inaccurate",
    )
    lowered = channeling_fields(custom)
    assert lowered["channeling_plane"] == (1, -1, 0)
    assert (
        lowered["channeling_basis_size"],
        lowered["channeling_bloch_cells"],
        lowered["channeling_dz_ang"],
        lowered["channeling_zone_max_ang"],
        lowered["channeling_admission"],
    ) == (12, 8, 100.0, 2000.0, "accept-inaccurate")
    assert channeling_from_fields(lowered) == custom
    assert channeling_from_fields(CHANNELING_OFF_FIELDS) is None


def test_section_is_frozen_and_normalizes_miller_indices():
    assert PLANAR.plane == (1, 1, 0)
    with pytest.raises(FrozenInstanceError):
        PLANAR.model = "axial"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"model": "off"}, "channeling=None for off"),
        ({"model": "planar"}, "requires plane"),
        ({"model": "planar", "plane": (0, 0, 0)}, "nonzero"),
        ({"model": "planar", "plane": (1, 1, 0), "admission": "yes"}, "Channeling.admission"),
        ({"model": "axial"}, "requires axial"),
        ({"model": "axial", "plane": (1, 1, 0)}, "planar-only"),
        ({"model": "planar", "plane": (1, 1, 0), "axial": {"version": 2}}, "model='axial'"),
    ],
)
def test_invalid_sections_refuse(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Channeling(**kwargs)


def test_invalid_zone_numerics_refuse():
    with pytest.raises(ValueError, match="ChannelingNumerics.basis_size"):
        ChannelingNumerics(basis_size=0)
    with pytest.raises(ValueError, match="ChannelingNumerics.zone_max_ang"):
        ChannelingNumerics(zone_max_ang=-10.0)
    with pytest.raises(TypeError, match="ChannelingNumerics"):
        Channeling("planar", plane=(1, 1, 0), numerics=Numerics())


@pytest.mark.parametrize(
    "physics, energy_model, message",
    [
        (Physics(), "midpoint", "radiative_model='uncoupled'"),
        (UNCOUPLED, "frozen", "energy_model='midpoint'"),
        (replace(UNCOUPLED, emission=("incoherent", "coherent")), "midpoint", "incoherent"),
        (replace(UNCOUPLED, transition_radiation="history"), "midpoint", "transition_radiation"),
    ],
)
def test_run_compatibility_refusals(physics, energy_model, message):
    with pytest.raises(ValueError, match=message):
        validate_channeling_run(PLANAR, physics, energy_model)
    validate_channeling_run(None, physics, energy_model)


def test_removed_spellings_name_the_replacement():
    with pytest.raises(TypeError, match=r"Physics\(channeling_model=.*Channeling\(model="):
        Physics(channeling_model="planar")
    with pytest.raises(TypeError, match=r"Physics\(channeling_admission=.*admission="):
        Physics(channeling_admission="accept-inaccurate")
    with pytest.raises(TypeError, match=r"Numerics\(channeling_dz_ang=.*step_ang="):
        Numerics(channeling_dz_ang=100.0)
    for name in ("QuantumSetup", "QuantumRegion", "QuantumDynamics", "QuantumNumerics"):
        with pytest.raises(AttributeError, match=f"{name} was removed.*Channeling"):
            getattr(quantum, name)
        with pytest.raises(AttributeError, match=f"{name} was removed"):
            getattr(pyrite, name)
    assert pyrite.Channeling is Channeling


def test_profile_table_lowers_and_flat_keys_refuse():
    assert channeling_table_values(
        {"channeling": {"model": "planar", "plane": [1, -1, 0], "step_ang": 100.0}}
    ) == {
        "channeling_model": "planar",
        "channeling_plane": [1, -1, 0],
        "channeling_dz_ang": 100.0,
    }
    with pytest.raises(ValueError, match=r"channeling_plane was removed.*\[channeling\].*plane"):
        channeling_table_values({"channeling_plane": [1, 1, 0]})
    with pytest.raises(ValueError, match="unknown \\[channeling\\] key: dz_ang"):
        channeling_table_values({"channeling": {"dz_ang": 1.0}})


def test_settings_keep_the_flat_record():
    settings = replace(
        default_settings(),
        channeling_model="planar",
        channeling_plane=[1, 1, 0],
        radiative_model="uncoupled",
        radiative_cutoff_eV=None,
        energy_model="midpoint",
    )
    assert settings.channeling_plane == (1, 1, 0)
    with pytest.raises(ValueError, match="radiative_model='uncoupled'"):
        replace(settings, radiative_model="bremslib-soft-hard", radiative_cutoff_eV=1000.0)


def test_simulate_takes_channeling_and_refuses_axial_outside_the_envelope():
    from pyrite.campaign.lowering import build_case
    from pyrite.campaign.model import Scene

    scene = Scene(
        pyrite.Beam(
            energy_keV=30.0,
            transverse=pyrite.beams.Spot(),
            longitudinal=pyrite.beams.PointBunch(),
        ),
        pyrite.geometry.Slab(
            "silicon",
            thickness_ang=20_000.0,
            tilt_deg=3.0,
            footprint=pyrite.geometry.Footprint(5.0, 5.0),
        ),
    )
    with pytest.raises(TypeError, match="Channeling or None"):
        build_case(scene, channeling=Physics())
    with pytest.raises(AxialAdmissionError, match="#623"):
        build_case(
            scene,
            physics=UNCOUPLED,
            channeling=Channeling(
                "axial",
                axial={
                    "axis_uvw": [0, 0, 1],
                    "zone_max_ang": 1000.0,
                    "entry": {"version": 1, "kind": "gaussian", "waist_ang": 0.5},
                },
            ),
        )


def _profile_cases(material, profile):
    sweep = material_sweep(material, catalog_profile=profile)
    settings = profiles.profile_run_settings(default_settings(), profile, sweep=sweep)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pyrite import api

        cases = api.build_configured_cases(sweep, settings)
    return sweep, settings, cases


def test_channeling_demo_and_off_profile_identities_are_unchanged(identity_pin):
    """The #1017 spelling migration preserves identity within a model generation.

    Pins refreshed for #1191's ``segment-mean-v8-node-support-tail`` marker;
    restoring the v7 marker reproduces every pre-#1191 pin. The demo content
    key, silicon stem and parameter hash follow ``CHANNELING_MODEL`` v11 (#1148
    in-plane zone PXR beam; v10, #1166 zone-row PXR/DCR source); every
    pin, channeling-off included, follows ``DIFFRACTED_BREM_MODEL`` (#1166 DBR). The
    silicon pins carry the crystal-conditional ``structure_factor_model`` marker
    (#1177 measured bonding charge); hopg, which has no measured table, is unchanged.
    The bin-mean demo key follows ``CXR_BIN_REDUCTION_MODEL`` ``fourier-auto-v4``
    (#1192), as do the silicon stem and parameter hash; restoring
    ``line-major-auto-v3-series`` reproduces the prior pins.
    #1173 re-mints every pin for ``measured-shell-cutoff-v2``; unsupported
    channeling routes retain four-family scoring. #1200 re-mints them for
    ``measured-shell-cutoff-v3`` (table-sized selection calls), and #1201 for
    ``segment-mean-v9-edge-geometric-tables`` (line-table mesh). #642 re-mints
    the shell-transport pins for default restricted soft-loss straggling.
    #1254 re-mints the adaptive standard-profile pin for variance-contributors-v2.
    #1262 re-mints it for variance-threshold-v3 (separate VOV threshold).
    #1243 re-mints them for the ``resonance-population`` default bandwidth
    (case payload) and its ``line_bandwidth_model`` digest marker.
    #1253 re-mints it for decision-points-v4.
    #1265 re-mints it for the bin-mean shape step on the kinematic axis
    (``sinc-oversampled`` + ``shape_rtol``) and ``line_resolution_model`` v2.
    #1296 re-mints the silicon pins for ``CXR_BIN_REDUCTION_MODEL``
    ``fourier-auto-v6`` (FP32 direct phase factors on the device route).
    #1281 re-mints every pin for ``measured-shell-cutoff-v6`` (electron-prefix
    selection).
    #1297 re-mints the silicon pins for ``fourier-auto-v7`` (sorted warp-gather
    ES spreading).
    #1301 re-mints every pin for ``measured-shell-cutoff-v7`` (fiber-texture
    candidate domain; HOPG cases now select automatically).
    #1320 re-mints the silicon pins for ``fourier-auto-v8`` (over-cap device
    lines on near/far).
    #1390 re-mints the silicon pins for ``fourier-auto-v9`` (host non-uniform
    bin-mean axes on the numba near/far queue).
    """
    sweep, settings, cases = _profile_cases("silicon", "channeling_demo")
    assert cases[0].line_quadrature == "bin-mean"
    assert len(cases) == 1
    identity_pin(
        "case/silicon/channeling_demo/case_content_key", profiles.case_content_key(cases[0])
    )
    identity_pin(
        "stem/silicon/channeling_demo",
        profiles.named_profile_stem("silicon", catalog_profile="channeling_demo"),
    )
    identity = profiles.dataset_identity("silicon", settings, sweep)
    identity_pin("dataset/silicon/channeling_demo/parameter_sha256", identity["parameter_sha256"])
    sweep = material_sweep("hopg", catalog_profile="standard")
    settings = profiles.profile_run_settings(default_settings(), "standard", sweep=sweep)
    identity_pin(
        "dataset/hopg/standard/parameter_sha256",
        profiles.dataset_identity("hopg", settings, sweep)["parameter_sha256"],
    )
    identity_pin("stem/hopg/standard", profiles.named_profile_stem("hopg"))
