"""Small resolved configurations pinned before issue #485 adapter changes."""

from dataclasses import replace

import numpy as np

from pyrite.campaign.config import default_settings
from pyrite.campaign.geometry import Footprint, Stack
from pyrite.campaign.profiles import case_content_key, dataset_identity
from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.detectors import EnergyBins


def resolved_baseline(kind):
    """Return JSON-ready mapping and both identities with explicit data markers.

    No installed external-table inventory participates: the two fake manifests
    exercise canonical digest encoding, not table physics or data availability.
    """
    sweep = Sweep(
        material="silicon",
        beam=BeamSpec(energy_keV=40.0),
        thickness_ang=20000.0,
        tilt_deg=30.0,
        energy_bins=EnergyBins(
            cxr=np.array([1500.0, 1750.0, 2000.0]), brem=np.array([5000.0, 10000.0, 15000.0])
        ),
    )
    selections = dict(
        elastic_model="mott",
        inelastic_model="continuous",
        bremsstrahlung_model="eedl",
        radiative_model="uncoupled",
    )
    markers = {}
    if kind == "elsepa":
        selections["elastic_model"] = "elsepa"
        markers = {"elsepa:Si": "a" * 64}
    elif kind in ("coupled", "cascade"):
        selections.update(
            bremsstrahlung_model="bremslib",
            radiative_model="bremslib-soft-hard",
            radiative_cutoff_eV=1000.0,
        )
        markers = {"bremslib:Si": "b" * 64}
        if kind == "cascade":
            selections.update(
                inelastic_model="shell-soft-hard",
                inelastic_cutoff_eV=50.0,
                secondary_threshold_eV=1000.0,
                pair_production_model="penelope-2024",
                positron_transport=True,
            )
    elif kind == "multilayer":
        sweep = replace(
            sweep,
            target=Stack.on_substrate(
                "silicon",
                5000.0,
                "sio2",
                15000.0,
                tilt_deg=30.0,
                footprint=Footprint(5.0, 5.0),
            ),
        )
    elif kind == "coherent":
        selections["coherent_emission"] = True
        sweep = replace(
            sweep,
            beam=BeamSpec(
                energy_keV=40.0,
                bunch_charge_pc=3.204353268e-7,
                transverse_fwhm_x_mm=None,
                transverse_fwhm_y_mm=None,
            ),
            target=replace(sweep.target, footprint=None),
        )
    elif kind == "eedl-fallback":
        selections["bremsstrahlung_model"] = "auto"
    settings = replace(
        default_settings(),
        n_electrons=4,
        n_electrons_brem=4,
        **{key: value for key, value in selections.items() if key != "coherent_emission"},
        emission="both" if kind == "coherent" else "incoherent",
    )
    case = build_cases(sweep, 4, 4, **selections)[0]
    identity = dataset_identity(
        "silicon",
        "full",
        settings,
        sweep,
        xsgen_tables=markers,
        bremsstrahlung_model="eedl"
        if kind == "eedl-fallback"
        else selections["bremsstrahlung_model"],
    )
    return {
        "case": case.to_dict(),
        "case_content_key": case_content_key(case, xsgen_tables=markers),
        "parameter_sha256": identity["parameter_sha256"],
    }


def transport_signature(segments):
    """Observable event counts plus seeded flight/energy samples; no private state."""
    names = ("E_end_keV", "L_ang", "r_mid", "event_kind", "electron_id")
    return {
        "rows": len(segments["L_ang"]),
        "n_backscattered": int(segments["n_backscattered"]),
        "n_transmitted": int(segments["n_transmitted"]),
        "path_ang": float(np.sum(segments["L_ang"])),
        **{name: segments[name][:8].tolist() for name in names},
    }


def assert_output_baseline(kind, actual):
    """Seeded CPU regression with fp64 roundoff across CPU math implementations."""
    import json
    from pathlib import Path

    expected = json.loads(
        (Path(__file__).parents[1] / "data/physics-baseline/outputs.json").read_text()
    )[kind]
    _assert_output_values(actual, expected, kind)


def _assert_output_values(actual, expected, path):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for name, value in expected.items():
            _assert_output_values(actual[name], value, f"{path}.{name}")
        return

    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.shape == expected_array.shape, path
    if expected_array.dtype.kind in "iu":
        np.testing.assert_array_equal(actual_array, expected_array, err_msg=path)
    else:
        # These fixtures accumulate up to a few thousand fp64 contributions.
        # Allow accumulated roundoff (roughly thousands of machine epsilons),
        # but no absolute floor that could mask tiny spectra or nonzero output.
        # Coherent fields also accumulate phases of order 1e5 radians (keV
        # photons over 1e4-1e5 Ang flights). Their fp64 phase error is of order
        # 1e5 * eps; intensity/interference needs a separate 1e-10 allowance.
        rtol = 1e-10 if path.endswith(".spec_coherent") else 1e-12
        np.testing.assert_allclose(
            actual_array, expected_array, rtol=rtol, atol=0, equal_nan=False, err_msg=path
        )


def resolved_channeling_baseline():
    """Return the planar channeling family (#506): resolved mapping and content key.

    Built through the public Scene path, the only one that can select
    channeling (profiles and checkpoints cannot, #507). Every selector is
    explicit so no installed table inventory participates. The beam lies in
    the Si (1-10) planes, 3 deg from the (110) face normal and far from a
    low-index axis, so the gates neither refuse nor warn.
    """
    import pyrite as pr
    from pyrite.api import build_case

    physics = pr.Physics(
        elastic_model="mott",
        inelastic_model="continuous",
        bremsstrahlung_model="eedl",
        radiative_model="uncoupled",
        channeling_model="planar",
        channeling_plane=(1, -1, 0),
    )
    scene = pr.Scene(
        pr.Beam(energy_keV=5000.0, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
        pr.Slab(
            "silicon",
            thickness_ang=20000.0,
            tilt_deg=3.0,
            tilt_azim_deg=45.0,
            footprint=Footprint(5.0, 5.0),
        ),
        pr.Detector(),
        physics=physics,
    )
    numerics = pr.Numerics(
        cxr_trials=4,
        brem_trials=4,
        energy_bins=EnergyBins(
            cxr=np.array([1500.0, 1750.0, 2000.0]), brem=np.array([5000.0, 10000.0, 15000.0])
        ),
    )
    case = build_case(scene, numerics)
    return {"case": case.to_dict(), "case_content_key": case_content_key(case)}
