"""Small resolved configurations pinned before issue #485 adapter changes."""

from dataclasses import replace

import numpy as np

from pyrite.campaign.config import default_settings
from pyrite.campaign.geometry import Footprint, Stack
from pyrite.campaign.profiles import case_content_key, dataset_identity
from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.detectors import Detector, EnergyBins


def resolved_baseline(kind):
    """Return JSON-ready mapping and both identities with explicit data markers.

    No installed external-table inventory participates: the two fake manifests
    exercise canonical digest encoding, not table physics or data availability.
    """
    sweep = Sweep(
        material="silicon",
        beam=BeamSpec(energy_keV=40.0),
        thickness_ang=2e4,
        tilt_deg=30.0,
        detector=Detector(
            energy_bins=EnergyBins(
                cxr=np.array([1500.0, 1750.0, 2000.0]),
                brem=np.array([5000.0, 10000.0, 15000.0]),
            )
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
    """Exact CPU regression; fixture provenance lives beside the JSON."""
    import json
    from pathlib import Path

    expected = json.loads(
        (Path(__file__).parents[1] / "data/physics-baseline/outputs.json").read_text()
    )[kind]
    assert actual == expected
