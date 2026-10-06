"""Factor diagnostics for Zhai SI Fig. S5b's 921 nm h-BN panel (#339).

Run on the lab worker through PyRITE remote plumbing, never as a local sweep.
This measures model sensitivities; the historical paper peaks are approximate
visual readings, not digitized measurements or regression targets.

Validation: zhai-hbn-921-detected
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pyrite._backend import BACKEND, REAL
from pyrite.detectors.response import convolve_detector, detector_efficiency
from pyrite.materials.crystal import reflection_coupling_tables
from pyrite.montecarlo import aperture_fwhm_eV, beta_from_keV, eds_fwhm_eV
from pyrite.montecarlo.spectrum.lines import _kernels, _setup
from pyrite.validation import anchor_figures as af


def response_stages(study, condition, spectrum):
    """Peak and in-window area after each response operator, with units."""
    if not np.isfinite(spectrum).all():
        raise ValueError(f"nonfinite spectrum at {condition.energy_keV:g} keV")
    energy = study.E_grid
    peak = float(energy[np.argmax(spectrum)])
    eds = float(eds_fwhm_eV(peak))
    aperture = float(
        aperture_fwhm_eV(
            peak,
            beta_from_keV(condition.energy_keV),
            study.theta_obs_rad,
            study.dtheta_obs_rad,
        )
    )
    scale = study.domega_sr * af.ZhaiAnchor.per_nA
    qe = detector_efficiency(energy)
    stages = {
        "source": spectrum,
        "eds_unit_qe": convolve_detector(energy, spectrum, eds),
        "eds_aperture_unit_qe": convolve_detector(energy, spectrum, np.hypot(eds, aperture)),
        "detected": af._supplementary_detected_spectrum(study, condition, spectrum) / scale,
    }
    return {
        "source_peak_eV": peak,
        "eds_fwhm_eV": eds,
        "aperture_fwhm_eV": aperture,
        "qe_at_source_peak": float(qe[np.argmax(spectrum)]),
        "solid_angle_sr": study.domega_sr,
        "stages": {
            name: {
                "peak_photons_per_eV_s_nA": float(values.max() * scale),
                "in_window_photons_per_s_nA": float(np.trapezoid(values, energy) * scale),
                "peak_eV": float(energy[np.argmax(values)]),
            }
            for name, values in stages.items()
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ne", type=int, default=200)
    parser.add_argument("--thickness-nm", type=float, default=921.0)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    study = af.supplementary_study("hbn")
    published_conditions = study.conditions_for(921.0)
    study = replace(study, conditions_by_thickness_nm=((args.thickness_nm, published_conditions),))
    conditions = iter(published_conditions)
    original = af.mc_spectrum
    rows = []
    arrays = {"energy_eV": study.E_grid}

    def measure(segments, energy, **kwargs):
        condition = next(conditions)
        variants = {"baseline": original(segments, energy, **kwargs)}
        if args.baseline_only:
            row = {
                "beam_energy_keV": condition.energy_keV,
                "variants": {
                    "baseline": response_stages(study, condition, variants["baseline"]),
                },
            }
            rows.append(row)
            arrays[f"E{condition.energy_keV:g}_baseline"] = variants["baseline"]
            print(f"{condition.energy_keV:g} keV: baseline complete", flush=True)
            return variants["baseline"]
        with patch.object(_kernels, "emission_coupling_tables", reflection_coupling_tables):
            variants["pre_338_pairing"] = original(segments, energy, **kwargs)
        # Removing opacity alone preserves the real refractive index and transport.
        with patch.object(
            _setup,
            "_elemental_log_mu_table",
            lambda comp, grid: np.full((len(comp), len(grid)), -700.0),
        ):
            variants["no_escape_loss"] = original(segments, energy, **kwargs)
        for name, overrides in (
            ("002_only", {"hkl_list": [h for h in kwargs["hkl_list"] if abs(h[2]) == 2]}),
            ("004_only", {"hkl_list": [h for h in kwargs["hkl_list"] if abs(h[2]) == 4]}),
            ("retired_B_0p6", {"B_ang2": 0.6}),
            ("zero_B", {"B_ang2": 0.0}),
            ("bin_mean", {"line_quadrature": "bin-mean"}),
        ):
            variants[name] = original(segments, energy, **(kwargs | overrides))
        row = {
            "beam_energy_keV": condition.energy_keV,
            "segments": int(len(segments["L_ang"])),
            "hkl_list": np.asarray(kwargs["hkl_list"]).tolist(),
            "variants": {},
        }
        for name, spectrum in variants.items():
            if not np.isfinite(spectrum).all():
                raise ValueError(f"nonfinite {condition.energy_keV} {name}")
            row["variants"][name] = response_stages(study, condition, spectrum)
            arrays[f"E{condition.energy_keV:g}_{name}"] = spectrum
        peak_index = int(
            np.argmax(af._supplementary_detected_spectrum(study, condition, variants["baseline"]))
        )
        width = float(
            np.hypot(
                row["variants"]["baseline"]["eds_fwhm_eV"],
                row["variants"]["baseline"]["aperture_fwhm_eV"],
            )
        )
        qe = detector_efficiency(energy)
        # Use the same response width for both families so their contributions add.
        row["basal_fraction_at_detected_peak"] = {
            name: float(
                convolve_detector(energy, variants[name] * qe, width)[peak_index]
                / convolve_detector(energy, variants["baseline"] * qe, width)[peak_index]
            )
            for name in ("002_only", "004_only")
        }
        row["pre_338_max_relative_difference"] = float(
            np.max(np.abs(variants["baseline"] - variants["pre_338_pairing"]))
            / variants["baseline"].max()
        )
        rows.append(row)
        print(f"{condition.energy_keV:g} keV: factor diagnostics complete", flush=True)
        return variants["baseline"]

    with patch.object(af, "mc_spectrum", measure):
        af.model_coherent_spectra(study, args.thickness_nm, ne=args.ne)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "ne": args.ne,
                "thickness_nm": args.thickness_nm,
                "seeds": [int(args.thickness_nm * 100) + i for i in range(4)],
                "backend": BACKEND.name,
                "dtype": np.dtype(REAL).name,
                "model": "post-338, segment-mean-v3-snell-resonance",
                "paper_peak_provenance": "historical approximate visual readings, not digitized data",
                "paper_peaks": [0.85, 0.80, 0.75, 0.70],
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
