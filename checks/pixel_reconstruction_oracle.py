"""Pixel-detector angular-reconstruction oracle (issue #23, task
`agentdocs/tasks/feature/timepix-pixel-spectra/`, Slice 1).

`SpatialResult` reconstructs each fine pixel's intrinsic spectrum by
inheriting its coarse angular tile's spectrum unchanged (nearest-tile), then
applying that pixel's own exact solid angle and filter transmission. This
script measures how much error that substitution introduces, by comparing
the coarse-tile spectrum against a *direct* per-pixel evaluation at the
pixel's own exact direction -- both drawn from the SAME transported electron
population in one `run_case_directions` call, so the comparison isolates the
angular-reconstruction step from Monte Carlo noise (task-doc recommendation
#10).

This is decision evidence, not a signed-off validation anchor. The
provisional tolerances recorded in the task doc (40 eV or 0.1 FWHM line
centroid, 1-2% integrated line flux, 1% continuum) are explicitly "design-
review proposals, not validated acceptance criteria" -- this script reports
per-pixel, per-angular_shape error and asserts nothing about its magnitude.
A physics reviewer sets the acceptance bar from this evidence before
`PixelScorer` accuracy is claimed anywhere.

v1 scope frozen for this evidence run (see task-doc "Decisions and open
questions", confirmed 2026-08-21):
  - reconstruction policy under test: nearest-tile (current landed
    behaviour); interpolation/adaptive refinement is only designed if this
    evidence shows nearest-tile insufficient.
  - Timepix "hit" semantics: clustered photon events attributed to the
    incident-ray pixel (current `Timepix3.score` behaviour) -- raw
    neighbor-pixel triggers are out of scope.

Filter transmission is computed once per pixel from exact `PixelRays`/
`filter_path_lengths` geometry and applied identically to both the direct
and reconstructed intrinsic spectra -- it cannot itself diverge between the
two branches, so there is no independent "filter error" source. What can
differ is the *filtered* flux/centroid comparison, since an energy-dependent
transmission reweights whatever discrepancy the angular substitution already
introduced; this script reports that filtered comparison alongside the
unfiltered one so a reviewer can see whether filtering amplifies or damps
the reconstruction error for partially covered pixels.

Run: `uv run python checks/pixel_reconstruction_oracle.py --help`
Named geometry presets and explicit statistics/output arguments make remote
evidence runs reproducible without editing this file on the lab box.
The three fixed seeds below exercise both transport and the Timepix response;
each report row records its seed. The statistics (`N_ELECTRONS`, Timepix
`n_mc`) are smoke-test sized for local execution. A multi-seed,
higher-statistics pass for reviewable evidence belongs on `pyrite remote`, not
local execution.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.signal import find_peaks, peak_widths

import pyrite as pr
from pyrite import api
from pyrite.detectors import Timepix3
from pyrite.instrument import FilterPlate, PixelGrid, PlanarDetector, PlanarPose
from pyrite.instrument.attenuation import attenuation_matrix, primary_transmission
from pyrite.instrument.geometry import angular_tiles, filter_path_lengths, planar_detector_rays
from pyrite.montecarlo.geometry import directions_to_sample_frame
from pyrite.montecarlo.runner import run_case_directions

GRID_SHAPE = (9, 9)
PITCH_MM = (1.5, 1.5)
DISTANCE_MM = 80.0
ANGULAR_SHAPES = [(1, 1), (3, 3), (5, 5), (9, 9)]
N_ELECTRONS = 200
N_ELECTRONS_BREM = 100
TIMEPIX_N_MC = 20_000
SEEDS = (1, 2, 3)
PEAK_REL_PROMINENCE = 0.03
FILTER_DISTANCE_MM = 40.0
FILTER_SIZE_MM = (6.0, 13.5)
FILTER_OFFSET_MM = (3.0, 0.0)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / (
    "agentdocs/tasks/feature/timepix-pixel-spectra/evidence/oracle-report.json"
)


class OracleGeometry(NamedTuple):
    """Reviewed detector/target angles for one Slice 1 evidence case."""

    polar_deg: float
    tilt_deg: float
    tilt_azim_deg: float


GEOMETRIES = {
    "baseline": OracleGeometry(60.0, 30.0, 0.0),
    "broken-symmetry": OracleGeometry(60.0, 30.0, 45.0),
    "detector-on-g": OracleGeometry(20.0, 20.0, 0.0),
    "near-pole-not-g-aligned": OracleGeometry(20.0, 30.0, 0.0),
}


def _representative_pixels(shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    """Select centre/corner/edge/polar-only/azimuth-only/interior pixels.

    `PixelGrid` rows index the local y axis (azimuth, per
    `PlanarPose.from_observation`); columns index local x (polar). A
    "polar-only" pixel therefore varies only in column; "azimuth-only" only
    in row.
    """
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    return {
        "center": (cy, cx),
        "corner_tl": (0, 0),
        "corner_tr": (0, nx - 1),
        "corner_bl": (ny - 1, 0),
        "corner_br": (ny - 1, nx - 1),
        "edge_top": (0, cx),
        "edge_bottom": (ny - 1, cx),
        "edge_left": (cy, 0),
        "edge_right": (cy, nx - 1),
        "polar_only": (cy, nx - 2),
        "azimuth_only": (ny - 2, cx),
        "interior": (max(cy - 2, 0), min(cx + 2, nx - 1)),
    }


def _centroid_eV(energy_eV: np.ndarray, density: np.ndarray) -> float:
    total = np.trapezoid(density, energy_eV)
    if total <= 0.0:
        return float("nan")
    return float(np.trapezoid(energy_eV * density, energy_eV) / total)


def _integrated_flux(energy_eV: np.ndarray, density: np.ndarray) -> float:
    return float(np.trapezoid(density, energy_eV))


def _relative_error(direct: float, reconstructed: float) -> float:
    if direct == 0.0:
        return float("nan")
    return float((reconstructed - direct) / direct)


def _spectrum_shape_metrics(
    energy_eV: np.ndarray,
    density: np.ndarray,
    *,
    rel_prominence: float = PEAK_REL_PROMINENCE,
) -> dict[str, float | int | list[float]]:
    """Dominant-peak and multi-peak diagnostics on one spectral density.

    FWHM is evaluated at half prominence using SciPy's fractional sample
    locations, then mapped onto the energy axis.  The significant-peak list is
    deliberately retained: a centroid alone cannot distinguish one shifted
    line from two competing resonances.
    """
    energy = np.asarray(energy_eV, dtype=float)
    spectrum = np.nan_to_num(np.asarray(density, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if energy.ndim != 1 or spectrum.shape != energy.shape or energy.size < 2:
        raise ValueError("energy and density must be matching one-dimensional arrays")
    maximum = float(np.max(spectrum))
    if maximum <= 0.0:
        return {
            "dominant_peak_eV": float("nan"),
            "dominant_fwhm_eV": float("nan"),
            "significant_peak_count": 0,
            "significant_peak_energies_eV": [],
        }

    peaks, _ = find_peaks(
        spectrum,
        prominence=rel_prominence * maximum,
    )
    significant_peaks = peaks.copy()
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(spectrum))])
    dominant_slot = int(np.argmax(spectrum[peaks]))
    dominant = int(peaks[dominant_slot])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        widths = peak_widths(spectrum, [dominant], rel_height=0.5)
    left_eV = float(np.interp(widths[2][0], np.arange(energy.size), energy))
    right_eV = float(np.interp(widths[3][0], np.arange(energy.size), energy))
    return {
        "dominant_peak_eV": float(energy[dominant]),
        "dominant_fwhm_eV": right_eV - left_eV,
        "significant_peak_count": int(significant_peaks.size),
        "significant_peak_energies_eV": [float(energy[index]) for index in significant_peaks],
    }


def _lineshape_total_variation(
    energy_eV: np.ndarray,
    direct_density: np.ndarray,
    reconstructed_density: np.ndarray,
) -> float:
    """Total-variation distance between unit-area spectral shapes.

    The dimensionless result is zero for identical shapes and one for disjoint
    nonnegative spectra.  It is insensitive to total-flux error, which is
    reported separately.
    """
    energy = np.asarray(energy_eV, dtype=float)
    direct = np.clip(np.nan_to_num(direct_density), 0.0, None)
    reconstructed = np.clip(np.nan_to_num(reconstructed_density), 0.0, None)
    direct_total = _integrated_flux(energy, direct)
    reconstructed_total = _integrated_flux(energy, reconstructed)
    if direct_total <= 0.0 or reconstructed_total <= 0.0:
        return float("nan")
    return float(
        0.5
        * np.trapezoid(
            np.abs(direct / direct_total - reconstructed / reconstructed_total),
            energy,
        )
    )


def _score_filtered_pair(response, energy_eV, direct, reconstructed, transmission):
    """Apply exact pixel transmission before the common detector response."""
    filtered_direct = direct * transmission
    filtered_reconstructed = reconstructed * transmission
    return (
        np.asarray(response.score(energy_eV, filtered_direct, fwhm_eV=None, scale=1.0)),
        np.asarray(
            response.score(
                energy_eV,
                filtered_reconstructed,
                fwhm_eV=None,
                scale=1.0,
            )
        ),
    )


def _build_scene(
    geometry: OracleGeometry,
    *,
    timepix_n_mc: int,
) -> tuple[pr.Beam, pr.Slab, PlanarDetector, tuple[FilterPlate, ...]]:
    beam = pr.Beam(energy_keV=30.0)
    target = pr.Slab(
        "hopg",
        thickness_ang=10_000.0,
        tilt_deg=geometry.tilt_deg,
        tilt_azim_deg=geometry.tilt_azim_deg,
    )
    pose = PlanarPose.from_observation(
        distance_mm=DISTANCE_MM,
        polar_deg=geometry.polar_deg,
    )
    detector = PlanarDetector(
        pose=pose,
        pixels=PixelGrid(GRID_SHAPE, PITCH_MM),
        response=Timepix3(n_mc=timepix_n_mc),
    )
    filter_pose = PlanarPose.from_observation(
        distance_mm=FILTER_DISTANCE_MM,
        polar_deg=geometry.polar_deg,
        offset_mm=FILTER_OFFSET_MM,
    )
    half_filter = FilterPlate(
        "silicon",
        thickness_mm=0.1,
        size_mm=FILTER_SIZE_MM,
        pose=filter_pose,
    )
    return beam, target, detector, (half_filter,)


def run(
    *,
    geometry_name: str = "baseline",
    n_electrons: int = N_ELECTRONS,
    n_electrons_brem: int = N_ELECTRONS_BREM,
    timepix_n_mc: int = TIMEPIX_N_MC,
    seeds: tuple[int, ...] = SEEDS,
) -> list[dict]:
    geometry = GEOMETRIES[geometry_name]
    beam, target, detector, filters = _build_scene(
        geometry,
        timepix_n_mc=timepix_n_mc,
    )
    numerics = pr.Numerics(
        n_electrons=n_electrons,
        n_electrons_brem=n_electrons_brem,
    )
    scene = pr.Scene(beam=beam, target=target, detector=detector, filters=filters)
    base_case = api.build_case(scene, numerics)

    rays = planar_detector_rays(detector)
    filter_paths_mm = filter_path_lengths(rays, filters)
    pixels = _representative_pixels(GRID_SHAPE)
    pixel_names = list(pixels)
    pixel_coords = [pixels[name] for name in pixel_names]
    fine_directions_lab = np.stack([rays.directions_lab[row, col] for row, col in pixel_coords])

    report: list[dict] = []
    for seed in seeds:
        case = replace(base_case, seed=seed)
        response = detector.response
        assert response is not None
        response = replace(response, seed=seed)
        for angular_shape in ANGULAR_SHAPES:
            tile_index, tile_directions_lab = angular_tiles(rays, angular_shape)
            n_tile = tile_directions_lab.shape[0]
            tile_pixel_counts = np.bincount(tile_index.ravel(), minlength=n_tile)

            all_directions_lab = np.concatenate([tile_directions_lab, fine_directions_lab], axis=0)
            all_directions_sample = directions_to_sample_frame(
                all_directions_lab,
                np.deg2rad(float(case.get("tilt_deg", 0.0))),
                np.deg2rad(float(case.get("tilt_azim_deg", 0.0))),
            )

            output = run_case_directions(case, all_directions_sample)
            line_energy = np.asarray(output["E_grid"])
            brem_energy = np.asarray(output["E_grid_brem"])
            spec_by_direction = np.asarray(output["spec_by_direction"])
            brem_by_direction = np.asarray(output["brem_wide_by_direction"])

            tile_spec = spec_by_direction[:n_tile]
            tile_brem = brem_by_direction[:n_tile]
            pixel_spec = spec_by_direction[n_tile:]
            pixel_brem = brem_by_direction[n_tile:]

            line_mu = attenuation_matrix(filters, line_energy)

            for pixel_slot, (name, (row, col)) in enumerate(
                zip(pixel_names, pixel_coords, strict=True)
            ):
                tile = int(tile_index[row, col])
                direct_line = pixel_spec[pixel_slot]
                recon_line = tile_spec[tile]
                direct_brem = pixel_brem[pixel_slot]
                recon_brem = tile_brem[tile]

                direct_centroid = _centroid_eV(line_energy, direct_line)
                recon_centroid = _centroid_eV(line_energy, recon_line)
                direct_shape = _spectrum_shape_metrics(line_energy, direct_line)
                recon_shape = _spectrum_shape_metrics(line_energy, recon_line)

                # Transmission depends only on this pixel's exact ray geometry,
                # not on the angular reconstruction. Apply it before Timepix so
                # the detector sees the production-representative spectrum.
                transmission = primary_transmission(
                    filter_paths_mm[row : row + 1, col : col + 1], line_mu
                )[0, 0]
                filtered_direct = direct_line * transmission
                filtered_recon = recon_line * transmission
                direct_measured, recon_measured = _score_filtered_pair(
                    response,
                    line_energy,
                    direct_line,
                    recon_line,
                    transmission,
                )
                direct_measured_centroid = _centroid_eV(line_energy, direct_measured)
                recon_measured_centroid = _centroid_eV(line_energy, recon_measured)
                pixel_direction = rays.directions_lab[row, col]
                angular_offset_deg = float(
                    np.rad2deg(
                        np.arccos(
                            np.clip(
                                np.dot(pixel_direction, tile_directions_lab[tile]),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )

                report.append(
                    {
                        "geometry": geometry_name,
                        "polar_deg": geometry.polar_deg,
                        "tilt_deg": geometry.tilt_deg,
                        "tilt_azim_deg": geometry.tilt_azim_deg,
                        "n_electrons": n_electrons,
                        "n_electrons_brem": n_electrons_brem,
                        "timepix_n_mc": timepix_n_mc,
                        "seed": seed,
                        "angular_shape": list(angular_shape),
                        "pixel": name,
                        "row": row,
                        "column": col,
                        "tile": tile,
                        "tile_pixel_count": int(tile_pixel_counts[tile]),
                        "tile_is_singleton": bool(tile_pixel_counts[tile] == 1),
                        "tile_angular_offset_deg": angular_offset_deg,
                        "filter_transmission_mean": float(np.mean(transmission)),
                        "line_centroid_error_eV": recon_centroid - direct_centroid,
                        "direct_dominant_peak_eV": direct_shape["dominant_peak_eV"],
                        "reconstructed_dominant_peak_eV": recon_shape["dominant_peak_eV"],
                        "dominant_peak_error_eV": (
                            recon_shape["dominant_peak_eV"] - direct_shape["dominant_peak_eV"]
                        ),
                        "direct_dominant_fwhm_eV": direct_shape["dominant_fwhm_eV"],
                        "reconstructed_dominant_fwhm_eV": recon_shape["dominant_fwhm_eV"],
                        "direct_significant_peak_count": direct_shape["significant_peak_count"],
                        "reconstructed_significant_peak_count": recon_shape[
                            "significant_peak_count"
                        ],
                        "direct_significant_peak_energies_eV": direct_shape[
                            "significant_peak_energies_eV"
                        ],
                        "reconstructed_significant_peak_energies_eV": recon_shape[
                            "significant_peak_energies_eV"
                        ],
                        "line_shape_total_variation": _lineshape_total_variation(
                            line_energy, direct_line, recon_line
                        ),
                        "line_flux_relative_error": _relative_error(
                            _integrated_flux(line_energy, direct_line),
                            _integrated_flux(line_energy, recon_line),
                        ),
                        "filtered_line_flux_relative_error": _relative_error(
                            _integrated_flux(line_energy, filtered_direct),
                            _integrated_flux(line_energy, filtered_recon),
                        ),
                        "continuum_relative_error": _relative_error(
                            _integrated_flux(brem_energy, direct_brem),
                            _integrated_flux(brem_energy, recon_brem),
                        ),
                        "timepix_binned_centroid_error_eV": (
                            recon_measured_centroid - direct_measured_centroid
                        ),
                        "timepix_binned_flux_relative_error": _relative_error(
                            _integrated_flux(line_energy, direct_measured),
                            _integrated_flux(line_energy, recon_measured),
                        ),
                        "timepix_binned_shape_total_variation": (
                            _lineshape_total_variation(line_energy, direct_measured, recon_measured)
                        ),
                    }
                )

    return report


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=GEOMETRIES, default="baseline")
    parser.add_argument("--n-electrons", type=_positive_int, default=N_ELECTRONS)
    parser.add_argument("--n-electrons-brem", type=_positive_int, default=N_ELECTRONS_BREM)
    parser.add_argument("--timepix-n-mc", type=_positive_int, default=TIMEPIX_N_MC)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = run(
        geometry_name=args.geometry,
        n_electrons=args.n_electrons,
        n_electrons_brem=args.n_electrons_brem,
        timepix_n_mc=args.timepix_n_mc,
        seeds=tuple(args.seeds),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(report)} rows to {args.output}")

    def _finite_abs(value: float) -> float:
        return abs(value) if value == value else 0.0  # NaN != NaN

    worst = max(report, key=lambda row: _finite_abs(row["line_flux_relative_error"]))
    print(
        f"largest line-flux relative error: {worst['line_flux_relative_error']:.4f} "
        f"at angular_shape={worst['angular_shape']} pixel={worst['pixel']}"
    )


if __name__ == "__main__":
    main()
