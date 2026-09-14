"""
Packaged literature-validation figure builders.

Publication validation figures: the Monte-Carlo model's PXR+CBS spectra
overlaid against first-principles THEORY anchors, for the Zhai et al.,
Nat. Commun. 16, 11218 (2025), Fig. 1c geometry -- tunable X-rays from a
1 mm HOPG bulk crystal under 17.5 / 20 / 22.5 / 25 keV electrons observed at
theta_obs = 119 deg into 0.066 sr, plus the bulk vs 29 nm thin-film
enhancement at 25 keV.

This is the "model versus the literature" deliverable (TODO P1 #2). With no
digitized Fig 1c curve in the repo, the model is anchored against EXACT theory:

  1. Dispersion-relation line energies  E = hbar c beta g / (1 - beta cos theta)
     (Feranchuk-Spence Eq. 10, the zero-scattering resonance): the MC peak
     positions must land on these. Drawn as vertical markers per beam energy.
  2. Feranchuk-Spence closed-form absolute line flux (Eq. 12, photons/electron
     into dOmega): a SINGLE straight segment integrated through mc_spectrum
     reproduces it to <1% (the lineshape-normalization anchor); the full
     transport-broadened MC line then shows the transport correction.
  3. Bulk vs 29 nm film enhancement (MC), with the analytic effective-length
     ratio L_eff = L_abs (1 - exp(-L_z / L_abs)) as the no-transport ceiling.

If a digitized Fig 1c curve is provided in reference_data/zhai_fig1c.csv
(schema in reference_data/README.md), the spectra figure overlays it
automatically -- so the same figure/notebook becomes a true model-vs-measured
plot the moment real data lands, with no code change.

Backend module: the functions return plain data + matplotlib Figures; main()
runs the (slow) MC, writes figures/, and prints the validation tables. The Zhai
section of src/pyrite/apps/validation_app.py is the thin interactive wrapper.

Run (CPU-force on a box with the cupy wheel but no CUDA device):
  uv run python -c "import sys;sys.modules['cupy']=None;from pyrite.validation.anchor_figures import main;main()"
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tabulate import tabulate

_HERE = Path(__file__).resolve().parent
_PACKAGE_ROOT = _HERE.parent
from pyrite.detectors import Detector, LegacyEDS  # noqa: E402, I001
from pyrite.materials.crystal import crystal_absorption_length_ang  # noqa: E402
from pyrite.montecarlo import (  # noqa: E402
    aperture_fwhm_eV,
    beta_from_keV,
    eds_fwhm_eV,
    mc_brem_spectrum,
    mc_spectrum,
    simulate_trajectories,
)
from pyrite.montecarlo.geometry import tilted_geometry  # noqa: E402
from pyrite.console.config import workspace_root  # noqa: E402
from pyrite.validation._anchor_conditions import (  # noqa: E402, F401, I001
    GRAPHITE_B_002 as GRAPHITE_B_002,
    ZHAI_SUPPLEMENTARY_STUDIES as ZHAI_SUPPLEMENTARY_STUDIES,
    SupplementaryCoherentStudy as SupplementaryCoherentStudy,
    SupplementaryCondition as SupplementaryCondition,
    ZhaiAnchor as ZhaiAnchor,
    ZhaiThermalBeam as ZhaiThermalBeam,
    _cache_payload as _cache_payload,
    _cache_record as _cache_record,
    _decode_supplementary_spectra as _decode_supplementary_spectra,
    _encode_supplementary_spectra as _encode_supplementary_spectra,
    _resolved_case as _resolved_case,
    _same_conditions_by_thickness as _same_conditions_by_thickness,
    _sem_conditions as _sem_conditions,
    _thickness_stem as _thickness_stem,
    feranchuk_line_flux as feranchuk_line_flux,
    fig1c_case as fig1c_case,
    line_energy_eV as line_energy_eV,
    single_segment_anchor as single_segment_anchor,
    theory_line_energies as theory_line_energies,
)
from pyrite.validation.feranchuk_spence import (  # noqa: E402
    escape_limited_length_ang,
)
from pyrite.validation._zhai import (  # noqa: E402
    ZHAI_CACHE_FORMAT as ZHAI_CACHE_FORMAT,
    ZHAI_CACHE_SCHEMA,
    ZHAI_DETECTOR as ZHAI_DETECTOR,
    detector_metadata as detector_metadata,
)
from pyrite.validation.validation_background import (  # noqa: E402
    compare_external_background,
    subtract_external_background,
)

_EXTERNAL_BREM_V1 = _HERE / "reference_data" / "external_brem" / "v1"
_ZHAI_FIG3B_BREM = _EXTERNAL_BREM_V1 / "zhai_fig3b_25kev_1mm_brem.csv"
_ZHAI_FIG3B_EXPERIMENT = _EXTERNAL_BREM_V1 / "zhai_fig3b_25kev_1mm_experiment.csv"


class ZhaiCacheMiss(FileNotFoundError):
    """Raised when a cache-only validation surface cannot load a valid record."""

    def __init__(self, path: Path):
        self.path = path
        super().__init__(
            f"Zhai cache missing or stale: {path}; populate it with "
            "`pyrite run --preset zhai --remote`"
        )


# ---- the (slow) Monte-Carlo model spectra ------------------------------------


def model_spectra(anchor: ZhaiAnchor, ne: int = 500, ne_brem: int = 200) -> dict:
    """Run the MC transport + PXR/CBS spectrum for each beam energy (1 mm bulk),
    plus the 29 nm film at the top energy. SLOW (CPU minutes for ne~500).

    Returns a dict keyed by beam energy [keV], each with the response-free source line
    spectrum, detector-convolved line + brem, peak energy, FWHM, and the
    per-electron integrated line flux into dOmega; plus a "film" entry.
    """
    E_grid = anchor.E_grid
    out: dict = {}
    for E0 in anchor.energies_keV:
        seed = int(E0 * 10)
        case = fig1c_case(
            anchor,
            E0,
            anchor.thick_bulk_ang,
            ne=ne,
            ne_brem=ne_brem,
            seed=seed,
        )
        beam_dir, n_hat = tilted_geometry(
            case["theta_obs_rad"],
            float(np.deg2rad(case["tilt_deg"])),
            float(np.deg2rad(case["tilt_azim_deg"])),
        )
        segs = simulate_trajectories(
            case["E0_keV"],
            case["Ne"],
            case["thickness_ang"],
            composition=case["composition"],
            E_cut_keV=5.0,
            seed=case["seed"],
            beam_dir=beam_dir,
        )
        spec = mc_spectrum(
            segs,
            E_grid,
            crystal=case["crystal"],
            hkl_list=case["hkl_list"],
            n_hat=n_hat,
            B_ang2=case["B_ang2"],
            composition=case["composition"],
            beam_uvw=case["beam_uvw"],
            surface_hkl=case["surface_hkl"],
        )
        beta = beta_from_keV(case["E0_keV"])
        E_pk = float(E_grid[np.argmax(spec)])
        fwhm = float(
            np.hypot(
                eds_fwhm_eV(E_pk),
                aperture_fwhm_eV(
                    E_pk,
                    beta,
                    case["theta_obs_rad"],
                    case["dtheta_obs_rad"],
                ),
            )
        )
        spec_det = Detector(response=LegacyEDS(convolve=True)).score(E_grid, spec, fwhm_eV=fwhm)
        segs_b = simulate_trajectories(
            case["E0_keV"],
            case["Ne_brem"],
            case["thickness_ang"],
            composition=case["composition"],
            E_cut_keV=1.0,
            seed=case["seed"] + 1,
            beam_dir=beam_dir,
        )
        brem = mc_brem_spectrum(
            segs_b,
            E_grid,
            composition=case["composition"],
            n_hat=n_hat,
        )
        brem_det = Detector(response=LegacyEDS(convolve=True)).score(E_grid, brem, fwhm_eV=fwhm)
        out[E0] = {
            "spec": spec,
            "spec_det": spec_det,
            "brem": brem,
            "brem_det": brem_det,
            "E_peak": E_pk,
            "fwhm": fwhm,
            "line_flux_per_e": float(np.trapezoid(spec, E_grid) * case["domega_sr"]),
            "backscatter": float(segs["n_backscattered"] / segs["Ne"]),
        }
    # 29 nm film at the top energy, same detector FWHM
    E_top = anchor.energies_keV[-1]
    film_case = fig1c_case(
        anchor,
        E_top,
        anchor.thick_film_ang,
        ne=ne,
        ne_brem=ne_brem,
        seed=7,
    )
    film_beam_dir, film_n_hat = tilted_geometry(
        film_case["theta_obs_rad"],
        float(np.deg2rad(film_case["tilt_deg"])),
        float(np.deg2rad(film_case["tilt_azim_deg"])),
    )
    segs_f = simulate_trajectories(
        film_case["E0_keV"],
        film_case["Ne"],
        film_case["thickness_ang"],
        composition=film_case["composition"],
        E_cut_keV=5.0,
        seed=film_case["seed"],
        beam_dir=film_beam_dir,
    )
    spec_f = mc_spectrum(
        segs_f,
        E_grid,
        crystal=film_case["crystal"],
        hkl_list=film_case["hkl_list"],
        n_hat=film_n_hat,
        B_ang2=film_case["B_ang2"],
        composition=film_case["composition"],
        beam_uvw=film_case["beam_uvw"],
        surface_hkl=film_case["surface_hkl"],
    )
    spec_f_det = Detector(response=LegacyEDS(convolve=True)).score(
        E_grid, spec_f, fwhm_eV=out[E_top]["fwhm"]
    )
    out["film"] = {
        "E0_keV": E_top,
        "spec": spec_f,
        "spec_det": spec_f_det,
        "line_flux_per_e": float(np.trapezoid(spec_f, E_grid) * film_case["domega_sr"]),
        "n_transmitted": int(segs_f.get("n_transmitted", 0)),
    }
    return out


def _zhai_cache_key(anchor: ZhaiAnchor, ne: int, ne_brem: int) -> str:
    """Fingerprint inputs and implementation files that affect the reproduction."""
    digest = hashlib.sha256()
    inputs = {
        "schema": ZHAI_CACHE_SCHEMA,
        "anchor": asdict(anchor),
        "ne": ne,
        "ne_brem": ne_brem,
    }
    digest.update(json.dumps(inputs, sort_keys=True).encode())
    # Anchor helpers live outside src/. Hash this whole module so changes to
    # case construction or component routing cannot reuse a stale v4 payload.
    digest.update(Path(__file__).read_bytes())
    implementation_files = _PACKAGE_ROOT.glob("**/*.py")
    for path in sorted(implementation_files):
        digest.update(path.relative_to(_PACKAGE_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def cached_model_spectra(
    anchor: ZhaiAnchor,
    ne: int = 500,
    ne_brem: int = 200,
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    cache_only: bool = False,
) -> tuple[dict, bool, Path]:
    """Load or atomically cache a Zhai reproduction keyed by inputs and code.

    Returns ``(model, cache_hit, path)``. Cache files are local generated
    artifacts under ``checkpoints/zhai_reproduction`` by default.
    """
    from pyrite.checkpoints import _checkpoint_io

    root = (
        Path(cache_dir)
        if cache_dir is not None
        else workspace_root() / "checkpoints" / "zhai_reproduction"
    )
    path = root / f"zhai-v{ZHAI_CACHE_SCHEMA}-{_zhai_cache_key(anchor, ne, ne_brem)}.pkl"
    if path.exists() and not refresh:
        cached = _cache_payload(
            _checkpoint_io.load(str(path)),
            kind="fig1c",
            detector=anchor.detector,
        )
        if cached is not None:
            return cached, True, path
    if cache_only:
        raise ZhaiCacheMiss(path)

    model = model_spectra(anchor, ne=ne, ne_brem=ne_brem)
    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        _checkpoint_io.dump(_cache_record("fig1c", anchor.detector, model), str(tmp))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return model, False, path


# ---- Zhai supplementary coherent-emission studies ---------------------------


def supplementary_study(crystal: str) -> SupplementaryCoherentStudy:
    """Return the configured Zhai supplementary study for ``crystal``.

    Supported keys are ``wse2``, ``mose2``, ``hbn``, and ``hopg``. Keeping the reported
    figure inputs in one immutable registry makes the dashboard a thin driver
    and avoids silently mixing these studies with the Fig. 1c HOPG anchor.
    """
    try:
        return ZHAI_SUPPLEMENTARY_STUDIES[crystal]
    except KeyError as exc:
        raise ValueError(
            f"unknown Zhai supplementary crystal {crystal!r}; "
            f"have {list(ZHAI_SUPPLEMENTARY_STUDIES)}"
        ) from exc


def supplementary_orientations(
    study: SupplementaryCoherentStudy, thickness_nm: float
) -> tuple[tuple[float, float], ...]:
    """Return the distinct reported SEM orientation pairs in Table 4 order."""
    orientations = tuple(
        (condition.polar_tilt_deg, condition.azimuth_deg)
        for condition in study.conditions_for(thickness_nm)
        if condition.azimuth_deg is not None
    )
    return tuple(dict.fromkeys(orientations))


def model_coherent_spectra(
    study: SupplementaryCoherentStudy,
    thickness_nm: float,
    ne: int = 500,
    *,
    exploratory_azimuth_deg: float | None = None,
) -> dict[SupplementaryCondition, np.ndarray]:
    """Simulate response-free coherent PXR+CBS source spectra for one requested thickness.

    Electron transport uses the material's compound composition and the sample
    geometry at each reported condition (beam energy, polar tilt, azimuth).
    Conditions whose azimuth Zhai does not report (the TEM-based TMD studies)
    are only modeled when ``exploratory_azimuth_deg`` is supplied explicitly;
    otherwise a ``ValueError`` is raised so an unreported angle is never
    silently presented as zero.  The returned spectra are keyed by condition
    and are ``d²N / (dE dOmega electron)``: no bremsstrahlung, detector
    response, or experimental solid-angle scaling is added.  The figure
    builders preserve this cache format, then apply the Fig. 1c detector
    response for display.
    """
    if ne < 1:
        raise ValueError("ne must be positive")

    thickness_ang = float(thickness_nm) * 10.0
    spectra: dict[SupplementaryCondition, np.ndarray] = {}
    conditions = study.conditions_for(thickness_nm)
    for index, condition in enumerate(conditions):
        azimuth_deg = condition.azimuth_deg
        if azimuth_deg is None:
            azimuth_deg = exploratory_azimuth_deg
        if azimuth_deg is None:
            raise ValueError(
                f"{study.crystal}'s TEM azimuth is unreported by Zhai et al.; "
                "pass exploratory_azimuth_deg explicitly to model this study"
            )
        seed = int(thickness_nm * 100) + index
        case = _resolved_case(
            material=study.crystal,
            detector=study.detector,
            energy_keV=condition.energy_keV,
            thickness_ang=thickness_ang,
            E_grid=study.E_grid,
            polar_tilt_deg=condition.polar_tilt_deg,
            azimuth_deg=azimuth_deg,
            ne=ne,
            ne_brem=1,
            seed=seed,
        )
        beam_dir, n_hat = tilted_geometry(
            case["theta_obs_rad"],
            float(np.deg2rad(case["tilt_deg"])),
            float(np.deg2rad(case["tilt_azim_deg"])),
        )
        # Each panel receives a reproducible, distinct transport realization.
        segments = simulate_trajectories(
            case["E0_keV"],
            case["Ne"],
            case["thickness_ang"],
            composition=case["composition"],
            beam_dir=beam_dir,
            E_cut_keV=5.0,
            seed=case["seed"],
        )
        spectra[condition] = mc_spectrum(
            segments,
            study.E_grid,
            crystal=case["crystal"],
            hkl_list=case["hkl_list"],
            n_hat=n_hat,
            B_ang2=case["B_ang2"],
            composition=case["composition"],
            beam_uvw=case["beam_uvw"],
            surface_hkl=case["surface_hkl"],
        )
    return spectra


def _supplementary_cache_key(
    study: SupplementaryCoherentStudy,
    thickness_nm: float,
    ne: int,
    exploratory_azimuth_deg: float | None,
) -> str:
    """Fingerprint a supplementary coherent-only calculation and its inputs."""
    digest = hashlib.sha256()
    inputs = {
        "schema": ZHAI_CACHE_SCHEMA,
        "study": asdict(study),
        "thickness_nm": thickness_nm,
        "ne": ne,
        "exploratory_azimuth_deg": exploratory_azimuth_deg,
    }
    digest.update(json.dumps(inputs, sort_keys=True).encode())
    digest.update(Path(__file__).read_bytes())
    for path in sorted(_PACKAGE_ROOT.glob("**/*.py")):
        digest.update(path.relative_to(_PACKAGE_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def cached_coherent_spectra(
    study: SupplementaryCoherentStudy,
    thickness_nm: float,
    ne: int = 500,
    *,
    exploratory_azimuth_deg: float | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    cache_only: bool = False,
) -> tuple[dict[SupplementaryCondition, np.ndarray], bool, Path]:
    """Load or atomically cache one supplementary coherent-only condition set."""
    from pyrite.checkpoints import _checkpoint_io

    root = (
        Path(cache_dir)
        if cache_dir is not None
        else workspace_root() / "checkpoints" / "zhai_reproduction"
    )
    key = _supplementary_cache_key(study, thickness_nm, ne, exploratory_azimuth_deg)
    path = root / f"zhai-supplement-v{ZHAI_CACHE_SCHEMA}-{key}.pkl"
    if path.exists() and not refresh:
        cached = _cache_payload(
            _checkpoint_io.load(str(path)),
            kind=f"supplementary:{study.crystal}",
            detector=study.detector,
        )
        if cached is not None:
            return _decode_supplementary_spectra(cached), True, path
    if cache_only:
        raise ZhaiCacheMiss(path)

    spectra = model_coherent_spectra(
        study, thickness_nm, ne=ne, exploratory_azimuth_deg=exploratory_azimuth_deg
    )
    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        _checkpoint_io.dump(
            _cache_record(
                f"supplementary:{study.crystal}",
                study.detector,
                _encode_supplementary_spectra(spectra),
            ),
            str(tmp),
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return spectra, False, path


def reproduce_all(
    ne: int = 20_000,
    ne_brem: int = 200,
    ne_supp: int = 200,
    *,
    tmd_exploratory_azimuth_deg: float = 0.0,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> list[tuple[str, Path, bool]]:
    """Force-populate every Zhai cache the validation app can hit, at the
    app's own default sample counts so a pulled cache is a guaranteed hit
    locally.

    Zhai does not report the TMD (WSe2/MoSe2) TEM azimuth, so those caches are
    computed at ``tmd_exploratory_azimuth_deg`` — the default matches the app's
    exploratory-azimuth control default; the value is part of the cache key.

    No figures -- this only leaves correct, hash-addressed .pkl files on disk
    under ``cache_dir`` (default checkpoints/zhai_reproduction/). This is the
    GPU-box-runnable unit behind ``pyrite._entry.reproduce_zhai`` /
    ``pyrite run --preset zhai --remote``.

    Returns [(label, path, cache_hit)] for the Fig.1c anchor plus every
    supplementary (study, thickness) pair.
    """
    results: list[tuple[str, Path, bool]] = []
    anchor = ZhaiAnchor()
    _, hit, path = cached_model_spectra(
        anchor, ne=ne, ne_brem=ne_brem, cache_dir=cache_dir, refresh=refresh
    )
    results.append(("zhai-fig1c", path, hit))
    for crystal, study in ZHAI_SUPPLEMENTARY_STUDIES.items():
        azimuth_deg = tmd_exploratory_azimuth_deg if study.has_unreported_azimuth else None
        for thickness_nm in study.thicknesses_nm:
            _, hit, path = cached_coherent_spectra(
                study,
                thickness_nm,
                ne=ne_supp,
                exploratory_azimuth_deg=azimuth_deg,
                cache_dir=cache_dir,
                refresh=refresh,
            )
            results.append((f"{crystal}-{_thickness_stem(thickness_nm)}nm", path, hit))
    return results


# ---- optional digitized reference (model-vs-measured hook) -------------------


def reference_curve(anchor: ZhaiAnchor | None = None, path: str | Path | None = None):
    """Load a digitized Zhai Fig 1c curve if present, else return None.

    Schema (see reference_data/README.md): a CSV with columns
        series, energy_eV, intensity
    where `series` labels the beam energy (e.g. "25keV"). Returns
        {series_label: (energy_eV[np], intensity[np])}
    or None when no file exists, so callers degrade gracefully to theory-only.
    """
    import pandas as pd

    if path is None:
        path = _HERE / "reference_data" / "zhai_fig1c.csv"
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#")
    cols = {c.lower().strip(): c for c in df.columns}
    ecol = cols.get("energy_ev") or cols.get("energy")
    icol = cols.get("intensity") or cols.get("counts")
    scol = cols.get("series") or cols.get("label")
    if ecol is None or icol is None:
        raise ValueError(
            f"reference CSV {path} needs energy_eV and intensity columns; got {list(df.columns)}"
        )
    if scol is None:
        df = df.assign(_series="measured")
        scol = "_series"
    return {str(s): (g[ecol].to_numpy(float), g[icol].to_numpy(float)) for s, g in df.groupby(scol)}


def _match_series(reference: dict, E0_keV: float) -> str | None:
    """Find the reference series whose label parses to ~E0_keV (tolerant of
    '25', '25keV', '25.0 keV', etc.)."""
    for key in reference:
        digits = "".join(ch if (ch.isdigit() or ch == ".") else " " for ch in key)
        for tok in digits.split():
            try:
                if abs(float(tok) - E0_keV) < 0.25:
                    return key
            except ValueError:
                continue
    return None


def zhai_background_validation(anchor: ZhaiAnchor, model: dict | None = None) -> dict:
    """Fit/subtract deposited Zhai Fig. 3b background and compare model if supplied.

    Figure 3b reports 25 keV electrons on 1 mm HOPG.  The sideband definition
    excludes 900--1040 eV, which contains the broadened coherent peak.  Fixture
    provenance and normalization are versioned beside the data.
    """
    data = np.genfromtxt(
        _ZHAI_FIG3B_EXPERIMENT,
        delimiter=",",
        names=True,
        comments="#",
        skip_header=5,
    )
    energy = data["energy_eV"]
    intensity = data["intensity_Phs_per_eV_s_nA"]
    sigma = data["sigma_Phs_per_eV_s_nA"]
    sidebands = (energy < 900.0) | (energy > 1040.0)
    residual, fitted_background, fit = subtract_external_background(
        energy,
        intensity,
        _ZHAI_FIG3B_BREM,
        sigma=sigma,
        fit_mask=sidebands,
    )
    result = {
        "energy_eV": energy,
        "intensity": intensity,
        "sigma": sigma,
        "sidebands": sidebands,
        "fitted_background": fitted_background,
        "subtracted": residual,
        "fit": fit,
        "comparison": None,
    }
    if model is not None:
        scale = anchor.domega_sr * anchor.per_nA
        background = model[25.0]["brem_det"] * scale
        overlap = (anchor.E_grid >= energy.min()) & (anchor.E_grid <= energy.max())
        result["comparison"] = compare_external_background(
            anchor.E_grid,
            background,
            _ZHAI_FIG3B_BREM,
            comparison_mask=overlap,
        )
    return result


def figure_background_validation(validation: dict, anchor: ZhaiAnchor, model: dict | None = None):
    """Plot deposited experiment, fitted DTSA-II background, and subtraction."""
    import matplotlib.pyplot as plt

    energy = validation["energy_eV"]
    fig, (ax_total, ax_sub) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax_total.errorbar(
        energy,
        validation["intensity"],
        yerr=validation["sigma"],
        fmt="o",
        ms=3,
        alpha=0.65,
        label="Zhai deposited experiment",
    )
    ax_total.plot(
        energy,
        validation["fitted_background"],
        "--",
        label=f"deposited external brem × {validation['fit'].scale:.4f}",
    )
    if model is not None:
        scale = anchor.domega_sr * anchor.per_nA
        ax_total.plot(
            anchor.E_grid,
            model[25.0]["brem_det"] * scale,
            ":",
            label="PyRITE Born+Elwert brem",
        )
    ax_total.set_ylabel("Detected intensity (Phs/eV/s/nA)")
    ax_total.set_title("Zhai Fig. 3b: external-background fit")
    ax_total.legend()

    ax_sub.axhline(0.0, color="0.5", lw=0.8)
    ax_sub.plot(energy, validation["subtracted"], "o-", ms=3)
    ax_sub.axvspan(900.0, 1040.0, color="C1", alpha=0.12, label="excluded peak window")
    ax_sub.set_xlabel("Photon energy (eV)")
    ax_sub.set_ylabel("Brem-subtracted intensity")
    ax_sub.set_title("Experimental residual after weighted sideband fit")
    ax_sub.legend()
    fig.tight_layout()
    return fig


# ---- figures -----------------------------------------------------------------


def figure_spectra(anchor: ZhaiAnchor, model: dict, reference: dict | None = None):
    """Fig 1c analog vs theory in three vertically stacked stages.

    The stages are response-free source PXR+CBS, detector-convolved PXR+CBS without the
    incoherent bremsstrahlung background, and the detector-convolved total.
    Eq.(10) line energies are vertical markers; the 29 nm film is overlaid on
    both detector views, and digitized measurements (when present) on the total.
    """
    import matplotlib.pyplot as plt

    scale = anchor.domega_sr * anchor.per_nA  # per e/sr/eV -> Phs/eV/s/nA
    lines = theory_line_energies(anchor)
    fig, (ax_i, ax_line, ax_total) = plt.subplots(3, 1, figsize=(9, 13), sharex=True)
    for i, E0 in enumerate(anchor.energies_keV):
        m = model[E0]
        c = f"C{i}"
        ax_i.plot(anchor.E_grid, m["spec"] * scale, color=c, label=f"{E0:g} keV")
        ax_line.plot(
            anchor.E_grid,
            m["spec_det"] * scale,
            color=c,
            label=f"{E0:g} keV (bulk)",
        )
        ax_total.plot(
            anchor.E_grid,
            (m["spec_det"] + m["brem_det"]) * scale,
            color=c,
            label=f"{E0:g} keV (bulk)",
        )
        ax_total.plot(anchor.E_grid, m["brem_det"] * scale, color=c, ls="--", lw=0.9)
        for ax in (ax_i, ax_line, ax_total):
            ax.axvline(lines[E0], color=c, ls=":", lw=1.2, alpha=0.7)
    if "film" in model:
        for ax in (ax_line, ax_total):
            ax.plot(
                anchor.E_grid,
                model["film"]["spec_det"] * scale,
                "c-",
                lw=1.0,
                label=f"{model['film']['E0_keV']:g} keV, 29 nm film",
            )
    if reference:
        for i, E0 in enumerate(anchor.energies_keV):
            key = _match_series(reference, E0)
            if key is None:
                continue
            e, inten = reference[key]
            peak = float((model[E0]["spec_det"] + model[E0]["brem_det"]).max() * scale)
            y = inten / np.nanmax(inten) * peak  # scale measured shape to model peak
            ax_total.scatter(
                e,
                y,
                s=14,
                facecolors="none",
                edgecolors=f"C{i}",
                alpha=0.8,
                label=f"{E0:g} keV (measured)",
            )
    ax_i.set_title("Response-free source PXR+CBS (1 mm HOPG)\ndotted = Eq.(10) line energy")
    ax_line.set_title("EDS-convolved PXR+CBS only (no incoherent brem)")
    ax_total.set_title("EDS-convolved total: PXR+CBS + brem (dashed)")
    for ax in (ax_i, ax_line, ax_total):
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel("Intensity (Phs/eV/s/nA)")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Model vs theory: PXR+CBS from HOPG, "
        rf"$\theta_{{obs}}$={anchor.detector.observation_angle_deg:g}$\degree$, "
        f"{anchor.domega_sr:g} sr"
    )
    fig.tight_layout()
    return fig


def figure_flux_anchor(anchor: ZhaiAnchor, model: dict):
    """Absolute-flux anchor. Left: the single straight-segment MC/closed-form
    ratio per beam energy (must hover at 1 -- validates the lineshape
    normalization). Right: per-electron integrated line flux, MC full transport
    (bulk) vs the Feranchuk Eq.(12) bulk estimate, log-y."""
    import matplotlib.pyplot as plt

    E0s = np.array(anchor.energies_keV, float)
    ratios = np.array([single_segment_anchor(anchor, E0)[2] for E0 in anchor.energies_keV])
    mc_bulk = np.array([model[E0]["line_flux_per_e"] for E0 in anchor.energies_keV])
    fer_bulk = np.array(
        [feranchuk_line_flux(anchor, E0, anchor.thick_bulk_ang) for E0 in anchor.energies_keV]
    )

    fig, (ax_r, ax_f) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_r.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.6)
    ax_r.plot(E0s, ratios, "o-", color="C3")
    ax_r.set_ylim(0.9, 1.1)
    ax_r.set_xlabel("Beam energy (keV)")
    ax_r.set_ylabel("MC / closed-form (single segment)")
    ax_r.set_title("Lineshape-normalization anchor\n(Eq. 12, should be 1)")
    ax_r.grid(alpha=0.3)

    ax_f.semilogy(E0s, mc_bulk, "o-", color="C0", label="MC (full transport, bulk)")
    ax_f.semilogy(E0s, fer_bulk, "s--", color="C1", label="Feranchuk Eq.(12), bulk")
    ax_f.set_xlabel("Beam energy (keV)")
    ax_f.set_ylabel(f"Line flux (photons / electron into {anchor.domega_sr:g} sr)")
    ax_f.set_title("Absolute line flux: MC vs analytic\n(transport vs escape-limited)")
    ax_f.grid(alpha=0.3, which="both")
    ax_f.legend(fontsize=8)
    fig.tight_layout()
    return fig


def figure_enhancement(anchor: ZhaiAnchor, model: dict):
    """Bulk vs 29 nm film at the top beam energy (EDS-convolved), annotated with
    the MC enhancement factor and the analytic no-transport ceiling
    L_eff(bulk)/L_eff(film)."""
    import matplotlib.pyplot as plt

    E0 = anchor.energies_keV[-1]
    scale = anchor.domega_sr * anchor.per_nA
    bulk = model[E0]["spec_det"] * scale
    film = model["film"]["spec_det"] * scale
    mc_enh = float(bulk.max() / film.max())

    E_line = line_energy_eV(anchor, E0)
    L_abs = crystal_absorption_length_ang(anchor.crystal, E_line)
    n_z = np.cos(anchor.theta_obs_rad)  # geometry="lif", as feranchuk_line_flux
    leff_bulk = escape_limited_length_ang(anchor.thick_bulk_ang, L_abs, n_z)
    leff_film = escape_limited_length_ang(anchor.thick_film_ang, L_abs, n_z)
    ceiling = float(leff_bulk / leff_film)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(anchor.E_grid, bulk, color="C0", label=f"{E0:g} keV, 1 mm bulk")
    ax.plot(anchor.E_grid, film, color="C1", label=f"{E0:g} keV, 29 nm film")
    ax.axvline(E_line, color="C3", ls=":", lw=1.2, alpha=0.7, label="Eq.(10) line energy")
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Intensity (Phs/eV/s/nA)")
    ax.set_title(
        f"Bulk vs thin-film enhancement\nMC peak ratio = {mc_enh:.1f}x  "
        f"(no-transport ceiling {ceiling:.0f}x)"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _supplementary_detected_spectrum(
    study: SupplementaryCoherentStudy,
    condition: SupplementaryCondition,
    spectrum: np.ndarray,
) -> np.ndarray:
    """Return one supplementary spectrum in Fig. 1c detected flux units.

    Applies the soft-X-ray window efficiency ``detector_efficiency`` to the
    response-free source spectrum first, then convolves with the same quadrature
    EDS-plus-aperture FWHM as ``model_spectra``: Zhai SI Eqs. (14) and (16),
    at the source spectrum's pre-response peak and the condition's reported beam
    energy (photons are lost in the detector window before the sensor
    electronics blur the surviving counts in energy).  It assumes the
    Fig. 1c collection aperture (0.066 sr) and 1 nA electron rate.  In the
    zero-FWHM, unit-efficiency limit the returned spectrum is the response-free source
    density scaled by those two factors.

    Caveat: Zhai's SI (S3/S4) never states whether their experimental
    spectra are efficiency-corrected or their theory includes window
    transmission; applying our nominal Moxtek AP3.3-class QE model here is
    a modeling choice, and the residual normalization gap vs Zhai SI
    Fig. S5b is tracked in docs/validation/physics-validation-ledger.md (id
    `zhai-hbn-921-detected`).
    """
    peak_eV = float(study.E_grid[np.argmax(spectrum)])
    fwhm_eV = float(
        np.hypot(
            eds_fwhm_eV(peak_eV),
            aperture_fwhm_eV(
                peak_eV,
                beta_from_keV(condition.energy_keV),
                study.theta_obs_rad,
                study.dtheta_obs_rad,
            ),
        )
    )
    return Detector(response=LegacyEDS(apply_qe=True, convolve=True)).score(
        study.E_grid,
        spectrum,
        fwhm_eV=fwhm_eV,
        scale=study.domega_sr * ZhaiAnchor.per_nA,
    )


def figure_supplementary_tmd(
    study: SupplementaryCoherentStudy,
    thickness_nm: float,
    spectra: dict[SupplementaryCondition, np.ndarray],
):
    """Return the four requested TMD coherent-only polar-tilt panels."""
    import matplotlib.pyplot as plt

    if study.crystal not in {"wse2", "mose2"}:
        raise ValueError("figure_supplementary_tmd is only for the WSe2 and MoSe2 studies")
    conditions = study.conditions_for(thickness_nm)
    if set(spectra) != set(conditions):
        raise ValueError("spectra must contain exactly the study's four conditions")

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True)
    for ax, condition in zip(axes.flat, conditions, strict=True):
        ax.plot(
            study.E_grid,
            _supplementary_detected_spectrum(study, condition, spectra[condition]),
            color="C0",
        )
        ax.set_title(f"Polar tilt {condition.polar_tilt_deg:g}° (azimuth unreported)")
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel("Intensity (Phs/eV/s/nA)")
        ax.set_xlim(study.e_min_eV, study.e_max_eV)
        ax.set_ylim(bottom=0.0)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"{study.label}, {conditions[0].energy_keV:g} keV, {thickness_nm:g} nm: "
        "Zhai-detector-convolved PXR+CBS"
    )
    # Matplotlib 3.10 can assign NaN axes bounds when tight_layout() measures
    # this shared 2×2 layout at the physical (~1e-9) intensity scale.
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.09, top=0.88, wspace=0.28, hspace=0.38)
    return fig


def figure_supplementary_sem(
    study: SupplementaryCoherentStudy,
    thickness_nm: float,
    spectra: dict[SupplementaryCondition, np.ndarray],
    *,
    orientation: tuple[float, float] | None = None,
):
    """Return one reported h-BN or HOPG SEM beam-energy comparison.

    When a sample has more than one reported orientation, the first Table 4
    pair is selected by default. Each figure therefore compares beam energy at
    one fixed polar/azimuthal orientation.
    """
    import matplotlib.pyplot as plt

    if study.crystal not in {"hbn", "hopg"}:
        raise ValueError("figure_supplementary_sem is only for the h-BN and HOPG studies")
    conditions = study.conditions_for(thickness_nm)
    if set(spectra) != set(conditions):
        raise ValueError("spectra must contain exactly the requested conditions")

    fig, ax = plt.subplots(figsize=(8, 5))
    orientations = supplementary_orientations(study, thickness_nm)
    if not orientations:
        raise ValueError("SEM conditions must include reported azimuths")
    selected_orientation = orientations[0] if orientation is None else tuple(orientation)
    if selected_orientation not in orientations:
        raise ValueError(
            f"orientation {selected_orientation} is not reported for {thickness_nm:g} nm"
        )
    selected_conditions = tuple(
        condition
        for condition in conditions
        if (condition.polar_tilt_deg, condition.azimuth_deg) == selected_orientation
    )
    for i, condition in enumerate(selected_conditions):
        ax.plot(
            study.E_grid,
            _supplementary_detected_spectrum(study, condition, spectra[condition]),
            color=f"C{i}",
            label=f"{condition.energy_keV:g} keV",
        )
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Intensity (Phs/eV/s/nA)")
    ax.set_xlim(study.e_min_eV, study.e_max_eV)
    ax.set_ylim(bottom=0.0)
    ax.set_title(
        f"{study.label}, {thickness_nm:g} nm, polar {selected_orientation[0]:g}°, "
        f"azimuth {selected_orientation[1]:g}°: Zhai-detector-convolved PXR+CBS"
    )
    ax.grid(alpha=0.3)
    ax.legend(title="Beam energy")
    fig.tight_layout()
    return fig


def figure_supplementary_hbn(
    study: SupplementaryCoherentStudy,
    thickness_nm: float,
    spectra: dict[SupplementaryCondition, np.ndarray],
):
    """Backward-compatible h-BN entry point for the generic SEM figure."""
    if study.crystal != "hbn":
        raise ValueError("figure_supplementary_hbn is only for the h-BN study")
    return figure_supplementary_sem(study, thickness_nm, spectra)


def figure_supplementary_overview(spectra: dict[str, np.ndarray]):
    """One row of WSe2, MoSe2, h-BN, and HOPG panels -- each
    showing that material's steepest requested polar tilt at its thinnest
    listed thickness: a single representative slice per material, for the
    paper's validation appendix. The full tilt x thickness grid is the
    per-material figure_supplementary_tmd/hbn panels above."""
    import matplotlib.pyplot as plt

    expected = {"wse2", "mose2", "hbn", "hopg"}
    if set(spectra) != expected:
        raise ValueError(f"spectra must contain exactly {sorted(expected)}, got {sorted(spectra)}")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, crystal in zip(axes, ("wse2", "mose2", "hbn", "hopg"), strict=True):
        study = supplementary_study(crystal)
        thickness_nm = study.thicknesses_nm[0]
        condition = study.conditions_for(thickness_nm)[-1]
        ax.plot(study.E_grid, spectra[crystal], color="C0")
        ax.set_title(
            f"{study.label}\n{thickness_nm:g} nm, {condition.energy_keV:g} keV, "
            f"tilt {condition.polar_tilt_deg:g}\N{DEGREE SIGN}"
        )
        ax.set_xlabel("Photon energy (eV)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"Coherent emission $d^2N/(dE\,d\Omega\,e^-)$")
    fig.suptitle("Zhai supplementary studies overview")
    fig.tight_layout()
    return fig


def export_all_figures(
    outdir: str | Path = "figures",
    ne: int = 20_000,
    ne_brem: int = 200,
    ne_supp: int = 200,
    *,
    tmd_exploratory_azimuth_deg: float = 0.0,
    cache_dir: str | Path | None = None,
) -> list[Path]:
    """Render the complete publication figure set from valid remote-populated caches.

    A missing, stale, or pre-detector cache raises :class:`ZhaiCacheMiss`;
    publication-sized Monte Carlo never starts locally from this rendering
    surface. Returns PNG paths; a PDF is written alongside each.
    """
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass

    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save(name, fig):
        import matplotlib.pyplot as plt

        for ext in ("png", "pdf"):
            fig.savefig(outpath / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        written.append(outpath / f"{name}.png")
        plt.close(fig)

    anchor = ZhaiAnchor()
    model, _hit, _path = cached_model_spectra(
        anchor,
        ne=ne,
        ne_brem=ne_brem,
        cache_dir=cache_dir,
        cache_only=True,
    )
    reference = reference_curve(anchor)
    _save("zhai_fig1c_spectra_vs_theory", figure_spectra(anchor, model, reference))
    _save("zhai_flux_anchor", figure_flux_anchor(anchor, model))
    _save("zhai_bulk_vs_film_enhancement", figure_enhancement(anchor, model))

    overview_spectra: dict[str, np.ndarray] = {}
    for crystal, study in ZHAI_SUPPLEMENTARY_STUDIES.items():
        azimuth_deg = tmd_exploratory_azimuth_deg if study.has_unreported_azimuth else None
        for thickness_nm in study.thicknesses_nm:
            spectra, _hit, _path = cached_coherent_spectra(
                study,
                thickness_nm,
                ne=ne_supp,
                exploratory_azimuth_deg=azimuth_deg,
                cache_dir=cache_dir,
                cache_only=True,
            )
            fig = (
                figure_supplementary_sem(study, thickness_nm, spectra)
                if crystal in {"hbn", "hopg"}
                else figure_supplementary_tmd(study, thickness_nm, spectra)
            )
            _save(f"zhai_supplementary_{crystal}_{_thickness_stem(thickness_nm)}nm", fig)
            if thickness_nm == study.thicknesses_nm[0]:
                overview_spectra[crystal] = spectra[study.conditions_for(thickness_nm)[-1]]

    _save("zhai_supplementary_overview", figure_supplementary_overview(overview_spectra))
    return written


# ---- validation table + CLI --------------------------------------------------


def validation_table(anchor: ZhaiAnchor, model: dict) -> list[list]:
    """Rows: per beam energy, the MC peak position vs the Eq.(10) line energy,
    the single-segment MC/closed-form ratio, and the MC bulk line flux."""
    lines = theory_line_energies(anchor)
    rows = []
    for E0 in anchor.energies_keV:
        m = model[E0]
        _, _, ratio = single_segment_anchor(anchor, E0)
        rows.append(
            [
                f"{E0:g}",
                f"{m['E_peak']:.0f}",
                f"{lines[E0]:.0f}",
                f"{m['E_peak'] - lines[E0]:+.0f}",
                f"{ratio:.3f}",
                f"{m['line_flux_per_e']:.3e}",
                f"{m['backscatter']:.3f}",
            ]
        )
    return rows


def main(outdir: str = "figures", ne: int = 500, ne_brem: int = 200) -> None:
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass

    anchor = ZhaiAnchor()
    print(f"Running MC model spectra (ne={ne}, ne_brem={ne_brem}); slow on CPU...")
    model = model_spectra(anchor, ne=ne, ne_brem=ne_brem)
    reference = reference_curve(anchor)
    print("reference data:", "LOADED" if reference else "none (theory-only overlay)")

    print()
    print(
        tabulate(
            validation_table(anchor, model),
            headers=[
                "E0\n[keV]",
                "MC peak\n[eV]",
                "Eq.10\n[eV]",
                "diff\n[eV]",
                "MC/closed\n(1 seg)",
                f"line flux\n[ph/e/{anchor.domega_sr:g}sr]",
                "backscatter",
            ],
            tablefmt="github",
        )
    )
    E0 = anchor.energies_keV[-1]
    enh = model[E0]["spec_det"].max() / model["film"]["spec_det"].max()
    print(
        f"\n25 keV bulk vs 29 nm film enhancement: {enh:.1f}x  "
        f"(film transmitted {model['film']['n_transmitted']} electrons)"
    )

    outpath = workspace_root() / outdir
    outpath.mkdir(exist_ok=True)
    figs = {
        "zhai_fig1c_spectra_vs_theory": figure_spectra(anchor, model, reference),
        "zhai_flux_anchor": figure_flux_anchor(anchor, model),
        "zhai_bulk_vs_film_enhancement": figure_enhancement(anchor, model),
    }
    for name, fig in figs.items():
        for ext in ("png", "pdf"):
            fig.savefig(outpath / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        print("wrote", (outpath / f"{name}.png").relative_to(workspace_root()))


if __name__ == "__main__":
    main()
