"""
anchor_figures.py  (checks/)

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
section of notebooks/validation_app.py is the thin interactive wrapper.

Run (CPU-force on a box with the cupy wheel but no CUDA device):
  uv run python -c "import sys;sys.modules['cupy']=None;sys.path.insert(0,'checks');import runpy;runpy.run_path('checks/anchor_figures.py',run_name='__main__')"
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tabulate import tabulate

# checks/ siblings (feranchuk_spence) and ../src (cxr_mc) on the path,
# regardless of CWD.
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from feranchuk_spence import photons_per_electron  # noqa: E402

from cxr_mc.crystallography import (  # noqa: E402
    CRYSTALS,
    HBARC_EV_ANG,
    absorption_length_ang,
    reciprocal_g_vector,
)
from cxr_mc.montecarlo import (  # noqa: E402
    aperture_fwhm_eV,
    beta_from_keV,
    convolve_detector,
    detector_efficiency,
    eds_fwhm_eV,
    mc_brem_spectrum,
    mc_spectrum,
    simulate_trajectories,
)
from cxr_mc.montecarlo.geometry import tilted_geometry  # noqa: E402
from cxr_mc.sweep import crystal_params  # noqa: E402

GRAPHITE_B_002 = 0.8  # graphite c-axis Debye-Waller B-factor [Ang^2], approx (Zhai SI)
_ZHAI_CACHE_SCHEMA = 2  # v2: per-condition azimuths; v1 caches assumed azimuth = 0


@dataclass(frozen=True)
class SupplementaryCondition:
    """One spectrum condition reported in Zhai et al. Supplementary Fig. 5."""

    energy_keV: float
    polar_tilt_deg: float
    azimuth_deg: float | None


@dataclass(frozen=True)
class SupplementaryCoherentStudy:
    """A Zhai supplementary-information coherent-emission comparison.

    The study holds the reported thicknesses, photon-energy window, and four
    per-spectrum conditions (beam energy, polar tilt, and azimuth — ``None``
    when Zhai does not report it).  Its cached spectra contain intrinsic
    coherent emission only; the figure builders apply the Fig. 1c detector
    response when converting them to the displayed observable.
    """

    crystal: str
    label: str
    thicknesses_nm: tuple[float, ...]
    e_min_eV: float
    e_max_eV: float
    conditions: tuple[SupplementaryCondition, ...]
    theta_obs_rad: float = float(np.deg2rad(119.0))

    @property
    def E_grid(self) -> np.ndarray:
        """One-eV photon-energy grid, excluding the upper endpoint."""
        return np.arange(self.e_min_eV, self.e_max_eV, 1.0)


ZHAI_SUPPLEMENTARY_STUDIES = {
    "wse2": SupplementaryCoherentStudy(
        crystal="wse2",
        label=r"$\mathrm{WSe_2}$",
        thicknesses_nm=(42.0, 55.0, 75.0),
        e_min_eV=800.0,
        e_max_eV=1200.0,
        conditions=tuple(
            SupplementaryCondition(200.0, polar_tilt_deg, None)
            for polar_tilt_deg in (10.0, 15.0, 17.5, 20.0)
        ),
    ),
    "mose2": SupplementaryCoherentStudy(
        crystal="mose2",
        label=r"$\mathrm{MoSe_2}$",
        thicknesses_nm=(47.0, 112.0, 147.0),
        e_min_eV=800.0,
        e_max_eV=1200.0,
        conditions=tuple(
            SupplementaryCondition(200.0, polar_tilt_deg, None)
            for polar_tilt_deg in (10.0, 15.0, 17.5, 20.0)
        ),
    ),
    "hbn": SupplementaryCoherentStudy(
        crystal="hbn",
        label="h-BN",
        thicknesses_nm=(921.0,),
        e_min_eV=600.0,
        e_max_eV=1200.0,
        conditions=tuple(
            SupplementaryCondition(energy_keV, 17.0, 130.0)
            for energy_keV in (17.5, 20.0, 22.5, 25.0)
        ),
    ),
}


@dataclass(frozen=True)
class ZhaiAnchor:
    """Experimental conditions of Zhai et al. Fig. 1c (SEM/EDS take-off geometry).

    All defaults reproduce the published setup; override fields to explore other
    crystals, energies or detector geometries with the same machinery.
    """

    crystal: str = "hopg"
    hkl: tuple[int, int, int] = (0, 0, 2)
    hkl_list: tuple[tuple[int, int, int], ...] = ((0, 0, 2), (0, 0, -2))
    energies_keV: tuple[float, ...] = (17.5, 20.0, 22.5, 25.0)
    theta_obs_rad: float = float(np.deg2rad(119.0))
    dtheta_obs_rad: float = float(np.deg2rad(16.6))
    domega_sr: float = 0.066
    per_nA: float = 6.2415e9  # electrons/s at 1 nA
    thick_bulk_ang: float = 1e7  # 1 mm
    thick_film_ang: float = 290.0  # 29 nm
    B_ang2: float = GRAPHITE_B_002
    e_min_eV: float = 500.0
    e_max_eV: float = 1250.0
    de_eV: float = 1.0

    @property
    def E_grid(self) -> np.ndarray:
        return np.arange(self.e_min_eV, self.e_max_eV, self.de_eV)

    @property
    def n_atoms_per_ang3(self) -> float:
        info = CRYSTALS[self.crystal]
        return len(info["basis"]) / info["V_cell"]


# ---- theory anchors (cheap, analytic) ----------------------------------------


def line_energy_eV(anchor: ZhaiAnchor, E0_keV: float) -> float:
    """Zero-scattering resonance energy, Feranchuk-Spence Eq. (10):

        E = hbar c beta g_z / (1 - beta cos theta_obs),

    for beam || g (HOPG c-axis), so g_z = |g|. The MC peak must land here.
    """
    info = CRYSTALS[anchor.crystal]
    beta = beta_from_keV(E0_keV)
    _, g = reciprocal_g_vector(anchor.hkl, info["lattice"])
    return HBARC_EV_ANG * beta * g / (1.0 - beta * np.cos(anchor.theta_obs_rad))


def theory_line_energies(anchor: ZhaiAnchor) -> dict[float, float]:
    """{beam energy [keV]: dispersion-relation line energy [eV]}."""
    return {E0: line_energy_eV(anchor, E0) for E0 in anchor.energies_keV}


def feranchuk_line_flux(anchor: ZhaiAnchor, E0_keV: float, thickness_ang: float) -> float:
    """Feranchuk-Spence Eq. (12) closed-form line flux [photons / electron into
    dOmega], absorption-limited, at the dispersion-relation line energy."""
    beta = beta_from_keV(E0_keV)
    E_line = line_energy_eV(anchor, E0_keV)
    L_abs = absorption_length_ang("C", E_line, anchor.n_atoms_per_ang3)
    return photons_per_electron(
        anchor.crystal,
        anchor.hkl,
        E_line,
        anchor.theta_obs_rad,  # geometry="lif": this slot carries theta_obs
        beta,
        L_z_ang=thickness_ang,
        L_abs_ang=L_abs,
        dOmega_sr=anchor.domega_sr,
        polarization="both",
        B_ang2=anchor.B_ang2,
        use_henke=True,
        geometry="lif",
    )


def single_segment_anchor(
    anchor: ZhaiAnchor, E0_keV: float, L_seg_ang: float = 290.0
) -> tuple[float, float, float]:
    """The cleanest analytic<->MC anchor. A single straight segment of length
    L_seg pushed through mc_spectrum, integrated over energy, equals the
    Eq. (12) closed form -- this isolates the finite-segment lineshape
    normalization from electron transport. Returns (mc_integral, closed_form,
    ratio); ratio should be ~1.
    """
    beta = beta_from_keV(E0_keV)
    E_line = line_energy_eV(anchor, E0_keV)
    fake = {
        "r_mid": np.array([[0.0, 0.0, 0.0]]),
        "v_hat": np.array([[0.0, 0.0, 1.0]]),
        "L_ang": np.array([L_seg_ang]),
        "E_keV": np.array([E0_keV]),
        "Ne": 1,
        "thickness_ang": anchor.thick_bulk_ang,
    }
    spec = mc_spectrum(
        fake,
        anchor.E_grid,
        crystal=anchor.crystal,
        hkl_list=(anchor.hkl,),
        theta_obs_rad=anchor.theta_obs_rad,
        B_ang2=anchor.B_ang2,
    )
    mc_int = float(np.trapezoid(spec, anchor.E_grid))  # photons / electron / sr
    # Closed form per electron per sr: dOmega=1, L_abs huge so L_eff -> L_seg.
    closed = float(
        photons_per_electron(
            anchor.crystal,
            anchor.hkl,
            E_line,
            anchor.theta_obs_rad,
            beta,
            L_z_ang=L_seg_ang,
            L_abs_ang=1e12,
            dOmega_sr=1.0,
            polarization="both",
            B_ang2=anchor.B_ang2,
            use_henke=True,
            geometry="lif",
        )
    )
    return mc_int, closed, mc_int / closed


# ---- the (slow) Monte-Carlo model spectra ------------------------------------


def model_spectra(anchor: ZhaiAnchor, ne: int = 500, ne_brem: int = 200) -> dict:
    """Run the MC transport + PXR/CBS spectrum for each beam energy (1 mm bulk),
    plus the 29 nm film at the top energy. SLOW (CPU minutes for ne~500).

    Returns a dict keyed by beam energy [keV], each with the intrinsic line
    spectrum, detector-convolved line + brem, peak energy, FWHM, and the
    per-electron integrated line flux into dOmega; plus a "film" entry.
    """
    n_atoms = anchor.n_atoms_per_ang3
    E_grid = anchor.E_grid
    out: dict = {}
    for E0 in anchor.energies_keV:
        seed = int(E0 * 10)
        segs = simulate_trajectories(
            E0,
            ne,
            anchor.thick_bulk_ang,
            element="C",
            n_atoms_per_ang3=n_atoms,
            E_cut_keV=5.0,
            seed=seed,
        )
        spec = mc_spectrum(
            segs,
            E_grid,
            crystal=anchor.crystal,
            hkl_list=anchor.hkl_list,
            theta_obs_rad=anchor.theta_obs_rad,
            B_ang2=anchor.B_ang2,
        )
        beta = beta_from_keV(E0)
        E_pk = float(E_grid[np.argmax(spec)])
        fwhm = float(
            np.hypot(
                eds_fwhm_eV(E_pk),
                aperture_fwhm_eV(E_pk, beta, anchor.theta_obs_rad, anchor.dtheta_obs_rad),
            )
        )
        spec_det = convolve_detector(E_grid, spec, fwhm)
        segs_b = simulate_trajectories(
            E0,
            ne_brem,
            anchor.thick_bulk_ang,
            element="C",
            n_atoms_per_ang3=n_atoms,
            E_cut_keV=1.0,
            seed=seed + 1,
        )
        brem = mc_brem_spectrum(
            segs_b,
            E_grid,
            element="C",
            n_atoms_per_ang3=n_atoms,
            theta_obs_rad=anchor.theta_obs_rad,
        )
        brem_det = convolve_detector(E_grid, brem, fwhm)
        out[E0] = {
            "spec": spec,
            "spec_det": spec_det,
            "brem": brem,
            "brem_det": brem_det,
            "E_peak": E_pk,
            "fwhm": fwhm,
            "line_flux_per_e": float(np.trapezoid(spec, E_grid) * anchor.domega_sr),
            "backscatter": float(segs["n_backscattered"] / segs["Ne"]),
        }
    # 29 nm film at the top energy, same detector FWHM
    E_top = anchor.energies_keV[-1]
    segs_f = simulate_trajectories(
        E_top,
        ne,
        anchor.thick_film_ang,
        element="C",
        n_atoms_per_ang3=n_atoms,
        E_cut_keV=5.0,
        seed=7,
    )
    spec_f = mc_spectrum(
        segs_f,
        E_grid,
        crystal=anchor.crystal,
        hkl_list=anchor.hkl_list,
        theta_obs_rad=anchor.theta_obs_rad,
        B_ang2=anchor.B_ang2,
    )
    spec_f_det = convolve_detector(E_grid, spec_f, out[E_top]["fwhm"])
    out["film"] = {
        "E0_keV": E_top,
        "spec": spec_f,
        "spec_det": spec_f_det,
        "line_flux_per_e": float(np.trapezoid(spec_f, E_grid) * anchor.domega_sr),
        "n_transmitted": int(segs_f.get("n_transmitted", 0)),
    }
    return out


def _zhai_cache_key(anchor: ZhaiAnchor, ne: int, ne_brem: int) -> str:
    """Fingerprint inputs and implementation files that affect the reproduction."""
    digest = hashlib.sha256()
    inputs = {
        "schema": _ZHAI_CACHE_SCHEMA,
        "anchor": asdict(anchor),
        "ne": ne,
        "ne_brem": ne_brem,
    }
    digest.update(json.dumps(inputs, sort_keys=True).encode())
    digest.update(inspect.getsource(model_spectra).encode())
    implementation_files = _HERE.parent.glob("src/cxr_mc/**/*.py")
    for path in sorted(implementation_files):
        digest.update(path.relative_to(_HERE.parent).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def cached_model_spectra(
    anchor: ZhaiAnchor,
    ne: int = 500,
    ne_brem: int = 200,
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> tuple[dict, bool, Path]:
    """Load or atomically cache a Zhai reproduction keyed by inputs and code.

    Returns ``(model, cache_hit, path)``. Cache files are local generated
    artifacts under ``checkpoints/zhai_reproduction`` by default.
    """
    from cxr_mc import _checkpoint_io

    root = (
        Path(cache_dir)
        if cache_dir is not None
        else _HERE.parent / "checkpoints" / "zhai_reproduction"
    )
    path = root / f"zhai-{_zhai_cache_key(anchor, ne, ne_brem)}.pkl"
    if path.exists() and not refresh:
        return _checkpoint_io.load(str(path)), True, path

    model = model_spectra(anchor, ne=ne, ne_brem=ne_brem)
    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        _checkpoint_io.dump(model, str(tmp))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return model, False, path


# ---- Zhai supplementary coherent-emission studies ---------------------------


def supplementary_study(crystal: str) -> SupplementaryCoherentStudy:
    """Return the configured Zhai supplementary study for ``crystal``.

    Supported keys are ``wse2``, ``mose2``, and ``hbn``.  Keeping the reported
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


def model_coherent_spectra(
    study: SupplementaryCoherentStudy,
    thickness_nm: float,
    ne: int = 500,
    *,
    exploratory_azimuth_deg: float | None = None,
) -> dict[SupplementaryCondition, np.ndarray]:
    """Simulate intrinsic coherent PXR+CBS spectra for one requested thickness.

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
    if thickness_nm not in study.thicknesses_nm:
        raise ValueError(
            f"{thickness_nm:g} nm is not one of {study.crystal}'s requested "
            f"thicknesses {study.thicknesses_nm}"
        )
    if ne < 1:
        raise ValueError("ne must be positive")

    params = crystal_params(study.crystal)
    thickness_ang = float(thickness_nm) * 10.0
    spectra: dict[SupplementaryCondition, np.ndarray] = {}
    for index, condition in enumerate(study.conditions):
        azimuth_deg = condition.azimuth_deg
        if azimuth_deg is None:
            azimuth_deg = exploratory_azimuth_deg
        if azimuth_deg is None:
            raise ValueError(
                f"{study.crystal}'s TEM azimuth is unreported by Zhai et al.; "
                "pass exploratory_azimuth_deg explicitly to model this study"
            )
        beam_dir, n_hat = tilted_geometry(
            study.theta_obs_rad,
            float(np.deg2rad(condition.polar_tilt_deg)),
            float(np.deg2rad(azimuth_deg)),
        )
        # Each panel receives a reproducible, distinct transport realization.
        seed = int(thickness_nm * 100) + index
        segments = simulate_trajectories(
            condition.energy_keV,
            ne,
            thickness_ang,
            composition=params["composition"],
            beam_dir=beam_dir,
            E_cut_keV=5.0,
            seed=seed,
        )
        spectra[condition] = mc_spectrum(
            segments,
            study.E_grid,
            crystal=study.crystal,
            hkl_list=params["hkl_list"],
            n_hat=n_hat,
            B_ang2=params["B_ang2"],
            composition=params["composition"],
            beam_uvw=params["beam_uvw"],
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
        "schema": _ZHAI_CACHE_SCHEMA,
        "study": asdict(study),
        "thickness_nm": thickness_nm,
        "ne": ne,
        "exploratory_azimuth_deg": exploratory_azimuth_deg,
    }
    digest.update(json.dumps(inputs, sort_keys=True).encode())
    digest.update(inspect.getsource(model_coherent_spectra).encode())
    for path in sorted(_HERE.parent.glob("src/cxr_mc/**/*.py")):
        digest.update(path.relative_to(_HERE.parent).as_posix().encode())
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
) -> tuple[dict[SupplementaryCondition, np.ndarray], bool, Path]:
    """Load or atomically cache one supplementary coherent-only condition set."""
    from cxr_mc import _checkpoint_io

    root = (
        Path(cache_dir)
        if cache_dir is not None
        else _HERE.parent / "checkpoints" / "zhai_reproduction"
    )
    key = _supplementary_cache_key(study, thickness_nm, ne, exploratory_azimuth_deg)
    path = root / f"zhai-supplement-{key}.pkl"
    if path.exists() and not refresh:
        return _checkpoint_io.load(str(path)), True, path

    spectra = model_coherent_spectra(
        study, thickness_nm, ne=ne, exploratory_azimuth_deg=exploratory_azimuth_deg
    )
    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        _checkpoint_io.dump(spectra, str(tmp))
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
    GPU-box-runnable unit behind ``reproduce_zhai.py`` / ``cxr remote check``.

    Returns [(label, path, cache_hit)] for the Fig.1c anchor plus every
    supplementary (study, thickness) pair -- 8 entries total.
    """
    results: list[tuple[str, Path, bool]] = []
    anchor = ZhaiAnchor()
    _, hit, path = cached_model_spectra(
        anchor, ne=ne, ne_brem=ne_brem, cache_dir=cache_dir, refresh=refresh
    )
    results.append(("zhai-fig1c", path, hit))
    for crystal, study in ZHAI_SUPPLEMENTARY_STUDIES.items():
        azimuth_deg = (
            tmd_exploratory_azimuth_deg
            if any(condition.azimuth_deg is None for condition in study.conditions)
            else None
        )
        for thickness_nm in study.thicknesses_nm:
            _, hit, path = cached_coherent_spectra(
                study,
                thickness_nm,
                ne=ne_supp,
                exploratory_azimuth_deg=azimuth_deg,
                cache_dir=cache_dir,
                refresh=refresh,
            )
            results.append((f"{crystal}-{thickness_nm:g}nm", path, hit))
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


# ---- figures -----------------------------------------------------------------


def figure_spectra(anchor: ZhaiAnchor, model: dict, reference: dict | None = None):
    """Fig 1c analog vs theory in three vertically stacked stages.

    The stages are intrinsic PXR+CBS, detector-convolved PXR+CBS without the
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
                "k-",
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
    ax_i.set_title("Intrinsic PXR+CBS (1 mm HOPG)\ndotted = Eq.(10) line energy")
    ax_line.set_title("EDS-convolved PXR+CBS only (no incoherent brem)")
    ax_total.set_title("EDS-convolved total: PXR+CBS + brem (dashed)")
    for ax in (ax_i, ax_line, ax_total):
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel("Intensity (Phs/eV/s/nA)")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(r"Model vs theory: PXR+CBS from HOPG, $\theta_{obs}$=119$\degree$, 0.066 sr")
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
    ax_f.set_ylabel("Line flux (photons / electron into 0.066 sr)")
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
    L_abs = absorption_length_ang("C", E_line, anchor.n_atoms_per_ang3)
    leff_bulk = L_abs * (1.0 - np.exp(-anchor.thick_bulk_ang / L_abs))
    leff_film = L_abs * (1.0 - np.exp(-anchor.thick_film_ang / L_abs))
    ceiling = float(leff_bulk / leff_film)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(anchor.E_grid, bulk, color="C0", label=f"{E0:g} keV, 1 mm bulk")
    ax.plot(anchor.E_grid, film, color="k", label=f"{E0:g} keV, 29 nm film")
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
    intrinsic spectrum first, then convolves with the same quadrature
    EDS-plus-aperture FWHM as ``model_spectra``: Zhai SI Eqs. (14) and (16),
    at the spectrum's intrinsic peak and the condition's reported beam
    energy (photons are lost in the detector window before the sensor
    electronics blur the surviving counts in energy).  It assumes the
    Fig. 1c collection aperture (0.066 sr) and 1 nA electron rate.  In the
    zero-FWHM, unit-efficiency limit the returned spectrum is the intrinsic
    density scaled by those two factors.

    Caveat: Zhai's SI (S3/S4) never states whether their experimental
    spectra are efficiency-corrected or their theory includes window
    transmission; applying our nominal Moxtek AP3.3-class QE model here is
    a modeling choice, and the residual normalization gap vs Zhai SI
    Fig. S5b is tracked in docs/physics-validation-ledger.md (id
    `zhai-hbn-921-detected`).
    """
    anchor = ZhaiAnchor()
    peak_eV = float(study.E_grid[np.argmax(spectrum)])
    fwhm_eV = float(
        np.hypot(
            eds_fwhm_eV(peak_eV),
            aperture_fwhm_eV(
                peak_eV,
                beta_from_keV(condition.energy_keV),
                study.theta_obs_rad,
                anchor.dtheta_obs_rad,
            ),
        )
    )
    spectrum_eff = spectrum * detector_efficiency(study.E_grid)
    return convolve_detector(study.E_grid, spectrum_eff, fwhm_eV) * (
        anchor.domega_sr * anchor.per_nA
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
    if set(spectra) != set(study.conditions):
        raise ValueError("spectra must contain exactly the study's four conditions")

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True)
    for ax, condition in zip(axes.flat, study.conditions, strict=True):
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
        f"{study.label}, {study.conditions[0].energy_keV:g} keV, {thickness_nm:g} nm: "
        "Zhai-detector-convolved PXR+CBS"
    )
    # Matplotlib 3.10 can assign NaN axes bounds when tight_layout() measures
    # this shared 2×2 layout at the physical (~1e-9) intensity scale.
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.09, top=0.88, wspace=0.28, hspace=0.38)
    return fig


def figure_supplementary_hbn(
    study: SupplementaryCoherentStudy,
    thickness_nm: float,
    spectra: dict[SupplementaryCondition, np.ndarray],
):
    """Return the requested h-BN four-beam-energy coherent-only comparison."""
    import matplotlib.pyplot as plt

    if study.crystal != "hbn":
        raise ValueError("figure_supplementary_hbn is only for the h-BN study")
    if set(spectra) != set(study.conditions):
        raise ValueError("spectra must contain exactly the study's four conditions")

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, condition in enumerate(study.conditions):
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
        f"h-BN, {thickness_nm:g} nm, polar {study.conditions[0].polar_tilt_deg:g}°, "
        f"azimuth {study.conditions[0].azimuth_deg:g}°: Zhai-detector-convolved PXR+CBS"
    )
    ax.grid(alpha=0.3)
    ax.legend(title="Beam energy")
    fig.tight_layout()
    return fig


def figure_supplementary_overview(spectra: dict[str, np.ndarray]):
    """One row of three panels -- WSe2, MoSe2, h-BN side by side -- each
    showing that material's steepest requested polar tilt at its thinnest
    listed thickness: a single representative slice per material, for the
    paper's validation appendix. The full tilt x thickness grid is the
    per-material figure_supplementary_tmd/hbn panels above."""
    import matplotlib.pyplot as plt

    expected = {"wse2", "mose2", "hbn"}
    if set(spectra) != expected:
        raise ValueError(f"spectra must contain exactly {sorted(expected)}, got {sorted(spectra)}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, crystal in zip(axes, ("wse2", "mose2", "hbn"), strict=True):
        study = supplementary_study(crystal)
        thickness_nm = study.thicknesses_nm[0]
        condition = study.conditions[-1]  # steepest tilt (TMD) / highest energy (h-BN)
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
    """Render the complete publication figure set from whatever is already
    cached under checkpoints/zhai_reproduction/ (a cache miss recomputes
    locally rather than failing) -- the Fig.1c trio, every supplementary
    panel, and the cross-material overview -- to `outdir`. One command turns
    a `cxr remote check` pull into the full figure set with no per-study
    clicking in the app. Returns the list of PNG paths written (a PDF is
    written alongside each)."""
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass

    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save(name, fig):
        for ext in ("png", "pdf"):
            fig.savefig(outpath / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        written.append(outpath / f"{name}.png")

    anchor = ZhaiAnchor()
    model, _hit, _path = cached_model_spectra(anchor, ne=ne, ne_brem=ne_brem, cache_dir=cache_dir)
    reference = reference_curve(anchor)
    _save("zhai_fig1c_spectra_vs_theory", figure_spectra(anchor, model, reference))
    _save("zhai_flux_anchor", figure_flux_anchor(anchor, model))
    _save("zhai_bulk_vs_film_enhancement", figure_enhancement(anchor, model))

    overview_spectra: dict[str, np.ndarray] = {}
    for crystal, study in ZHAI_SUPPLEMENTARY_STUDIES.items():
        azimuth_deg = (
            tmd_exploratory_azimuth_deg
            if any(condition.azimuth_deg is None for condition in study.conditions)
            else None
        )
        for thickness_nm in study.thicknesses_nm:
            spectra, _hit, _path = cached_coherent_spectra(
                study,
                thickness_nm,
                ne=ne_supp,
                exploratory_azimuth_deg=azimuth_deg,
                cache_dir=cache_dir,
            )
            fig = (
                figure_supplementary_hbn(study, thickness_nm, spectra)
                if crystal == "hbn"
                else figure_supplementary_tmd(study, thickness_nm, spectra)
            )
            _save(f"zhai_supplementary_{crystal}_{thickness_nm:g}nm", fig)
            if thickness_nm == study.thicknesses_nm[0]:
                overview_spectra[crystal] = spectra[study.conditions[-1]]

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
                "line flux\n[ph/e/0.066sr]",
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

    outpath = _HERE.parent / outdir
    outpath.mkdir(exist_ok=True)
    figs = {
        "zhai_fig1c_spectra_vs_theory": figure_spectra(anchor, model, reference),
        "zhai_flux_anchor": figure_flux_anchor(anchor, model),
        "zhai_bulk_vs_film_enhancement": figure_enhancement(anchor, model),
    }
    for name, fig in figs.items():
        for ext in ("png", "pdf"):
            fig.savefig(outpath / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        print("wrote", (outpath / f"{name}.png").relative_to(_HERE.parent))


if __name__ == "__main__":
    main()
