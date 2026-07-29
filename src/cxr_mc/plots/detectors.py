"""detectors

Timepix3 and Eagle XO detector-view figures (efficiency, detected, charge).
"""

import matplotlib.pyplot as plt
import numpy as np

from ..detectors import eaglexo_response as eag
from ..detectors import timepix_response as tpx
from ..results import (
    PER_NA,
    beam_current_na,
    records,
)
from ._common import (
    _EFF_CACHE,
    _best_azimuth,
    _case_title,
    _peak_line,
    _per_tilt_figs,
)
from ._style import energy_color
from .sweeps import plot_heatmaps


# ---- Timepix3 detector view --------------------------------------------------
def _thr_keV():
    return tpx.THRESHOLD_E * tpx.W_EHP_EV / 1e3


def plot_timepix_efficiency(thickness_um=300.0, bias_v=100.0, n_mc=80000, seed=0):
    """Detection efficiency (absorption x counting turn-on) and energy
    resolution / charge-loss bias vs photon energy for the Si quad."""
    E_eff = np.arange(200.0, 60000.0, 25.0)
    key = (thickness_um, bias_v, n_mc, seed)
    resp = _EFF_CACHE.get(key)
    if resp is None:
        resp = tpx.build_response(
            E_eff,
            np.arange(0.0, 60000.0, 100.0),
            n_mc=n_mc,
            seed=seed,
            thickness_um=thickness_um,
            bias_v=bias_v,
        )
        _EFF_CACHE[key] = resp
    E_thr = _thr_keV() * 1e3
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    axL.plot(
        E_eff,
        resp["eps_abs"],
        "k--",
        lw=1.2,
        label=r"$\epsilon_\mathrm{abs}$ (Si absorption)",
    )
    axL.plot(
        E_eff,
        resp["P_det"],
        "b-",
        lw=1.8,
        label=r"$P_\mathrm{det}$ (abs $\times$ counting)",
    )
    axL.axvline(E_thr, color="r", ls=":", label=f"threshold = {E_thr:.2f} eV")
    axL.set(
        xlabel="Photon energy (eV)",
        ylabel="efficiency",
        ylim=(0, 1.05),
        title=f"Detection efficiency ({thickness_um:g} $\\mu$m Si, {bias_v:g} V, "
        f"$\\sigma_\\mathrm{{diff}}$={resp['sigma_diff_um']:.1f} $\\mu$m)",
    )
    axL.set_xscale("log")
    axL.set_xlim((min(E_eff), max(E_eff)))
    axL.margins(x=0)
    axL.grid(alpha=0.3)
    axL.legend()
    axR.plot(E_eff, tpx.energy_fwhm_eV(E_eff), "k-", lw=1.5, label="analytic, single-pixel")
    axR.plot(E_eff, resp["fwhm_rec"], "b.", ms=4, label="MC effective (tail + multi-pixel)")
    axR.set(
        xlabel="Photon energy (keV)",
        ylabel="energy FWHM (eV)",
        title="Energy resolution & charge-loss bias",
    )
    axR.margins(x=0)
    axR.grid(alpha=0.3)
    axR.legend(loc="upper left")
    axR2 = axR.twinx()
    axR2.plot(E_eff, 100 * (1 - resp["mean_rec"] / E_eff), "g-", lw=1, alpha=0.5)
    axR2.set_ylabel("mean charge-loss deficit (%)", color="g")
    axR2.tick_params(axis="y", colors="g")  # axR2.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def _tpx_detected(r, settings, thickness_um, bias_v, n_mc, seed):
    """Incident and Timepix3-detected (line + brem) [Phs/eV/s/nA] on r['E_grid'];
    the per-grid response is cached by tpx.get_response."""
    incident = (r["spec"] + r["brem"]) * r["scale"]
    resp = tpx.get_response(
        r["E_grid"], n_mc=n_mc, seed=seed, thickness_um=thickness_um, bias_v=bias_v
    )
    return incident, resp.apply(incident)


def _draw_timepix_detected(
    fig,
    trecs,
    settings,
    thickness_um=300.0,
    bias_v=100.0,
    collapse_azimuth=True,
    n_mc=80000,
    seed=0,
    floor_frac=1e-3,
):
    """Render ONE polar tilt of the Timepix detected/incident view onto ``fig``
    (cleared first): all energies overlaid, incident dotted / detected solid."""
    fig.clear()
    ax = fig.subplots(1, 1)
    E_thr = _thr_keV()
    energies = sorted({r["case"]["E0_keV"] for r in trecs})
    ymax = 0.0
    for _i, E0 in enumerate(energies):
        grp = [r for r in trecs if r["case"]["E0_keV"] == E0]
        if not grp:
            continue
        grp = _best_azimuth(grp, collapse_azimuth)
        c = energy_color(E0, energies)
        for r in sorted(grp, key=lambda r: r["case"]["tilt_azim_deg"]):
            inc, det = _tpx_detected(r, settings, thickness_um, bias_v, n_mc, seed)
            fin = inc[np.isfinite(inc)]
            if fin.size:
                ymax = max(ymax, float(fin.max()))
            az = r["case"]["tilt_azim_deg"]
            ax.plot(r["E_grid"], inc, color=c, ls=":", lw=1.0, alpha=0.7)
            ax.plot(
                r["E_grid"],
                det,
                color=c,
                ls="-",
                lw=1.2,
                label=rf"{E0:g} keV ($\phi$={az:.1f}$\degree$)",
            )
    case = trecs[0]["case"]
    ax.axvline(E_thr, color="0.4", ls=":", lw=0.8, label=f"threshold {E_thr:.2f} keV")
    if ymax > 0:
        ax.set_yscale("log")
        ax.set_ylim(ymax * floor_frac, ymax * 2)
    ax.set_title(
        _case_title(case, "Timepix3 detected (solid) vs incident (dotted)"),
        fontsize=12,
    )
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Phs/eV/s/nA")
    ax.margins(x=0)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()


def plot_timepix_detected(
    results,
    settings,
    thickness_um=300.0,
    bias_v=100.0,
    collapse_azimuth=True,
    n_mc=80000,
    seed=0,
    ncols=5,
    floor_frac=1e-3,
):
    """Incident (dotted) vs Timepix3-detected (solid) spectra, log scale; ONE
    figure per polar tilt, all energies overlaid (best azimuth each). For
    click-through use ``browse(results, settings, kind="timepix")``. (``ncols``
    is accepted for backward compatibility and ignored.)"""
    return _per_tilt_figs(
        records(results),
        settings,
        _draw_timepix_detected,
        (9.0, 5.2),
        thickness_um=thickness_um,
        bias_v=bias_v,
        collapse_azimuth=collapse_azimuth,
        n_mc=n_mc,
        seed=seed,
        floor_frac=floor_frac,
    )


def plot_timepix_poisson(
    results,
    settings,
    integration_s=600.0,
    thickness_um=300.0,
    bias_v=100.0,
    n_mc=80000,
    seed=0,
):
    """A Poisson 'measured' realization for the highest-rate config at each
    energy, over ``integration_s`` at the configured beam current."""
    recs = records(results)
    if not recs:
        print("no results yet")
        return None
    rng = np.random.default_rng(seed)
    E_thr = _thr_keV() * 1e3
    energies = sorted({r["case"]["E0_keV"] for r in recs})
    # per-panel ~4.6" with a sensible minimum total width (a single energy was
    # only 3.7" -> too narrow + a clipped suptitle); constrained_layout reserves
    # room for the suptitle instead of tight_layout clipping it.
    n = len(energies)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(max(4.6 * n, 6.8), 4.6),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, E0 in zip(axes.ravel(), energies, strict=False):
        grp = [r for r in recs if r["case"]["E0_keV"] == E0]
        r = max(grp, key=_peak_line)
        _, det = _tpx_detected(r, settings, thickness_um, bias_v, n_mc, seed)
        cur = beam_current_na(r, settings)
        counts, expected = tpx.poisson_counts(r["E_grid"], det * cur, integration_s, rng)
        ax.step(
            r["E_grid"],
            counts,
            where="mid",
            color="k",
            lw=0.7,
            label=f"measured ({integration_s:g} s @ {cur:g} nA)",
        )
        ax.plot(r["E_grid"], expected, "r-", lw=1.3, label="expected mean")
        ax.axvline(E_thr, color="b", ls=":", lw=0.8, label="threshold")
        ax.set_title(
            rf"{E0:g} keV, $\theta_\mathrm{{tilt}}={r['case']['tilt_deg']:g}\degree$, "
            rf"$\phi={r['case']['tilt_azim_deg']:g}\degree$  "
            rf"({counts.sum():.0f} cts)",
            fontsize=10,
        )
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel("counts / bin")
        ax.margins(x=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Timepix3 Poisson 'measured' spectra ({thickness_um:g} $\\mu$m Si, {bias_v:g} V)",
        fontsize=14,
    )
    return fig


# ---- Eagle XO detector view --------------------------------------------------
SI_K_EDGE_EV = 1839.0  # silicon K absorption edge -> the QE notch the lines cross


def _domega_of(r):
    """The solid angle [sr] actually baked into a record (scale = domega * PER_NA),
    so plot annotations report the geometry the sweep was run with -- not a value
    re-guessed here that might disagree with it."""
    return r["scale"] / PER_NA


def plot_eaglexo_efficiency(sensor="4240", distance_m=None, coating="BN"):
    """The Eagle XO's two knobs vs photon energy. LEFT: quantum efficiency -- the
    measured datasheet curve (BN solid, BEN dashed) plus the thin-Si absorption
    cross-check -- on log energy, with the Si-K notch marked and the soft-line /
    hard-brem regimes shaded. RIGHT: the photon-counting (energy-resolving)
    resolution, Fano + read-noise limited. The title carries the solid angle
    (knob 1) for the chosen sensor + working distance."""
    geo = eag.geometry(sensor, distance_m)
    E = np.geomspace(100.0, 60000.0, 600)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    # -- QE (knob 2) --
    axL.axvspan(E[0], 3000.0, color="g", alpha=0.05)
    axL.axvspan(3000.0, E[-1], color="r", alpha=0.05)
    axL.plot(E, eag.qe(E, "BN"), "b-", lw=1.8, label="QE (BN, no coating)")
    axL.plot(E, eag.qe(E, "BEN"), "g--", lw=1.4, label="QE (BEN, enhanced)")
    axL.plot(
        E,
        eag.qe_absorption_model(E),
        "0.5",
        ls=":",
        lw=1.2,
        label=rf"abs. model ({eag.ACTIVE_SI_UM:g} $\mu$m Si)",
    )
    axL.axvline(SI_K_EDGE_EV, color="0.4", ls="--", lw=0.8)
    axL.text(SI_K_EDGE_EV * 1.05, 0.05, "Si-K", color="0.3", fontsize=8)
    axL.set_xscale("log")
    axL.set_xlim(E[0], E[-1])
    axL.set_ylim(0, 1.0)
    axL.set(
        xlabel="Photon energy (eV)",
        ylabel="quantum efficiency",
        title="QE: soft lines pass (green), hard brem crushed (red)",
    )
    axL.grid(alpha=0.3, which="both")
    axL.legend(fontsize=8, loc="center left")
    # -- photon-counting energy resolution --
    axR.plot(E, eag.energy_fwhm_eV(E), "b-", lw=1.6)
    axR.set_xscale("log")
    axR.set_xlim(E[0], E[-1])
    axR.set(
        xlabel="Photon energy (eV)",
        ylabel="energy FWHM (eV)",
        title=f"Photon-counting resolution (Fano + {eag.READ_NOISE_E:g} e- read)",
    )
    axR.grid(alpha=0.3, which="both")
    fig.suptitle(
        f"Eagle XO {geo['sensor']} @ {geo['distance_m']:g} m  —  "
        rf"$\Omega$ = {geo['domega_sr']:.3e} sr "
        rf"($\Delta\theta$ = {geo['dtheta_obs_deg']:.2f}$\degree$, "
        f"{geo['active_mm'][0]:g}$\\times${geo['active_mm'][1]:g} mm)",  # type: ignore[reportIndexIssue]
        fontsize=13,
    )
    return fig


def _eag_detected(r, settings, coating="BN", resolve_energy=False):
    """Incident and Eagle-detected (line + brem) [Phs/eV/s/nA] on r['E_grid'];
    the per-grid response is cached by eag.get_response."""
    incident = (r["spec"] + r["brem"]) * r["scale"]
    resp = eag.get_response(r["E_grid"], coating=coating, resolve_energy=resolve_energy)
    return incident, resp.apply(incident)


def _eag_wide_brem(r, coating="BN"):
    """Wide-grid Eagle XO incident and detected brem [Phs/eV/s/nA]."""
    E = np.asarray(r["E_grid_brem"], dtype=float)
    incident = np.asarray(r["brem_wide"], dtype=float) * r["scale"]
    return E, incident, incident * eag.qe(E, coating)


def _eag_wide_charge(r, coating="BN", beam_current_na=1.0):
    """Wide-grid Eagle XO charge density [e-/eV/s] from brem."""
    E = np.asarray(r["E_grid_brem"], dtype=float)
    incident = np.nan_to_num(np.asarray(r["brem_wide"], dtype=float) * r["scale"])
    return E, incident * eag.qe(E, coating) * (E / eag.W_EHP_EV) * beam_current_na


def _draw_eaglexo_detected(
    fig,
    trecs,
    settings,
    coating="BN",
    resolve_energy=False,
    collapse_azimuth=True,
    floor_frac=1e-4,
):
    """Render ONE polar tilt of the Eagle XO detected/incident view onto ``fig``
    (cleared first): all energies overlaid, incident dotted / detected solid, on
    log-log axes so the soft lines and the hard-brem roll-off both show. The wide
    brem (out to the beam energy) is included when present -- that is where the
    thin-sensor QE roll-off visibly crushes the background. A faint QE curve on a
    right axis shows why."""
    fig.clear()
    ax = fig.subplots(1, 1)
    energies = sorted({r["case"]["E0_keV"] for r in trecs})
    ymax = 0.0
    xlo, xhi = np.inf, 0.0
    for _i, E0 in enumerate(energies):
        grp = [r for r in trecs if r["case"]["E0_keV"] == E0]
        if not grp:
            continue
        grp = _best_azimuth(grp, collapse_azimuth)
        c = energy_color(E0, energies)
        for r in sorted(grp, key=lambda r: r["case"]["tilt_azim_deg"]):
            inc, det = _eag_detected(r, settings, coating, resolve_energy)
            fin = inc[np.isfinite(inc)]
            if fin.size:
                ymax = max(ymax, float(fin.max()))
            xlo = min(xlo, float(r["E_grid"][0]))
            xhi = max(xhi, float(r["E_grid"][-1]))
            az = r["case"]["tilt_azim_deg"]
            ax.plot(r["E_grid"], inc, color=c, ls=":", lw=1.0, alpha=0.7)
            ax.plot(
                r["E_grid"],
                det,
                color=c,
                ls="-",
                lw=1.2,
                label=rf"{E0:g} keV ($\phi$={az:.1f}$\degree$)",
            )
            if r.get("brem_wide") is not None:  # full range -> the brem roll-off
                Eb, inc_b, det_b = _eag_wide_brem(r, coating)
                xhi = max(xhi, float(Eb[-1]))
                if inc_b.size:  # keep the broad brem in the y-range, not clipped
                    ymax = max(ymax, float(np.nanmax(inc_b)))
                ax.plot(Eb, inc_b, color=c, ls=":", lw=0.8, alpha=0.5)
                ax.plot(Eb, det_b, color=c, ls="-", lw=0.8, alpha=0.9)
    case = trecs[0]["case"]
    ax.axvline(SI_K_EDGE_EV, color="0.4", ls="--", lw=0.8, label="Si-K edge 1.84 keV")
    # faint QE curve on a twin axis -- the "viewed through it" envelope
    axq = ax.twinx()
    Eqe = np.geomspace(max(xlo, 1.0), xhi if xhi > 0 else 6e4, 400)
    axq.plot(Eqe, eag.qe(Eqe, coating), color="0.6", ls="-", lw=1.0, alpha=0.6)
    axq.set_ylim(0, 1.05)
    axq.set_ylabel("QE", color="0.5")
    axq.tick_params(axis="y", colors="0.5")
    if ymax > 0:
        ax.set_yscale("log")
        ax.set_ylim(ymax * floor_frac, ymax * 2)
    ax.set_xscale("log")
    if xhi > 0:
        ax.set_xlim(xlo, xhi)
    blur = ", energy-resolved" if resolve_energy else ""
    ax.set_title(
        _case_title(case, f"Eagle XO detected (solid) vs incident (dotted), {coating}{blur}"),
        fontsize=11,
    )
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Phs/eV/s/nA")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()


def plot_eaglexo_detected(
    results,
    settings,
    coating="BN",
    resolve_energy=False,
    collapse_azimuth=True,
    floor_frac=1e-4,
):
    """Incident (dotted) vs Eagle-XO-detected (solid) spectra, log-log; ONE figure
    per polar tilt, all energies overlaid (best azimuth each), with a faint QE
    envelope. Shows the camera's signature: soft PXR lines survive at ~90% QE
    while the hard bremsstrahlung is suppressed by the thin back-thinned sensor.
    Uses the wide brem grid when present (run the sweep with ``E_grid_brem``). The
    solid angle is whatever the sweep was run with -- point it at the Eagle with
    ``Sweep(..., **eaglexo_response.sweep_geometry(...))``. For click-through use
    ``browse(results, settings, kind="eaglexo")``."""
    return _per_tilt_figs(
        records(results),
        settings,
        _draw_eaglexo_detected,
        (9.0, 5.2),
        coating=coating,
        resolve_energy=resolve_energy,
        collapse_azimuth=collapse_azimuth,
        floor_frac=floor_frac,
    )


# NOTE: there is deliberately no ``plot_eaglexo_measured`` (a "measured spectrum")
# here. A bare Eagle XO is an INTEGRATING CCD -- it accumulates charge and cannot
# resolve individual photons, so it does not return a spectrum at all; presenting a
# Poisson "measured spectrum" misrepresents the instrument. What the camera
# actually reports is the recorded CHARGE -- see plot_eaglexo_charge (where the
# signal comes from) and plot_eaglexo_charge_map (the integrated geometry map).
# The energy-resolving photon-counting mode (eaglexo_response.poisson_counts /
# resolve_energy) is a special low-occupancy extra, not the default readout.


# ---- Eagle XO recorded-charge view (a CCD integrates charge, not photons) -----
def _eag_charge_rate(r, coating="BN"):
    """Total detected charge RATE [e-/s/nA] for one record: the Eagle XO
    integrates every photon that lands -- coherent lines AND bremsstrahlung --
    weighted by E/W_Si, over the full measured range when the wide brem grid is
    present. The scalar 'brightness' the CCD reports for a geometry (no spectrum).
    Coherent lines come from the fine line grid; the brem from the wide grid (or
    the line-grid brem as a fallback) so the two don't double-count their overlap."""
    resp = eag.get_response(r["E_grid"], coating=coating)
    q = resp.integrated_charge(r["spec"] * r["scale"])  # coherent lines [e-/s/nA]
    if r.get("brem_wide") is not None:
        Eb, cd_b = _eag_wide_charge(r, coating)
        q += float(np.trapezoid(cd_b, Eb))
    else:
        q += resp.integrated_charge(r["brem"] * r["scale"])
    return q


def _draw_eaglexo_charge(
    fig, trecs, settings, coating="BN", collapse_azimuth=True, floor_frac=1e-4
):
    """Render ONE polar tilt of the Eagle XO CHARGE spectral density [e-/eV/s]:
    where on the spectrum the CCD's recorded charge comes from, lines (fine grid,
    solid) + wide brem (dashed), log-log. Every photon is weighted by E/W_Si, so
    the hard brem carries far more charge than its photon count -- the legend
    reports each curve's integrated charge rate [e-/s]."""
    fig.clear()
    ax = fig.subplots(1, 1)
    energies = sorted({r["case"]["E0_keV"] for r in trecs})
    ymax, xlo, xhi = 0.0, np.inf, 0.0
    for E0 in energies:
        grp = [r for r in trecs if r["case"]["E0_keV"] == E0]
        if not grp:
            continue
        grp = _best_azimuth(grp, collapse_azimuth)
        c = energy_color(E0, energies)
        for r in sorted(grp, key=lambda r: r["case"]["tilt_azim_deg"]):
            cur = beam_current_na(r, settings)
            resp = eag.get_response(r["E_grid"], coating=coating)
            E = np.asarray(r["E_grid"], dtype=float)
            cd_line = resp.charge_density((r["spec"] + r["brem"]) * r["scale"]) * cur
            xlo, xhi = min(xlo, float(E[0])), max(xhi, float(E[-1]))
            fin = cd_line[np.isfinite(cd_line)]
            if fin.size:
                ymax = max(ymax, float(fin.max()))
            az = r["case"]["tilt_azim_deg"]
            rate = _eag_charge_rate(r, coating) * cur
            ax.plot(
                E,
                cd_line,
                color=c,
                lw=1.3,
                label=rf"{E0:g} keV ($\phi$={az:.1f}$\degree$, {rate:.2g} e$^-$/s)",
            )
            if r.get("brem_wide") is not None:
                Eb, cd_b = _eag_wide_charge(r, coating, cur)
                xhi = max(xhi, float(Eb[-1]))
                fb = cd_b[np.isfinite(cd_b)]
                if fb.size:
                    ymax = max(ymax, float(fb.max()))
                ax.plot(Eb, cd_b, color=c, ls="--", lw=0.8, alpha=0.85)
    case = trecs[0]["case"]
    ax.axvline(SI_K_EDGE_EV, color="0.4", ls="--", lw=0.8, label="Si-K edge 1.84 keV")
    if ymax > 0:
        ax.set_yscale("log")
        ax.set_ylim(ymax * floor_frac, ymax * 2)
    ax.set_xscale("log")
    if xhi > 0:
        ax.set_xlim(max(xlo, 1.0), xhi)
    ax.set_title(
        _case_title(case, f"Eagle XO recorded charge density ({coating}, dashed = brem)"),
        fontsize=11,
    )
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("charge density (e$^-$/eV/s)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()


def plot_eaglexo_charge(results, settings, coating="BN", collapse_azimuth=True, floor_frac=1e-4):
    """Eagle XO recorded CHARGE spectral density [e-/eV/s], ONE figure per polar
    tilt, all energies overlaid (best azimuth each), lines + wide brem on log-log.
    A CCD integrates charge rather than counting photons, so each photon is
    weighted by E/W_Si: this shows where the recorded signal actually comes from
    (the hard brem pulls more weight than its photon flux suggests). The companion
    to plot_eaglexo_detected (photon density) and plot_eaglexo_charge_map (the
    integrated geometry map)."""
    return _per_tilt_figs(
        records(results),
        settings,
        _draw_eaglexo_charge,
        (9.0, 5.2),
        coating=coating,
        collapse_azimuth=collapse_azimuth,
        floor_frac=floor_frac,
    )


def plot_eaglexo_charge_map(
    results,
    settings,
    cases=None,
    x="tilt_azim_deg",
    y="tilt_deg",
    panel="E0_keV",
    coating="BN",
    exposure_s=None,
    auto_lines=True,
):
    """Geometry map of the Eagle XO's recorded SIGNAL: the integrated detected
    charge rate [e-/s] (coherent lines + brem, energy-weighted -- see
    _eag_charge_rate) over ``x`` x ``y``, one panel per ``panel`` value, the best
    (max) geometry per cell. This is the "what the CCD actually reports" view the
    QE-only detected-spectrum plots miss: a CCD integrates charge, it cannot
    resolve photons, so its figure of merit is collected charge, not line flux.

    With ``exposure_s`` set, the map shows the WELL-FILL FRACTION (collected e- /
    FULL_WELL_E) for that exposure at each case's pulse current instead -- how
    close the brightest geometry comes to saturating the well. ``cases`` restricts
    to one sweep (cf. plot_heatmaps). Honors the same thin-axis -> line-plot
    fallback as plot_heatmaps (``auto_lines``): a single-valued x or y becomes a
    line plot (signal vs the varying axis, one line per the other). A thin
    wrapper around plot_heatmaps' ``value=`` mode -- see there for the shared
    panel/line machinery."""

    def _val(r):
        cur = beam_current_na(r, settings)
        rate = _eag_charge_rate(r, coating) * cur  # e-/s
        if exposure_s is not None:
            return rate * exposure_s / eag.FULL_WELL_E  # well-fill fraction
        return rate

    label = (
        f"well-fill fraction ({exposure_s:g} s; case pulse current)"
        if exposure_s is not None
        else "detected charge rate  (e$^-$/s)"
    )
    figs = plot_heatmaps(
        results,
        settings,
        cases=cases,
        x=x,
        y=y,
        panel=panel,
        value=_val,
        value_label=label,
        value_cmap="inferno",
        auto_lines=auto_lines,
        title=f"Eagle XO recorded signal: {label}  ({coating}, best per cell)",
    )
    return figs[0] if figs else None
