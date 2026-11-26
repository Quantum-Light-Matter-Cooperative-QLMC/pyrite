"""Non-basal CXR of fiber-textured crystals (HOPG, pyrolytic h-BN), #1480 Phase I.

The catalog restricts fiber crystals to their basal ``(00l)`` domain on the
argument that random crystallite azimuths remove non-basal lines. This script
measures what that restriction omits. It is research evidence; it changes no
production default.

Subcommands::

  structure  CRYSTAL          family table: |S|^2, multiplicity, g_z, g_perp, DW
  kinematics                  classical/exact resonance vs crystallite azimuth;
                              asserts the ``cxr-fiber-azimuthal-kinematics`` claims
  single-flight [--json OUT]  per-family PXR/CBS yields of one straight flight,
                              azimuth averaged, against (002)
  frozen CRYSTAL E0_KEV THICK_NM TILT_DEG [--electrons N] [--json OUT]
                              arms A-E on one frozen local transport
  campaign E0_KEV THICK_NM TILT_DEG --energy-max EV [--json OUT]
                              #1483 campaign fiber route vs the Q arm

Conventions (production): sample +z is the reciprocal ``surface_hkl = (001)``
axis (``_orientation_R``); a crystallite at azimuth ``phi`` is rolled by
``Rz(phi)`` (``azimuth_rad``). Lines exist where ``v.g > 0``; the electron
gives up ``hbar g`` (``p_i = p_f + hbar g + hbar k``, ``materials.recoil``).
Anisotropic Debye--Waller: ``W = (U_perp g_perp^2 + U_par g_z^2)/2``, passed
to the scalar-``B`` kernel as ``B_eff = 8 pi^2 (U_perp g_perp^2 + U_par
g_z^2)/g^2`` (exact for one uniaxial site tensor).

Validation: cxr-fiber-azimuthal-kinematics, graphite-nonbasal-structure-factor

Run:  pixi run python checks/fiber_nonbasal_cxr_1480.py kinematics
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np

from pyrite.materials._reflection_thermal import common_uniaxial_b
from pyrite.materials.crystal import (
    CRYSTALS,
    HBARC_EV_ANG,
    M_E_EV,
    _orientation_R,
    beta_from_Ee,
    reciprocal_g_vector,
    structure_factor,
)
from pyrite.materials.recoil import recoil_resonance_eV

# Graphite ADPs come from the packaged #535 source, not duplicated constants.
# hBN lacks a complete tensor; its scalar B33 fallback remains provisional.
TENSOR_CRYSTALS = frozenset({"hopg"})
FIBER_CRYSTALS = ("hbn", "hopg")


# ---- reciprocal families --------------------------------------------------
def _frame_R(crystal):
    """Production crystal->sample rotation for the catalog cut, zero azimuth."""
    return _orientation_R(CRYSTALS[crystal]["lattice"], None, 0.0, None, (0, 0, 1))


def g_sample(crystal, hkl, phi=0.0):
    """Sample-frame g [1/A] of ``hkl`` for a crystallite rolled by ``phi``."""
    R = _orientation_R(CRYSTALS[crystal]["lattice"], None, phi, None, (0, 0, 1))
    g, _ = reciprocal_g_vector(hkl, CRYSTALS[crystal]["lattice"])
    return np.asarray(R @ g if R is not None else g, dtype=float)


def geometric_s2(crystal, hkl):
    """``|sum_j exp(2 pi i h.R_j)|^2`` with unit form factors."""
    s = sum(np.exp(2j * np.pi * np.dot(hkl, R)) for _, R in CRYSTALS[crystal]["basis"])
    return float(abs(s) ** 2)


def b_effective(crystal, hkl, B_ang2):
    """Effective B for one reflection, retaining exact catalog basal B33."""
    if crystal in TENSOR_CRYSTALS and (hkl[0] != 0 or hkl[1] != 0):
        return common_uniaxial_b(crystal, hkl, 300.0)
    return B_ang2


def families(crystal, g_max=8.0, B_ang2=None):
    """Fiber families: members equivalent under rotation about c.

    Grouped by ``(l, |g_perp|, |S_geo|^2)``; every allowed member is kept,
    including ``hk0`` and odd ``l``. Returns rows sorted by ``(|l|, g_perp)``.
    """
    from pyrite.materials.catalog import load_material_catalog

    if B_ang2 is None:
        B_ang2 = load_material_catalog().crystal(crystal).B_ang2
    lat = CRYSTALS[crystal]["lattice"]
    a_len = max(lat["a"], lat["b"])
    n_ab = int(g_max * a_len / (2 * math.pi)) + 1
    n_c = int(g_max * lat["c"] / (2 * math.pi)) + 1
    groups = defaultdict(list)
    for h in range(-n_ab - 1, n_ab + 2):
        for k in range(-n_ab - 1, n_ab + 2):
            for ell in range(-n_c, n_c + 1):
                if (h, k, ell) == (0, 0, 0):
                    continue
                g = g_sample(crystal, (h, k, ell))
                if np.linalg.norm(g) > g_max:
                    continue
                s2 = geometric_s2(crystal, (h, k, ell))
                key = (ell, round(math.hypot(g[0], g[1]), 6), round(s2, 6))
                groups[key].append((h, k, ell))
    rows = []
    for (ell, g_perp, s2_geo), members in groups.items():
        rep = max(members, key=lambda m: (min(m) >= 0, m))
        g = g_sample(crystal, rep)
        S0, gmag = structure_factor(crystal, rep, 1000.0, B_ang2=0.0, use_henke=False)
        b_eff = b_effective(crystal, rep, B_ang2)
        dw_t = math.exp(-b_eff * gmag**2 / (16 * math.pi**2))
        dw_s = math.exp(-B_ang2 * gmag**2 / (16 * math.pi**2))
        rows.append(
            dict(
                rep=rep,
                members=sorted(members),
                multiplicity=len(members),
                l=ell,
                g=float(gmag),
                g_z=float(g[2]),
                g_perp=g_perp,
                s2_geo=s2_geo,
                s2_f0=float(abs(S0) ** 2),
                dw2_tensor=dw_t**2,
                dw2_scalar=dw_s**2,
                b_eff=b_eff,
            )
        )
    return sorted(rows, key=lambda r: (abs(r["l"]), r["g_perp"], -r["l"]))


def cmd_structure(args):
    rows = families(args.crystal, args.g_max)
    print(f"{args.crystal}: fiber families with |g| <= {args.g_max} 1/A, l >= 0 shown")
    print(
        f"{'rep':>12} {'m':>3} {'|g|':>7} {'g_z':>7} {'g_perp':>7} {'|S|^2geo':>9}"
        f" {'|S|^2 f0':>9} {'e^-2W ten':>9} {'e^-2W B33':>9} {'B_eff':>6}"
    )
    for r in rows:
        if r["l"] < 0:
            continue
        print(
            f"{str(r['rep']):>12} {r['multiplicity']:>3} {r['g']:7.3f} {r['g_z']:7.3f}"
            f" {r['g_perp']:7.3f} {r['s2_geo']:9.4f} {r['s2_f0']:9.4f}"
            f" {r['dw2_tensor']:9.4f} {r['dw2_scalar']:9.4f} {r['b_eff']:6.3f}"
        )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
    if args.crystal == "hopg":
        _assert_graphite_classes(rows)


def _assert_graphite_classes(rows):
    """``graphite-nonbasal-structure-factor``: |S|^2 classes, orbits and g_perp.

    Six-member families hold below the {21} row (g_perp = 7.80 1/A); from it on, the mirror
    orbits {hk} and {kh} share l, g_perp and |S| and group as twelve.
    """
    bad = []
    a_star = 4 * math.pi / (math.sqrt(3) * CRYSTALS["hopg"]["lattice"]["a"])
    rows_allowed = {round(a_star * x, 3) for x in (0.0, 1.0, math.sqrt(3), 2.0)}
    for r in rows:
        for h, k, ell in r["members"]:
            m0 = (h - k) % 3 == 0
            want = (16.0 if m0 else 1.0) if ell % 2 == 0 else (0.0 if m0 else 3.0)
            if abs(geometric_s2("hopg", (h, k, ell)) - want) > 1e-9:
                bad.append(((h, k, ell), "class"))
        if r["g_perp"] > 0 and r["g_perp"] < 7.79 and r["multiplicity"] != 6:
            bad.append((r["rep"], f"multiplicity {r['multiplicity']}"))
        if r["g_perp"] < 7.79 and round(r["g_perp"], 3) not in rows_allowed:
            bad.append((r["rep"], f"g_perp {r['g_perp']}"))
    if bad:
        raise SystemExit(f"graphite structure-factor checks failed: {bad[:8]}")
    print(f"PASS graphite |S|^2 classes, six-member orbits and g_perp over {len(rows)} families")


# ---- kinematics -------------------------------------------------------------
def _unit(theta, psi=0.0):
    return np.array(
        [math.sin(theta) * math.cos(psi), math.sin(theta) * math.sin(psi), math.cos(theta)]
    )


def classical_eV(T_eV, u, g, n):
    """``hbar c beta u.g / (1 - beta u.n)`` (vacuum, no recoil)."""
    beta = float(beta_from_Ee(T_eV))
    return HBARC_EV_ANG * beta * (u @ g) / (1 - beta * (u @ n))


def route_eV(T_eV, u, g, n, hbarc=HBARC_EV_ANG):
    """Production route form ``(v.g - r_num)/(D + r_den)``, with a hbar hook."""
    from pyrite.montecarlo.spectrum.cxr import _recoil

    saved = _recoil._RECOIL_HBARC_EV_ANG
    _recoil._RECOIL_HBARC_EV_ANG = hbarc
    try:
        beta = float(beta_from_Ee(T_eV))
        gamma = 1 + T_eV / M_E_EV
        r_num, r_den, _ = _recoil._line_recoil_terms(gamma, float(g @ g), float(n @ g))
        omega = (beta * (u @ g) - r_num) / (1 - beta * (u @ n) + r_den)
    finally:
        _recoil._RECOIL_HBARC_EV_ANG = saved
    return HBARC_EV_ANG * omega


def azimuth_scan(crystal, hkl, T_eV, alpha, theta, psi=0.0, n_phi=721):
    """Classical and exact resonance [eV] of ``hkl`` over crystallite azimuth."""
    phi = np.linspace(0, 2 * math.pi, n_phi)
    u = _unit(alpha)
    n = _unit(theta, psi)
    gs = np.array([g_sample(crystal, hkl, p) for p in phi])
    cl = np.array([classical_eV(T_eV, u, g, n) for g in gs])
    ex = recoil_resonance_eV(T_eV, u, gs, n)
    return phi, cl, np.asarray(ex)


KIN_FAMILIES = (
    (0, 0, 2),
    (1, 0, 2),
    (1, 1, 2),
    (2, 0, 2),
    (0, 0, 4),
    (1, 0, 4),
    (1, 1, 4),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (1, 0, 3),
)


def cmd_kinematics(args):
    crystal = "hopg"
    failures = []

    def check(ok, label):
        print(("  PASS " if ok else "  FAIL ") + label)
        if not ok:
            failures.append(label)

    g002 = g_sample(crystal, (0, 0, 2))
    g102 = g_sample(crystal, (1, 0, 2))
    ratio = math.hypot(g102[0], g102[1]) / g102[2]
    print(
        f"HOPG g_z(002) = {g002[2]:.4f}, g_perp(10l) = {math.hypot(*g102[:2]):.4f}, ratio {ratio:.4f}"
    )

    # 1. axial classical degeneracy: every member, every phi, same l
    print("1. axial classical degeneracy (alpha = 0)")
    for T in (30e3, 1e6):
        worst = 0.0  # over every polar/azimuthal observation angle below
        for theta in np.deg2rad([20.0, 90.0, 119.0, 160.0]):
            for psi in (0.0, 0.7):
                ref = classical_eV(T, _unit(0), g002, _unit(theta, psi))
                for hkl in ((1, 0, 2), (1, 1, 2), (2, 0, 2), (-1, 2, 2)):
                    _, cl, _ = azimuth_scan(crystal, hkl, T, 0.0, theta, psi, 181)
                    worst = max(worst, float(np.max(np.abs(cl / ref - 1))))
        check(worst < 1e-12, f"T={T / 1e3:g} keV: max |E_cl(hkl,phi)/E_cl(002) - 1| = {worst:.1e}")

    # 2. production parity and recoil phi dependence
    print("2. exact root: materials.recoil vs CXR route, and its phi dependence")
    worst = 0.0
    for T in (30e3, 100e3, 1e6):
        for hkl in KIN_FAMILIES:
            for theta in np.deg2rad([30.0, 119.0]):
                u, n = _unit(math.radians(3.0)), _unit(theta, 0.4)
                for p in np.linspace(0, 2 * math.pi, 13):
                    g = g_sample(crystal, hkl, p)
                    ex = float(recoil_resonance_eV(T, u, g, n))
                    if np.isfinite(ex):
                        worst = max(worst, abs(route_eV(T, u, g, n) / ex - 1))
    check(worst < 1e-10, f"route vs recoil_resonance_eV max rel diff {worst:.1e}")

    print("   axial exact (102),(112) vs (002), theta = 119 deg (shift mean, half-range):")
    for T in (30e3, 100e3, 300e3, 1e6):
        _, _, ex002 = azimuth_scan(crystal, (0, 0, 2), T, 0.0, math.radians(119), n_phi=5)
        e0 = float(ex002[0])
        out = []
        for hkl in ((1, 0, 2), (1, 1, 2)):
            _, cl, ex = azimuth_scan(crystal, hkl, T, 0.0, math.radians(119))
            g = g_sample(crystal, hkl)
            beta, E = float(beta_from_Ee(T)), T + M_E_EV
            D = 1 - beta * math.cos(math.radians(119))
            # first order: dE/E = -hbar c g^2/(2 E beta g_z) - hbar c n.g/(E D)
            mean1 = -HBARC_EV_ANG * (g @ g) / (2 * E * beta * g[2]) - HBARC_EV_ANG * g[
                2
            ] * math.cos(math.radians(119)) / (E * D)
            half1 = HBARC_EV_ANG * math.hypot(g[0], g[1]) * math.sin(math.radians(119)) / (E * D)
            mean = 0.5 * (ex.max() + ex.min()) / cl[0] - 1
            half = 0.5 * (ex.max() - ex.min()) / cl[0]
            # claimed term: fractional shift relative to the (00l) exact line
            rel = 0.5 * (ex.max() + ex.min()) / e0 - 1
            rel1 = -HBARC_EV_ANG * (g[0] ** 2 + g[1] ** 2) / (2 * E * beta * g[2])
            check(
                abs(rel - rel1) < 0.05 * abs(rel1) + 2e-4,
                f"   {hkl} {T / 1e3:g} keV shift vs (00l) {rel:+.5f} (claimed {rel1:+.5f})",
            )
            check(
                abs(mean - mean1) < 0.05 * abs(mean1) + 2e-4,
                f"   {hkl} {T / 1e3:g} keV total recoil shift vs classical, first order",
            )
            out.append((hkl, e0, ex.min(), ex.max(), mean, mean1, half, half1))
        for hkl, e0, lo, hi, mean, mean1, half, half1 in out:
            print(
                f"   T={T / 1e3:6g} keV {hkl}: E002={e0:9.2f}  [{lo:9.2f},{hi:9.2f}] eV"
                f"  shift {mean:+.4f} (1st order {mean1:+.4f})  half {half:.4f} ({half1:.4f})"
            )
            check(
                abs(half / half1 - 1) < 0.1 * (1 + 30e3 / T),
                f"   {hkl} {T / 1e3:g} keV half-range vs first order",
            )

    # 3. oblique broadening (classical)
    print("3. oblique incidence broadening: half-range/mean = (g_perp/g_z) tan(alpha)")
    for alpha_deg in (1.0, 5.0):
        alpha = math.radians(alpha_deg)
        _, cl, _ = azimuth_scan(crystal, (1, 0, 2), 300e3, alpha, math.radians(90.0))
        # theta = 90 deg in the x-z plane mixes D with phi only through u.n;
        # use the numerator alone (D fixed by u, n) for the identity.
        u = _unit(alpha)
        num = np.array(
            [u @ g_sample(crystal, (1, 0, 2), p) for p in np.linspace(0, 2 * math.pi, 721)]
        )
        rel = 0.5 * (num.max() - num.min()) / (0.5 * (num.max() + num.min()))
        pred = ratio * math.tan(alpha)
        print(
            f"   alpha={alpha_deg:g} deg: measured {rel:.4f}, predicted {pred:.4f}; E range "
            f"[{cl.min():.1f}, {cl.max():.1f}] eV"
        )
        check(abs(rel / pred - 1) < 1e-9, f"   alpha={alpha_deg:g} deg identity")

    # 4. vanishing recoil limit
    print("4. vanishing recoil: hbar -> 0 hook reproduces the classical root; shift ~ 1/E")
    worst = 0.0
    for hkl in KIN_FAMILIES:
        for p in (0.0, 1.1):
            g = g_sample(crystal, hkl, p)
            u, n = _unit(math.radians(2.0)), _unit(math.radians(119), 0.3)
            cl = classical_eV(100e3, u, g, n)
            if cl > 0:
                worst = max(worst, abs(route_eV(100e3, u, g, n, hbarc=0.0) / cl - 1))
    check(worst < 1e-13, f"hook route == classical, max rel {worst:.1e}")
    g = g_sample(crystal, (1, 0, 2), 0.5)
    shifts = []
    for T in (1e7, 1e8, 1e9):
        u, n = _unit(0), _unit(math.radians(119))
        shifts.append(float(recoil_resonance_eV(T, u, g, n)) / classical_eV(T, u, g, n) - 1)
    slope = shifts[1] / shifts[0], shifts[2] / shifts[1]
    print(f"   (102) shift at 10/100/1000 MeV: {shifts}; decade ratios {slope}")
    check(all(abs(s - 0.1) < 0.01 for s in slope), "   ultra-relativistic shift scales as 1/E")

    if args.plot:
        _plot_kinematics(crystal, args.plot)
    if failures:
        raise SystemExit(f"{len(failures)} kinematics check(s) failed")
    print("all kinematics checks passed")


def _plot_kinematics(crystal, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (T, alpha_deg) in zip(axes, ((30e3, 0.0), (1e6, 0.0), (300e3, 5.0)), strict=True):
        for hkl in ((0, 0, 2), (1, 0, 2), (1, 1, 2), (2, 0, 2)):
            phi, cl, ex = azimuth_scan(crystal, hkl, T, math.radians(alpha_deg), math.radians(119))
            ax.plot(np.rad2deg(phi), ex, label=f"{hkl} exact")
        ax.plot(np.rad2deg(phi), cl, "k--", lw=0.8, label="classical (102)")
        ax.set(
            xlabel="crystallite azimuth phi [deg]",
            ylabel="E_res [eV]",
            title=f"HOPG {T / 1e3:g} keV, tilt {alpha_deg:g} deg, 119 deg",
        )
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


# ---- spectra ----------------------------------------------------------------
def _quadrature(n_phi):
    """Uniform azimuths over one 60-degree sector (sixfold family closure)."""
    return (np.arange(n_phi) + 0.5) * (math.pi / 3) / n_phi


def family_spectrum(
    segments,
    E_grid,
    crystal,
    fam,
    n_hat,
    n_phi,
    composition,
    B_ang2,
    components=True,
    fixed_phi=None,
    phis=None,
):
    """Azimuth-averaged incoherent spectrum of one fiber family, all members.

    Members are summed per azimuth and averaged over ``n_phi`` uniform roll
    angles in one 60-degree sector; for a sixfold-closed family this equals
    the full-circle average. ``fixed_phi`` evaluates one orientation (arm B).
    """
    from pyrite.montecarlo import mc_cxr_spectrum

    if phis is None:
        phis = [fixed_phi] if fixed_phi is not None else _quadrature(n_phi)
    # Basal rows are invariant under roll; preserve the exact catalog B33
    # used by the production baseline instead of its rounded tensor value.
    if fam["g_perp"] == 0.0:
        phis = [0.0]
        b_eff = B_ang2
    else:
        b_eff = fam["b_eff"]
    total = None
    for phi in phis:
        out = mc_cxr_spectrum(
            segments,
            E_grid,
            crystal,
            fam["members"],
            n_hat=n_hat,
            B_ang2=b_eff,
            composition=composition,
            surface_hkl=(0, 0, 1),
            azimuth_rad=float(phi),
            components=components,
            line_quadrature="bin-mean",
            use_henke=True,
        )
        out = np.stack(out) if components else np.asarray(out)[None]
        total = out if total is None else total + out
    return total / len(phis)


def straight_flight(T_keV, length_ang, alpha_deg=0.0, depth_ang=None):
    """One constant-velocity flight through a slab of thickness ``length cos(alpha)``."""
    alpha = math.radians(alpha_deg)
    u = _unit(alpha)
    thick = length_ang * math.cos(alpha)
    r_mid = u * (0.5 * length_ang)
    return {
        "r_mid": r_mid[None, :],
        "v_hat": u[None, :],
        "L_ang": np.array([length_ang]),
        "E_keV": np.array([T_keV]),
        "t_ang": np.zeros(1),
        "t0_ang": np.zeros(1),
        "elec_id": np.zeros(1, dtype=int),
        "layer": np.zeros(1, dtype=int),
        "Ne": 1,
        "thickness_ang": thick,
        "crystal_width_ang": 1e9,
        "crystal_height_ang": 1e9,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
    }


def _grid_for(T_keV, g_max, n_hat, alpha_deg, step):
    beta = float(beta_from_Ee(T_keV * 1e3))
    u = _unit(math.radians(alpha_deg))
    top = HBARC_EV_ANG * beta * g_max / max(1 - beta * float(u @ n_hat), 1e-3) * 1.05
    return np.arange(step, min(top, 0.95 * T_keV * 1e3), step)


def cmd_single_flight(args):
    from pyrite.materials.catalog import load_material_catalog

    results = []
    for crystal in ("hopg", "hbn"):
        cat = load_material_catalog().crystal(crystal)
        fams = families(crystal, args.g_max)
        for T_keV in (30.0, 100.0, 1000.0):
            for theta_deg in (20.0, 45.0, 119.0, 160.0):
                for alpha_deg in (0.0, 5.0):
                    n_hat = _unit(math.radians(theta_deg))
                    seg = straight_flight(T_keV, args.length, alpha_deg)
                    E = _grid_for(T_keV, args.g_max, n_hat, alpha_deg, args.step)
                    row = dict(
                        crystal=crystal,
                        T_keV=T_keV,
                        theta_deg=theta_deg,
                        alpha_deg=alpha_deg,
                        length_ang=args.length,
                        families={},
                    )
                    started = time.time()
                    for fam in fams:
                        spec = family_spectrum(
                            seg, E, crystal, fam, n_hat, args.n_phi, cat.composition, cat.B_ang2
                        )
                        dE = np.gradient(E)
                        tot, pxr, cbs = (float(np.sum(s * dE)) for s in spec)
                        if tot <= 0:
                            continue
                        peak = float(E[np.argmax(spec[0])])
                        row["families"][str(fam["rep"])] = dict(
                            yield_=tot, pxr=pxr, cbs=cbs, peak_eV=peak, m=fam["multiplicity"]
                        )
                    basal = sum(
                        v["yield_"] for k, v in row["families"].items() if k.startswith("(0, 0,")
                    )
                    allv = sum(v["yield_"] for v in row["families"].values())
                    row.update(basal=basal, all=allv, seconds=time.time() - started)
                    results.append(row)
                    top = sorted(row["families"].items(), key=lambda kv: -kv[1]["yield_"])[:5]
                    print(
                        f"{crystal} {T_keV:6g} keV th={theta_deg:5g} a={alpha_deg:3g}: "
                        f"nonbasal/basal = {(allv - basal) / basal if basal else float('nan'):.3e}; top "
                        + ", ".join(f"{k}:{v['yield_'] / basal:.2e}" for k, v in top)
                    )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)


# ---- frozen transport arms ----------------------------------------------------
def _electron_subset(segs, mask_e):
    """Segments of the electrons ``mask_e`` (by id), ids renumbered from zero."""
    ne = int(segs["Ne"])
    ids = np.asarray(segs["elec_id"])
    keep = mask_e[ids]
    nseg = ids.size
    new_id = np.cumsum(mask_e) - 1
    out = {}
    for key, val in segs.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1:
            if key.startswith("initial_") and val.shape[0] == ne:
                out[key] = val[mask_e]
                continue
            if val.shape[0] == nseg:
                out[key] = val[keep]
                continue
            if val.shape[0] == ne:
                out[key] = val[mask_e]
                continue
        out[key] = val
    out["elec_id"] = new_id[ids[keep]]
    if "electron_id" in segs:
        out["electron_id"] = out["elec_id"].copy()
    out["Ne"] = int(mask_e.sum())
    return out


def _line_metrics(E, spec):
    """Integrated yield, peak, centroid and FWHM of the strongest feature."""
    dE = np.gradient(E)
    total = float(np.sum(spec * dE))
    if total <= 0:
        return dict(yield_=0.0)
    i = int(np.argmax(spec))
    half = spec[i] / 2
    lo, hi = i, i
    while lo > 0 and spec[lo] > half:
        lo -= 1
    while hi < spec.size - 1 and spec[hi] > half:
        hi += 1
    return dict(
        yield_=total,
        peak_eV=float(E[i]),
        peak=float(spec[i]),
        centroid_eV=float(np.sum(E * spec * dE) / total),
        fwhm_eV=float(E[hi] - E[lo]),
    )


def _spectral_error(E, candidate, reference):
    """Common-bin absolute mass L1 and peak-normalized L-infinity errors."""
    mass = float(np.sum(np.abs(reference) * np.gradient(E)))
    peak = float(np.max(np.abs(reference)))
    diff = np.abs(candidate - reference)
    return dict(
        mass_l1=float(np.sum(diff * np.gradient(E)) / mass) if mass else None,
        peak_linf=float(np.max(diff) / peak) if peak else None,
    )


def cmd_frozen(args):
    from pyrite.materials.catalog import load_material_catalog
    from pyrite.montecarlo.geometry import detector_directions
    from pyrite.montecarlo.transport import _simulate_trajectories

    if args.electrons < 1 or args.step <= 0 or args.g_max <= 0:
        raise ValueError("positive electrons, step and g-max required")
    if args.reference < 1 or any(k < 1 for k in (*args.samples, *args.quadratures)):
        raise ValueError("all azimuth counts must be positive")
    if any(k > args.reference for k in args.quadratures):
        raise ValueError("reference must be at least as fine as every sector quadrature")
    if args.full_circle_reference < 0:
        raise ValueError("full-circle-reference must be nonnegative")
    if args.full_circle_reference and math.gcd(args.full_circle_reference, 6) > 1:
        raise ValueError("full-circle-reference must be coprime with 6, or its nodes alias")
    cuts = sorted(set([*args.g_cuts, args.g_max]))
    if any(g <= 0 or g > args.g_max for g in cuts):
        raise ValueError("g-cuts must be positive and no greater than g-max")
    cat = load_material_catalog().crystal(args.crystal)
    tp, ta = math.radians(args.tilt), math.radians(args.tilt_azim)
    t0 = time.time()
    segs = _simulate_trajectories(
        args.E0,
        args.electrons,
        args.thick_nm * 10.0,
        composition=cat.composition,
        seed=args.seed,
        tilt_polar_rad=tp,
        tilt_azim_rad=ta,
        radiative_model="uncoupled",
    )
    t_transport = time.time() - t0
    n_hats, _ = detector_directions(math.radians(args.theta), tp, ta, n_side=1, domega_sr=1.0)
    n_hat = n_hats[0]
    fams = families(args.crystal, args.g_max)
    basal = [f for f in fams if f["g_perp"] == 0.0]
    u0 = np.asarray(segs["initial_v_hat"], dtype=float).mean(axis=0)
    u0 /= np.linalg.norm(u0)
    beta0 = float(beta_from_Ee(args.E0 * 1e3))
    top = 1.3 * HBARC_EV_ANG * beta0 * args.g_max / (1 - beta0 * float(u0 @ n_hat))
    # The old automatic ceiling is an incident-direction heuristic, not a
    # guarantee for all scattered histories. An explicit band lets cutoff
    # and grid-refinement runs compare exactly the same photon energies.
    stop = min(top, 0.95 * args.E0 * 1e3) if args.energy_max is None else args.energy_max
    if not 0 < stop <= args.E0 * 1e3:
        raise ValueError("energy-max must be positive and at most the incident kinetic energy")
    E = np.arange(args.step, stop, args.step)
    if E.size < 2:
        raise ValueError("energy band requires at least two grid nodes")
    rng = np.random.default_rng(args.seed + 1480)
    ne = int(segs["Ne"])
    record = dict(
        crystal=args.crystal,
        E0_keV=args.E0,
        thick_nm=args.thick_nm,
        tilt_deg=args.tilt,
        tilt_azim_deg=args.tilt_azim,
        theta_deg=args.theta,
        n_hat=n_hat.tolist(),
        electrons=ne,
        segments=int(np.asarray(segs["L_ang"]).size),
        seed=args.seed,
        g_max=args.g_max,
        g_cuts=cuts,
        step_eV=args.step,
        energy_band_eV=[float(E[0] - args.step / 2), float(E[-1] + args.step / 2)],
        automatic_band=args.energy_max is None,
        thermal_model=(
            "packaged uniaxial ADP (300 K), catalog basal B33"
            if args.crystal in TENSOR_CRYSTALS
            else "provisional scalar B33"
        ),
        continuum="not scored; CXR line yield only",
        transport_s=t_transport,
        arms={},
    )
    spectra = {"E_eV": E}

    def run_arm(name, fam_list, phis_per_stratum, b_override=None):
        started = time.time()
        acc = np.zeros((3, E.size))
        per_fam = defaultdict(lambda: np.zeros((3, E.size)))
        strata = len(phis_per_stratum)
        label = rng.permutation(strata)[np.arange(ne) % strata]
        rng.shuffle(label)
        stratum_counts = np.bincount(label, minlength=strata)
        for j, phis in enumerate(phis_per_stratum):
            mask = label == j
            # With S128 and fewer than 128 electrons there are empty strata.
            # A nonempty stratum can also contain only missed electrons.
            # Its incident weight still belongs in the overall denominator.
            if not stratum_counts[j]:
                continue
            sub = _electron_subset(segs, mask) if strata > 1 else segs
            w = stratum_counts[j] / ne
            if not np.asarray(sub["L_ang"]).size:
                continue
            for fam in fam_list:
                b = b_override if b_override is not None else fam["b_eff"]
                f2 = dict(fam, b_eff=b)
                s = family_spectrum(
                    sub,
                    E,
                    args.crystal,
                    f2,
                    n_hat,
                    None,
                    cat.composition,
                    cat.B_ang2,
                    fixed_phi=None,
                    phis=phis,
                )
                per_fam[str(fam["rep"])] += w * s
                acc += w * s
        dE = np.gradient(E)
        fam_yield = {k: float(np.sum(v[0] * dE)) for k, v in per_fam.items()}
        out = _line_metrics(E, acc[0])
        pxr = float(np.sum(acc[1] * dE))
        cbs = float(np.sum(acc[2] * dE))
        cutoff_metrics = {}
        for cutoff in cuts:
            prefix = np.zeros_like(acc)
            basal_prefix = np.zeros(E.size)
            for fam in fam_list:
                key = str(fam["rep"])
                if fam["g"] <= cutoff and key in per_fam:
                    prefix += per_fam[key]
                    if fam["g_perp"] == 0.0:
                        basal_prefix += per_fam[key][0]
            metrics = _line_metrics(E, prefix[0])
            metrics.update(_spectral_error(E, prefix[0], acc[0]))
            metrics["basal_yield"] = float(np.sum(basal_prefix * dE))
            metrics["nonbasal_yield"] = metrics["yield_"] - metrics["basal_yield"]
            cutoff_metrics[str(cutoff)] = metrics
            spectra[f"{name}_g{cutoff:g}"] = prefix
        out.update(
            pxr=pxr,
            cbs=cbs,
            interference=out["yield_"] - pxr - cbs,
            families={k: v for k, v in sorted(fam_yield.items(), key=lambda kv: -kv[1]) if v > 0},
            cutoffs=cutoff_metrics,
            stratum_counts=stratum_counts.tolist(),
            azimuths_rad=[np.asarray(phis).tolist() for phis in phis_per_stratum],
            seconds=time.time() - started,
        )
        if args.save_families and name == f"Q{args.reference}":
            for key, value in per_fam.items():
                spectra[f"{name}_family_{key}"] = value
        record["arms"][name] = out
        spectra[name] = acc
        print(
            f"  {name:>10}: yield {out['yield_']:.4e}  peak {out.get('peak_eV', 0):8.1f} eV"
            f"  fwhm {out.get('fwhm_eV', 0):6.1f}  {out['seconds']:.0f} s",
            flush=True,
        )

    print(
        f"{args.crystal} {args.E0} keV {args.thick_nm} nm tilt {args.tilt} theta {args.theta}: "
        f"{record['segments']} segments, {ne} e, transport {t_transport:.0f} s",
        flush=True,
    )
    run_arm("A_basal", basal, [[0.0]], b_override=cat.B_ang2)
    run_arm("B_fixed", fams, [[0.0]])
    for k in dict.fromkeys(args.samples):
        # stratified uniform azimuths on the full circle, one per electron stratum
        phis = 2 * math.pi * (np.arange(k) + rng.random(k)) / k
        run_arm(f"S{k}", fams, [[p] for p in phis])
    # Unlike S, Q evaluates every history at every node: this measures
    # quadrature convergence separately from the stochastic texture estimator.
    for k in sorted(set([*args.quadratures, args.reference])):
        run_arm(f"Q{k}", fams, [list(_quadrature(k))])
    if args.full_circle_reference:
        k = args.full_circle_reference
        phis = 2 * math.pi * (np.arange(k) + 0.5) / k
        run_arm(f"F{k}", fams, [phis])
    if args.b33_iso:
        run_arm("B33iso_Q", fams, [list(_quadrature(args.reference))], b_override=cat.B_ang2)
    ref_name = f"Q{args.reference}"
    ref = record["arms"][ref_name]["yield_"]
    for name, arm in record["arms"].items():
        arm["rel_to_reference"] = arm["yield_"] / ref if ref else None
        arm.update(_spectral_error(E, spectra[name][0], spectra[ref_name][0]))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=1)
        np.savez_compressed(args.json.replace(".json", ".npz"), **spectra)
    print(json.dumps({k: v.get("rel_to_reference") for k, v in record["arms"].items()}))


# ---- campaign route parity (#1483) --------------------------------------------
def cmd_campaign(args):
    """Campaign fiber route versus the Q arm on one campaign transport (#1483).

    Lowers a HOPG Scene with ``CxrReflections(fiber=...)`` on this script's
    explicit photon axis, transports it once, then scores (a) the campaign
    line seam, (b) the full campaign spectrum phase and (c) the Q arm's
    per-family loop on the same segments. With ``--basal-b tensor`` the Q
    arm uses the model's projected basal B instead of catalog B33, so (a) and
    (c) differ only by summation order.
    """
    from pyrite import Scene
    from pyrite.beams import Beam, PointBunch, Spot
    from pyrite.campaign.lowering import lower_simulation
    from pyrite.campaign.resolution import resolve_simulation
    from pyrite.config import CxrReflections, FiberAverage, FixedSampling, Numerics, Physics
    from pyrite.detectors import Detector, EnergyBins
    from pyrite.geometry import Footprint, Slab
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.fiber_emission import fiber_domain

    E = np.arange(args.step, args.energy_max, args.step)
    average = FiberAverage(
        cutoff_invang=args.g_max, azimuth_count=args.reference, reflection_batch_size=args.batch
    )
    scene = Scene(
        Beam(args.E0, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", args.thick_nm * 10.0, tilt_deg=args.tilt, footprint=Footprint(5.0, 5.0)),
        Detector(observation_angle_deg=args.theta),
    )
    numerics = Numerics(
        sampling=FixedSampling(cxr_histories=args.electrons, brem_histories=args.electrons),
        cxr_reflections=CxrReflections(fiber=average),
        energy_bins=EnergyBins(cxr=E),
        seed=args.seed,
    )
    physics = Physics(bremsstrahlung_model="bremslib", radiative_model="uncoupled")
    case = lower_simulation(resolve_simulation(scene, physics=physics, numerics=numerics))
    t0 = time.time()
    tp = runner._transport_case(case, transport_core=args.transport_core)
    t_transport = time.time() - t0
    segs, n_hat, ne = tp["segs"], tp["n_hat"], int(tp["Ne_lines"])
    t0 = time.time()
    seam = np.asarray(
        runner._cxr_for_segments(
            runner._segments_on_device(segs),
            tp["E_grid"],
            case,
            n_hat,
            case.get("abs_layers"),
            tp.get("groove"),
            Ne=ne,
        )
    )
    t_seam = time.time() - t0
    t0 = time.time()
    full = runner._spectrum_case(case, tp)
    t_full = time.time() - t0
    domain = fiber_domain(case)
    lookup = domain._thermal_lookup()
    basal_b = lookup[(0, 0, 2)] if args.basal_b == "tensor" else CRYSTALS["hopg"]["B_ang2"]
    t0 = time.time()
    q_arm = np.zeros(E.size)
    composition = list(case["composition"])
    for fam in [] if args.no_q_arm else families("hopg", args.g_max):
        q_arm += family_spectrum(
            segs,
            E,
            "hopg",
            fam,
            n_hat,
            None,
            composition,
            basal_b,
            components=False,
            phis=list(_quadrature(args.reference)),
        )[0]
    t_q = time.time() - t0
    dE = np.gradient(E)
    record = dict(
        E0_keV=args.E0,
        thick_nm=args.thick_nm,
        tilt_deg=args.tilt,
        theta_deg=args.theta,
        electrons=ne,
        segments=int(np.asarray(segs["L_ang"]).size),
        seed=args.seed,
        g_max=args.g_max,
        rolls=args.reference,
        batch=args.batch,
        basal_b=basal_b,
        reflections=len(domain.hkls),
        step_eV=args.step,
        energy_max_eV=args.energy_max,
        case_fiber=case["cxr_fiber"],
        yield_seam=float(np.sum(seam * dE)),
        yield_q=float(np.sum(q_arm * dE)),
        yield_full_spec=float(np.sum(np.asarray(full["spec"]) * dE)),
        seam_vs_q=None if args.no_q_arm else _spectral_error(E, seam, q_arm),
        full_minus_seam_yield=float(np.sum((np.asarray(full["spec"]) - seam) * dE)),
        continuum=full.get("brem_lattice_share"),
        seconds=dict(transport=t_transport, seam=t_seam, full_spectrum=t_full, q_arm=t_q),
    )
    print(json.dumps({k: record[k] for k in ("yield_seam", "yield_q", "seam_vs_q", "seconds")}))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(record, handle, indent=1, default=str)
        np.savez_compressed(
            args.json.removesuffix(".json") + ".npz",
            E_eV=E,
            seam=seam,
            q_arm=q_arm,
            full_spec=np.asarray(full["spec"]),
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("structure")
    p.add_argument("crystal", choices=FIBER_CRYSTALS)
    p.add_argument("--g-max", type=float, default=8.0)
    p.add_argument("--json")
    p = sub.add_parser("kinematics")
    p.add_argument("--plot")
    p = sub.add_parser("single-flight")
    p.add_argument("--g-max", type=float, default=6.0)
    p.add_argument("--length", type=float, default=5000.0, help="flight length [A]")
    p.add_argument("--n-phi", type=int, default=8)
    p.add_argument("--step", type=float, default=2.0, help="grid step [eV]")
    p.add_argument("--json")
    p = sub.add_parser("frozen")
    p.add_argument("crystal", choices=FIBER_CRYSTALS)
    p.add_argument("E0", type=float, help="beam kinetic energy [keV]")
    p.add_argument("thick_nm", type=float)
    p.add_argument("tilt", type=float, help="target tilt [deg]")
    p.add_argument("--tilt-azim", type=float, default=180.0)
    p.add_argument("--theta", type=float, default=45.0, help="lab observation angle [deg]")
    p.add_argument("--electrons", type=int, default=400)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--g-max", type=float, default=6.0)
    p.add_argument("--g-cuts", type=float, nargs="*", default=[], help="nested reciprocal cutoffs")
    p.add_argument("--step", type=float, default=2.0)
    p.add_argument("--energy-max", type=float, help="explicit common photon band ceiling [eV]")
    p.add_argument("--samples", type=int, nargs="*", default=[8, 32, 128])
    p.add_argument(
        "--quadratures", type=int, nargs="*", default=[], help="sector quadrature ladder"
    )
    p.add_argument("--reference", type=int, default=32, help="sector quadrature nodes")
    p.add_argument(
        "--full-circle-reference", type=int, default=0, help="full-circle midpoint nodes"
    )
    p.add_argument("--b33-iso", action="store_true", help="also run full domain with scalar B33")
    p.add_argument(
        "--save-families", action="store_true", help="save reference family spectra in NPZ"
    )
    p.add_argument("--json")
    p = sub.add_parser("campaign")
    p.add_argument("E0", type=float, help="beam kinetic energy [keV]")
    p.add_argument("thick_nm", type=float)
    p.add_argument("tilt", type=float, help="target tilt [deg]")
    p.add_argument("--theta", type=float, default=45.0, help="lab observation angle [deg]")
    p.add_argument("--electrons", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--g-max", type=float, default=6.0)
    p.add_argument("--reference", type=int, default=32, help="roll nodes")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--step", type=float, default=2.0)
    p.add_argument("--energy-max", type=float, required=True, help="photon band ceiling [eV]")
    p.add_argument("--basal-b", choices=("tensor", "catalog"), default="tensor")
    p.add_argument("--transport-core", default="auto")
    p.add_argument("--no-q-arm", action="store_true", help="campaign cost only")
    p.add_argument("--json")
    args = parser.parse_args()
    {
        "frozen": cmd_frozen,
        "structure": cmd_structure,
        "kinematics": cmd_kinematics,
        "single-flight": cmd_single_flight,
        "campaign": cmd_campaign,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
