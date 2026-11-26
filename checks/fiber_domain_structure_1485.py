"""Turbostratic rods, mosaic with roll, and coherent domain phases for fiber CXR (#1485).

Research prototype for #1480 Phase II. It measures how three crystallite
structure effects change the ideal Bernal fiber average of
``checks/fiber_nonbasal_cxr_1480.py`` on the #1480 frozen scenes. It changes
no production default and no catalog entry.

Subcommands::

  profile                     stacking interference function: limits, Parseval
                              and the Bernal limit against the Bernal |S|^2
  single-flight               one straight flight, one (hk) rod: Bernal family
                              sum against rods at several stacking-fault
                              probabilities (normalization check)
  frozen CRYSTAL E0_KEV THICK_NM TILT_DEG [--theta DEG] [--arms ...] [--json OUT]
                              stacking, mosaic and coherent-domain arms on one
                              frozen local transport

Stacking model (Warren 1941 random layer lattice; Hendricks & Teller 1942
partially ordered layers). Layers keep the Bernal spacing ``d = c/2``. Each
interface is a turbostratic fault (random rotation about c and random in-plane
translation) with probability ``p``; otherwise the Bernal A/B alternation
continues. A fault removes every in-plane (hk != 0) correlation across it,
so the pair correlation of layers ``n`` apart is ``(1-p)^|n| C_n``, with
``C_even = (|F_A|^2 + |F_B|^2)/2`` and ``C_odd = Re(F_A F_B^*)``. The rod
intensity per layer at ``x = g_z d`` is

    I(x) = C_0 [1 + 2 S_even] + 2 Re(F_A F_B^*) S_odd,
    S_all = Re[z/(1-z)],  S_even = Re[z^2/(1-z^2)],  S_odd = S_all - S_even,
    z = (1-p) e^{ix}.

This is the Hendricks & Teller (1942) Section 4 correlated-layer sum, their
Eq. (14), with the per-interface factor propagated by complete induction as in
their Eqs. (15)-(19) and summed geometrically as in their Eq. (20). With one
layer kind (F_A = F_B) it reduces to their Section 2 closed form, Eq. (6),
``I = |V|^2 (1 - r^2)/(1 - 2 r cos x + r^2)`` with ``r = 1 - p``.
``p = 1`` is Warren's (1941) random layer lattice: (hk) layers scatter
incoherently (his Eq. (7)) along rods ``s - s0 = H + p3 b3`` (his Eq. (8)),
and (00l) stays crystalline (his Eq. (6)). ``I = C_0``; ``p -> 0`` collapses to the
Bernal points with ``(2/pi) int I dx = |S_Bernal|^2`` per point; the mean
over x is ``C_0`` for every ``p`` (Parseval). Basal (00l) rows are unchanged.

Rod scoring. A one-layer hexagonal cell (layer A of the catalog basis,
``c = d``) gives each rod node the exact layer amplitude ``F_A(g)`` with the
production form factors, recoil, polarization, absorption and PXR/CBS
couplings. One node at layer index ``l'`` stands for ``Delta l'`` of rod; its
intensity weight ``w = Delta l' I(x)/|F_A|^2`` is folded into the scalar
Debye--Waller argument, ``B -> B - 8 pi^2 ln(w)/g^2``, which multiplies both
couplings by ``sqrt(w)`` exactly. Node positions are jittered per electron
stratum, so the comb of nodes is an unbiased estimate of the continuous rod.

Mosaic arms compose production Gauss--Hermite tilt quadrature (or per-history
sampled tilts) with the roll, which production already applies after the roll
(``R_tilt R_roll g``).

Coherent-domain arms. The coherent route sums fields within one electron when
the inter-electron form factor is zero (finite footprint, long bunch). A track
is cut into crystallite cells (``L_a`` in plane, ``L_c`` along c) and each
(electron, cell) piece becomes its own emitter with its own roll: non-basal
fields of distinct crystallites never interfere (distinct g), and basal fields
lose phase across a cell boundary with random stacking registry.

Run:  pixi run python checks/fiber_domain_structure_1485.py profile
"""

from __future__ import annotations

import argparse
import json
import math
import time

import fiber_nonbasal_cxr_1480 as F1480
import numpy as np

from pyrite.materials.crystal import CRYSTALS, cromer_mann_f0, reciprocal_g_vector

LAYER_SUFFIX = "__layer1485"
THERMAL_T_K = 300.0


# ---- stacking interference ---------------------------------------------------
def layer_factors(crystal, h, k, g):
    """``(|F_A|^2, |F_B|^2, Re F_A F_B^*)`` of the two Bernal layers at |g| (f0 only)."""
    fa = fb = 0.0j
    for el, R in CRYSTALS[crystal]["basis"]:
        term = cromer_mann_f0(el, g) * np.exp(2j * np.pi * (h * R[0] + k * R[1]))
        if abs(R[2]) < 1e-9:
            fa += term
        else:
            fb += term
    return abs(fa) ** 2, abs(fb) ** 2, float((fa * np.conj(fb)).real)


def stacking_intensity(x, p, ca2, cb2, re_ab):
    """Rod intensity per layer at ``x = g_z d`` for fault probability ``p``."""
    if not 0.0 < p <= 1.0:
        raise ValueError("fault probability must lie in (0, 1]")
    z = (1.0 - p) * np.exp(1j * np.asarray(x, dtype=float))
    s_all = (z / (1.0 - z)).real
    s_even = (z * z / (1.0 - z * z)).real
    c0 = 0.5 * (ca2 + cb2)
    return c0 * (1.0 + 2.0 * s_even) + 2.0 * re_ab * (s_all - s_even)


def cmd_profile(args):
    rows = []
    for crystal in ("hopg", "hbn"):
        for h, k in ((1, 0), (1, 1), (2, 0)):
            g = float(
                np.linalg.norm(reciprocal_g_vector((h, k, 0), CRYSTALS[crystal]["lattice"])[0])
            )
            ca2, cb2, re_ab = layer_factors(crystal, h, k, g)
            x = (np.arange(200000) + 0.5) * 2 * np.pi / 200000
            for p in (1.0, 0.5, 0.1, 0.02, 0.002):
                prof = stacking_intensity(x, p, ca2, cb2, re_ab)
                mean = prof.mean() / (0.5 * (ca2 + cb2))
                assert abs(mean - 1) < 1e-6, (crystal, h, k, p, mean)
                assert prof.min() >= -1e-9 * prof.max()
                even = (
                    (2 / np.pi) * prof[(x < np.pi / 2) | (x > 3 * np.pi / 2)].sum() * (x[1] - x[0])
                )
                odd = (
                    (2 / np.pi) * prof[(x > np.pi / 2) & (x < 3 * np.pi / 2)].sum() * (x[1] - x[0])
                )
                # Bernal |S|^2 with the same f0 (l even -> x = 0, l odd -> x = pi)
                s_even = ca2 + cb2 + 2 * re_ab
                s_odd = ca2 + cb2 - 2 * re_ab
                rows.append(
                    dict(
                        crystal=crystal,
                        hk=(h, k),
                        p=p,
                        mean=mean,
                        even=even,
                        odd=odd,
                        bernal_even=s_even,
                        bernal_odd=s_odd,
                        peak_over_c0=prof.max() / (0.5 * (ca2 + cb2)),
                    )
                )
                if p == 0.002:
                    assert abs(even - s_even) < 0.01 * (s_even + s_odd), (
                        crystal,
                        h,
                        k,
                        even,
                        s_even,
                    )
                    assert abs(odd - s_odd) < 0.01 * (s_even + s_odd), (crystal, h, k, odd, s_odd)
                if p == 1.0:
                    assert np.allclose(prof, 0.5 * (ca2 + cb2))
            # Bernal |S|^2 from the catalog basis agrees with the layer split.
            for ell in (2, 1):
                from pyrite.materials.crystal import structure_factor

                S, _ = structure_factor(crystal, (h, k, ell), 1000.0, B_ang2=0.0, use_henke=False)
                gl = float(
                    np.linalg.norm(
                        reciprocal_g_vector((h, k, ell), CRYSTALS[crystal]["lattice"])[0]
                    )
                )
                fa2, fb2, re2 = layer_factors(crystal, h, k, gl)
                sgn = 1 if ell % 2 == 0 else -1
                assert abs(abs(S) ** 2 - (fa2 + fb2 + 2 * sgn * re2)) < 1e-9 * abs(S) ** 2 + 1e-12
    print(
        f"{'crystal':7} {'hk':6} {'p':>6} {'mean/C0':>8} {'peak/C0':>8} {'even':>8} {'Bernal':>8} {'odd':>8} {'Bernal':>8}"
    )
    for r in rows:
        print(
            f"{r['crystal']:7} {str(r['hk']):6} {r['p']:6.3f} {r['mean']:8.5f} {r['peak_over_c0']:8.2f}"
            f" {r['even']:8.3f} {r['bernal_even']:8.3f} {r['odd']:8.3f} {r['bernal_odd']:8.3f}"
        )
    print("profile: Parseval, p=1 rod and p->0 Bernal limits hold")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


# ---- reflection domains --------------------------------------------------------
def layer_crystal(crystal):
    """Register the one-layer cell of ``crystal`` (layer A, ``c = d``) and return its key."""
    key = crystal + LAYER_SUFFIX
    if key not in CRYSTALS:
        info = CRYSTALS[crystal]
        lat = dict(info["lattice"])
        lat["c"] = info["lattice"]["c"] / 2.0
        basis = tuple(
            (el, np.array([R[0], R[1], 0.0])) for el, R in info["basis"] if abs(R[2]) < 1e-9
        )
        if 2 * len(basis) != len(info["basis"]):
            raise ValueError("expected a two-layer Bernal basis")
        CRYSTALS[key] = dict(
            info, lattice=lat, basis=basis, V_cell=info["V_cell"] / 2.0, texture=None
        )
    return key


def bernal_domain(crystal, g_max):
    """Full Bernal domain split into basal and non-basal rows with per-row B."""
    basal, other, b = [], [], {}
    for fam in F1480.families(crystal, g_max):
        b_fam = fam["b_eff"]
        for hkl in fam["members"]:
            b[tuple(hkl)] = b_fam
            (basal if fam["g_perp"] == 0.0 else other).append(tuple(hkl))
    return basal, other, b


def thermal_b(crystal, hkl_bernal, B33):
    """Per-reflection B: projected tensor for HOPG, catalog B33 otherwise (as in #1480)."""
    return F1480.b_effective(crystal, hkl_bernal, B33)


def rod_domain(crystal, g_max, dl, jitter, p, B33):
    """Rod nodes ``(h, k, l')`` of the one-layer cell, and B with the rod weight folded in.

    Nodes sit at ``l' = (j + jitter) dl``; node weight ``dl I(x)/|F_A|^2``.
    """
    key = layer_crystal(crystal)
    lat = CRYSTALS[key]["lattice"]
    n_ab = int(g_max * lat["a"] / (2 * math.pi)) + 2
    hkls, b = [], {}
    for h in range(-n_ab, n_ab + 1):
        for k in range(-n_ab, n_ab + 1):
            if (h, k) == (0, 0):
                continue
            g_perp = float(np.linalg.norm(reciprocal_g_vector((h, k, 0), lat)[0]))
            if g_perp > g_max:
                continue
            gz_max = math.sqrt(g_max**2 - g_perp**2)
            l_max = gz_max * lat["c"] / (2 * math.pi)
            j0, j1 = math.floor(-l_max / dl - jitter) - 1, math.ceil(l_max / dl - jitter) + 1
            for j in range(j0, j1 + 1):
                lp = (j + jitter) * dl
                hkl = (float(h), float(k), float(lp))
                g = float(np.linalg.norm(reciprocal_g_vector(hkl, lat)[0]))
                if g > g_max:
                    continue
                ca2, cb2, re_ab = layer_factors(crystal, h, k, g)
                w = dl * stacking_intensity(2 * math.pi * lp, p, ca2, cb2, re_ab) / ca2
                if w <= 1e-12:
                    continue
                b_th = thermal_b(crystal, (h, k, 2.0 * lp), B33)
                hkls.append(hkl)
                b[hkl] = b_th - 8 * math.pi**2 * math.log(w) / g**2
    return key, hkls, b


# ---- scoring -------------------------------------------------------------------
def _int_key(hkl):
    """The key production's coupling tables pass to a callable B: int-cast Friedel mate."""
    return tuple(int(-x) for x in hkl)


def _collision_free_groups(hkls):
    """Split rows so no two rows of a group share an int-cast key.

    ``reflection_coupling_tables`` looks a callable B up by ``int`` of each
    (negated) index; fractional rod nodes ``j + u`` and ``j - 1 + u`` near
    ``l' = 0`` collide there, so they are scored in separate calls.
    """
    groups, seen = [], []
    for hkl in hkls:
        key = _int_key(hkl)
        for grp, keys in zip(groups, seen, strict=True):
            if key not in keys:
                grp.append(hkl)
                keys.add(key)
                break
        else:
            groups.append([hkl])
            seen.append({key})
    return groups


def score(segs, E, crystal, hkls, b, n_hat, phi, components=True, **kw):
    """Production incoherent/coherent CXR of ``hkls`` with per-row B from ``b``."""
    from pyrite.montecarlo import mc_cxr_spectrum

    composition = kw.pop("composition")
    quadrature = kw.pop("line_quadrature", "bin-mean")
    total = np.zeros((3 if components else 1, E.size))
    for group in _collision_free_groups(list(hkls)):
        lookup = {tuple(h): b[tuple(h)] for h in group}
        lookup.update({_int_key(h): b[tuple(h)] for h in group})
        out = mc_cxr_spectrum(
            segs,
            E,
            crystal,
            group,
            n_hat=n_hat,
            B_ang2=lookup.__getitem__,
            composition=composition,
            surface_hkl=(0, 0, 1),
            azimuth_rad=float(phi),
            components=components,
            line_quadrature=quadrature,
            use_henke=True,
            **kw,
        )
        total += np.stack(out) if components else np.asarray(out)[None]
    return total


def strata_labels(n, k, rng):
    """Equal-count random assignment of ``n`` emitters to ``k`` strata."""
    label = rng.permutation(k)[np.arange(n) % k]
    rng.shuffle(label)
    return label


def stratified(segs, ne_ref, labels, fn):
    """Sum of per-stratum scores, each weighted by its emitter share of ``ne_ref``."""
    total = None
    for j in range(int(labels.max()) + 1):
        mask = labels == j
        if not mask.any():
            continue
        sub = F1480._electron_subset(segs, mask)
        if not np.asarray(sub["L_ang"]).size:
            continue
        s = fn(sub, j) * (mask.sum() / ne_ref)
        total = s if total is None else total + s
    return total


def yields(E, spec):
    dE = np.gradient(E)
    return float(np.sum(spec * dE))


def smooth(spec, width):
    if width <= 1:
        return spec
    kern = np.ones(width) / width
    return np.convolve(spec, kern, mode="same")


def compare(E, cand, ref, width):
    """Relative yield change and raw / smoothed L1 of ``cand`` against ``ref``."""
    y_ref = yields(E, ref)
    raw = F1480._spectral_error(E, cand, ref)["mass_l1"]
    sm = F1480._spectral_error(E, smooth(cand, width), smooth(ref, width))["mass_l1"]
    return dict(rel=yields(E, cand) / y_ref - 1 if y_ref else None, l1=raw, l1_smooth=sm)


# ---- coherent domains ----------------------------------------------------------
def incoherent_between_electrons(segs, seed):
    """Zero inter-electron form factor: random arrival times, finite footprint, long bunch."""
    out = dict(segs)
    rng = np.random.default_rng(seed)
    t0 = rng.normal(0.0, 1e7, int(out["Ne"]))
    out["initial_t0_ang"] = t0
    out["t0_ang"] = t0[np.asarray(out["elec_id"])]
    out["crystal_width_ang"] = out["crystal_height_ang"] = 1e10
    return out


def split_domains(segs, L_a, L_c, seed):
    """Relabel each (electron, crystallite cell) piece of a track as its own emitter."""
    rng = np.random.default_rng(seed)
    r = np.asarray(segs["r_mid"], dtype=float)
    e = np.asarray(segs["elec_id"])
    cell = np.floor(r / np.array([L_a, L_a, L_c]) + rng.random(3)).astype(np.int64)
    uniq, inv = np.unique(np.column_stack([e, cell]), axis=0, return_inverse=True)
    inv = inv.ravel()
    out = dict(segs)
    out["elec_id"] = inv
    out["electron_id"] = inv.copy()
    for key, val in segs.items():
        if key.startswith("initial_"):
            out[key] = np.asarray(val)[uniq[:, 0]]
    out["Ne"] = int(len(uniq))
    out["t0_ang"] = np.asarray(segs["t0_ang"])
    return out, uniq[:, 0]


# ---- single flight -------------------------------------------------------------
def cmd_single_flight(args):
    from pyrite.materials.catalog import load_material_catalog

    out = []
    for crystal in ("hopg", "hbn"):
        cat = load_material_catalog().crystal(crystal)
        n_hat = F1480._unit(math.radians(args.theta))
        seg = F1480.straight_flight(args.E0, args.length, args.tilt)
        E = F1480._grid_for(args.E0, args.g_max, n_hat, args.tilt, args.step)
        kw = dict(composition=cat.composition)
        h, k = args.hk
        _, other, b = bernal_domain(crystal, args.g_max)
        rows = [hkl for hkl in other if (hkl[0], hkl[1]) == (h, k)]
        # mean over the sixfold roll sector, as in #1480
        phis = F1480._quadrature(args.n_phi)
        bern = sum(score(seg, E, crystal, rows, b, n_hat, phi, **kw) for phi in phis) / len(phis)
        rec = dict(crystal=crystal, hk=(h, k), bernal=yields(E, bern[0]), rods={})
        for p, dl in zip(args.p, args.dl, strict=True):
            acc = 0
            for i, phi in enumerate(phis):
                key, hk_all, brod = rod_domain(
                    crystal, args.g_max, dl, (i + 0.5) / len(phis), p, cat.B_ang2
                )
                sel = [x for x in hk_all if (x[0], x[1]) == (h, k)]
                acc = acc + score(seg, E, key, sel, brod, n_hat, phi, **kw)
            acc = acc / len(phis)
            rec["rods"][f"p{p:g}_dl{dl:g}"] = dict(
                yield_=yields(E, acc[0]),
                **compare(E, acc[0], bern[0], args.smooth),
            )
        out.append(rec)
        print(
            crystal,
            (h, k),
            f"Bernal {rec['bernal']:.4e}",
            "  ".join(
                f"{n}: {v['rel']:+.3f} (L1s {v['l1_smooth']:.2f})" for n, v in rec["rods"].items()
            ),
        )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


# ---- frozen transport ----------------------------------------------------------
def cmd_frozen(args):
    from pyrite.materials.catalog import load_material_catalog
    from pyrite.montecarlo.geometry import detector_directions
    from pyrite.montecarlo.transport import _simulate_trajectories

    cat = load_material_catalog().crystal(args.crystal)
    kw = dict(composition=cat.composition)
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
    n_hat = detector_directions(math.radians(args.theta), tp, ta, n_side=1, domega_sr=1.0)[0][0]
    ne = int(segs["Ne"])
    u0 = np.asarray(segs["initial_v_hat"], dtype=float).mean(axis=0)
    u0 /= np.linalg.norm(u0)
    beta0 = float(F1480.beta_from_Ee(args.E0 * 1e3))
    top = 1.3 * F1480.HBARC_EV_ANG * beta0 * args.g_max / (1 - beta0 * float(u0 @ n_hat))
    stop = min(top, 0.95 * args.E0 * 1e3) if args.energy_max is None else args.energy_max
    E = np.arange(args.step, stop, args.step)
    basal, other, b_bern = bernal_domain(args.crystal, args.g_max)
    rng = np.random.default_rng(args.seed + 1485)
    K = args.strata
    labels = strata_labels(ne, K, rng)
    phis = 2 * math.pi * (rng.permutation(K) + rng.random(K)) / K
    jitter = (rng.permutation(K) + rng.random(K)) / K
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
        step_eV=args.step,
        strata=K,
        smooth_bins=args.smooth,
        transport_s=t_transport,
        energy_band_eV=[float(E[0]), float(E[-1])],
        thermal_model="packaged uniaxial ADP (300 K), catalog basal B33"
        if args.crystal in F1480.TENSOR_CRYSTALS
        else "provisional scalar B33",
        arms={},
    )
    spectra = {"E_eV": E}

    def keep(name, spec, started, **extra):
        rec = dict(yield_=yields(E, spec[0]), seconds=time.time() - started, **extra)
        if spec.shape[0] == 3:
            rec.update(pxr=yields(E, spec[1]), cbs=yields(E, spec[2]))
        record["arms"][name] = rec
        spectra[name] = spec
        print(f"  {name:>22}: yield {rec['yield_']:.4e}  {rec['seconds']:.0f} s", flush=True)

    def bernal_arm(name, **mk):
        started = time.time()
        history = mk.get("mosaic_sampling") == "history"
        seed0 = mk.pop("mosaic_seed", 0)

        def per_stratum(rows, roll):
            def fn(sub, j):
                extra = dict(mosaic_seed=seed0 + j) if history else {}
                phi = phis[j] if roll else 0.0
                return score(sub, E, args.crystal, rows, b_bern, n_hat, phi, **extra, **mk, **kw)

            return fn

        if history:
            # Sampled tilts are keyed by history; strata reuse ids, so each
            # stratum gets its own seed.
            spec_b = stratified(segs, ne, labels, per_stratum(basal, False))
        else:
            spec_b = score(segs, E, args.crystal, basal, b_bern, n_hat, 0.0, **mk, **kw)
        fo = per_stratum(other, True)
        spec_o = stratified(segs, ne, labels, fo)
        keep(
            name,
            spec_b + spec_o,
            started,
            basal=yields(E, spec_b[0]),
            nonbasal=yields(E, spec_o[0]),
        )
        spectra[name + "_basal"] = spec_b
        return spec_b, spec_o

    print(
        f"{args.crystal} {args.E0} keV {args.thick_nm} nm tilt {args.tilt} theta {args.theta}: "
        f"{record['segments']} segments, {ne} e, {len(basal)}+{len(other)} Bernal rows",
        flush=True,
    )
    arms = set(args.arms)
    bern_b, bern_o = bernal_arm("bernal")
    ref = bern_b + bern_o
    if "stacking" in arms:
        for p, dl in zip(args.p, args.dl, strict=True):
            started = time.time()
            n_nodes = []

            def fr(sub, j, p=p, dl=dl, n_nodes=n_nodes):
                key, hk, brod = rod_domain(args.crystal, args.g_max, dl, jitter[j], p, cat.B_ang2)
                n_nodes.append(len(hk))
                return score(sub, E, key, hk, brod, n_hat, phis[j], **kw)

            spec_o = stratified(segs, ne, labels, fr)
            name = f"rod_p{p:g}"
            keep(
                name,
                bern_b + spec_o,
                started,
                nonbasal=yields(E, spec_o[0]),
                dl=dl,
                mean_rod_nodes=float(np.mean(n_nodes)),
                **compare(E, (bern_b + spec_o)[0], ref[0], args.smooth),
            )
    if "mosaic" in arms:
        for fwhm in args.mosaic_fwhm:
            for nodes in args.mosaic_nodes:
                started = time.time()
                spec_b, spec_o = bernal_arm(
                    f"mosaic{fwhm:g}_n{nodes}",
                    mosaic_fwhm_rad=math.radians(fwhm),
                    mosaic_nodes=nodes,
                )
                record["arms"][f"mosaic{fwhm:g}_n{nodes}"].update(
                    compare(E, (spec_b + spec_o)[0], ref[0], args.smooth)
                )
            spec_b, spec_o = bernal_arm(
                f"mosaic{fwhm:g}_hist",
                mosaic_fwhm_rad=math.radians(fwhm),
                mosaic_nodes=1,
                mosaic_sampling="history",
                mosaic_seed=args.seed * 1000,
            )
            record["arms"][f"mosaic{fwhm:g}_hist"].update(
                compare(E, (spec_b + spec_o)[0], ref[0], args.smooth)
            )
    if "coherent" in arms:
        run_coherent(args, segs, E, basal, other, b_bern, n_hat, kw, record, spectra, keep)
    for name, arm in record["arms"].items():
        if "rel" not in arm and not name.startswith("coh"):
            arm.update(compare(E, spectra[name][0], ref[0], args.smooth))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=1)
        np.savez_compressed(args.json.replace(".json", ".npz"), **spectra)
    print(json.dumps({k: round(v.get("rel") or 0.0, 5) for k, v in record["arms"].items()}))


def run_coherent(args, segs, E, basal, other, b_bern, n_hat, kw, record, spectra, keep):
    """Coherent (node quadrature) arms on the first ``--coh-electrons`` histories."""
    nc = min(args.coh_electrons, int(segs["Ne"]))
    sub = F1480._electron_subset(segs, np.arange(int(segs["Ne"])) < nc)
    sub = incoherent_between_electrons(sub, args.seed + 7)
    rng = np.random.default_rng(args.seed + 2485)
    opts = dict(line_quadrature="node", components=False, **kw)
    coh = dict(coherent=True, longitudinal_rms_fs=1e3)

    # One roll per parent electron, shared by every arm and inherited by the
    # crystallite pieces of its track: the arms then differ only in phase
    # breaks. Independent rolls per piece change the mean by nothing (the
    # mean is linear in the orientation distribution), only its noise.
    K = min(args.strata, nc)
    parent_labels = strata_labels(nc, K, rng)
    phis = 2 * math.pi * (rng.permutation(K) + rng.random(K)) / K

    def arm(name, s, coherent_kw, parent=None):
        started = time.time()
        n_emit = int(s["Ne"])
        labels = parent_labels if parent is None else parent_labels[parent]
        spec_b = score(s, E, args.crystal, basal, b_bern, n_hat, 0.0, **coherent_kw, **opts) * (
            n_emit / nc
        )
        spec_o = stratified(
            s,
            nc,
            labels,
            lambda q, j: score(
                q, E, args.crystal, other, b_bern, n_hat, phis[j], **coherent_kw, **opts
            ),
        )
        keep(
            name,
            spec_b + spec_o,
            started,
            basal=yields(E, spec_b[0]),
            nonbasal=yields(E, spec_o[0]),
            emitters=n_emit,
            emitters_per_electron=n_emit / nc,
        )
        spectra[name + "_basal"] = spec_b
        return spec_b, spec_o

    inc_b, inc_o = arm("coh_ref_incoherent", sub, {})
    ce_b, ce_o = arm("coh_electron", sub, coh)
    results = {}
    for spec in args.domains:
        L_a, L_c = (float(v) * 10.0 for v in spec.split(":"))  # nm -> Angstrom
        s, parent = split_domains(sub, L_a, L_c, args.seed + 11)
        s = incoherent_between_electrons(s, args.seed + 13)
        name = f"coh_dom_{spec}"
        d_b, d_o = arm(name, s, coh, parent)
        results[name] = (d_b, d_o)
    w = args.smooth
    for name, (d_b, d_o) in results.items():
        rec = record["arms"][name]
        for part, d, c, i in (
            ("total", d_b + d_o, ce_b + ce_o, inc_b + inc_o),
            ("basal", d_b, ce_b, inc_b),
            ("nonbasal", d_o, ce_o, inc_o),
        ):
            y_c, y_i, y_d = yields(E, c[0]), yields(E, i[0]), yields(E, d[0])
            rec[part] = dict(
                coherent_excess=y_c / y_i - 1 if y_i else None,
                retained=(y_d - y_i) / (y_c - y_i) if y_c != y_i else None,
                vs_coherent=compare(E, d[0], c[0], w),
                coherent_vs_incoherent=compare(E, c[0], i[0], w),
                vs_incoherent=compare(E, d[0], i[0], w),
            )
        print(
            f"  {name}: coherent excess / retained: "
            + "; ".join(
                f"{part} {rec[part]['coherent_excess']:+.4f} / {rec[part]['retained']:.3f}"
                for part in ("total", "basal", "nonbasal")
            )
            + f"; L1s vs coherent {rec['total']['vs_coherent']['l1_smooth']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--json")
    p = sub.add_parser("single-flight")
    p.add_argument("--E0", type=float, default=30.0)
    p.add_argument("--tilt", type=float, default=15.0)
    p.add_argument("--theta", type=float, default=45.0)
    p.add_argument("--length", type=float, default=5000.0, help="flight length [A]")
    p.add_argument("--hk", type=int, nargs=2, default=(1, 0))
    p.add_argument("--g-max", type=float, default=6.0)
    p.add_argument("--n-phi", type=int, default=8)
    p.add_argument("--step", type=float, default=2.0)
    p.add_argument("--smooth", type=int, default=25, help="boxcar bins for smoothed L1")
    p.add_argument("--p", type=float, nargs="+", default=[1.0, 0.3, 0.05])
    p.add_argument("--dl", type=float, nargs="+", default=[1 / 16, 1 / 64, 1 / 512])
    p.add_argument("--json")
    p = sub.add_parser("frozen")
    p.add_argument("crystal", choices=F1480.FIBER_CRYSTALS)
    p.add_argument("E0", type=float, help="beam kinetic energy [keV]")
    p.add_argument("thick_nm", type=float)
    p.add_argument("tilt", type=float, help="target tilt [deg]")
    p.add_argument("--tilt-azim", type=float, default=180.0)
    p.add_argument("--theta", type=float, default=45.0, help="lab observation angle [deg]")
    p.add_argument("--electrons", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--g-max", type=float, default=6.0)
    p.add_argument("--step", type=float, default=2.0)
    p.add_argument("--energy-max", type=float)
    p.add_argument("--strata", type=int, default=32, help="roll strata (one roll per emitter)")
    p.add_argument("--smooth", type=int, default=25, help="boxcar bins for smoothed L1")
    p.add_argument(
        "--arms",
        nargs="+",
        default=["stacking", "mosaic", "coherent"],
        choices=["stacking", "mosaic", "coherent"],
    )
    p.add_argument(
        "--p", type=float, nargs="*", default=[1.0, 0.5], help="stacking fault probabilities"
    )
    p.add_argument(
        "--dl", type=float, nargs="*", default=[1 / 16, 1 / 32], help="rod node spacing per p"
    )
    p.add_argument("--mosaic-fwhm", type=float, nargs="*", default=[0.4, 0.8, 3.5], help="deg")
    p.add_argument("--mosaic-nodes", type=int, nargs="*", default=[3, 5])
    p.add_argument("--coh-electrons", type=int, default=100)
    p.add_argument(
        "--domains",
        nargs="*",
        default=["1000:30", "1000:100", "10000:300"],
        help="L_a:L_c crystallite cells [nm]",
    )
    p.add_argument("--json")
    args = parser.parse_args()
    {"profile": cmd_profile, "single-flight": cmd_single_flight, "frozen": cmd_frozen}[args.cmd](
        args
    )


if __name__ == "__main__":
    main()
