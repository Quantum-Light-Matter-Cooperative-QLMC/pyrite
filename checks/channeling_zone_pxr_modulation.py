"""Channeled-state modulation of off-own-row PXR on planar zone rows (#1148).

Native line scoring treats a planar zone row as a free plane-wave electron at
the beam direction. This check evaluates the first-order (kinematic, flat
``chi_g``) PXR amplitude of each off-own-row reflection ``g`` between the zone
solve's tracked transverse Bloch states, over the depth-averaged retained
density of the admitted ``channeling_demo`` zone (Si(1-10), 17 MeV, 2 um), and
compares its band-integrated yield at a finite detector with the plane-wave
reference evaluated in the same model.

Amplitude for photon ``(E, n)``, initial state ``i`` (fibre ``kappa``) and
final state ``f`` (fibre ``kappa - Q_x``), ``Q = k + g``::

    a_fi(E) = chi_g [alpha_s(E) A_fi + b_s(E) B_fi] K_fi(E)
    A = <f|exp(-iQx)|i>,   B = <f|{v_x, exp(-iQx)}/2|i> / c
    alpha_s = beta v.F_s,  b_s = F_s.x,  F_s = Q (g.e_s) - k^2 e_s
    K_fi(E) = (1/det) int_0^L dz exp(i Delta z) [1 - exp(i q s (L - z))]
    Delta = D (E - E_fi) / (hbar c beta),  E_fi = E0 + (eps_i - eps_f) / D
    det = g^2 + 2 k.g,  q = -det / (2 (k+g).s_hat),  s = s_hat.v

``[1 - exp(iqs(L-z))]`` is the finite-slab kinematic (Laue) photon mode with
zero diffracted amplitude at the exit face. Its first term is PXR; its second
is the free diffracted wave, i.e. forward channeling radiation Bragg-diffracted
into ``n`` (DCR), which removes the kinematic ``1/det`` pole. With
``eps_i = eps_f`` (no transverse transition) the DCR term has no phase match
(``omega/v > k_h.v``), so the plane-wave reference is recovered. The yield is
``d2N/dE dOmega ∝ omega |a|^2`` (as ``_cxr_weight_core``) times the uniform-
emission Beer-Lambert escape mean. The plane-wave reference is the same with
``A = 1``, ``B = 0``, frozen at the free root ``E0`` (exact recoil).

Reported per detector centre (square, face-on, solid angle and distance as
the #1148 JUNGFRAU arms, tensor Gauss order ``--order``):

- ``ratio_estimate``: band-integrated modulated/plane-wave yield of all kept
  reflections (``frozen_full - frozen_diag`` adds the coherence correction);
- ``hard_ratio_estimate``: same for reflections with ``E0 >= 2 keV``;
- per-reflection ratios, Bragg-pole offsets and windowed weight leakage.

With ``--production`` the check also scores the production zone-row source
(:func:`pyrite.montecarlo.runner.planar_zone_pxr.case_zone_pxr_spectrum`, #1166)
on the same rays and reflections. The reference per reflection is the native
zone-row band (``_cxr_for_segments`` on zone rows, bin-mean) times this model's
``estimate / native`` ratio, so normalization cancels. Reported per centre:

- ``production_over_reference_zone``: production zone-row band (the source for
  source-owned reflections, native for the rest) over the reference;
- ``production_over_reference_total``: the same with the native rows outside
  the zone added to both, i.e. the reflections' whole ``spec`` share.

The production source adds CBS, ``chi_0`` and the two-beam coupling that this
model omits; native's CBS carries over through the ratio.

Assumptions: kinematic two-beam diffraction (2 um << extinction length),
flat ``chi_g`` across the +-400 eV window, refraction and ``chi_0`` omitted,
stationary depth-averaged density (finite state lifetimes omitted), diagonal
populations in the frequency-dependent term, CBS distorted-wave resolvents
not included. Report only: exit 0 when every number is finite, 1 otherwise.

    uv run python checks/channeling_zone_pxr_modulation.py --order 3 \\
        --centre "(-1 -1 3) mirror" --json scratch/pxr_modulation.json

Validation: cr-planar-zone-pxr-modulation
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np

HBARC = 1973.269804
ME = 510998.95
WINDOW_EV, STEP_EV, HARD_EV = 400.0, 0.1, 2000.0
# Transitions with |eps_i - eps_f| below this are elastic (no transverse transition):
# their pre-crystal path is coherent with the in-crystal one (Born form, no pole).
ELASTIC_EV = 0.15
POLAR_DEG = (30.0, 37.5, 42.807296, 44.478804, 45.0)


def _capture_density(case):
    """Depth-averaged retained density of the live zone solve (8 electrons)."""
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.channeling.density import DensityMatrixSolver

    captured, solver = [], []
    original = DensityMatrixSolver.band_populations

    def hook(self, rho):
        captured.append(
            (np.asarray(rho).copy(), self.basis.energies.copy(), self.basis.vectors.copy())
        )
        solver[:] = [self, self.basis.gamma]
        return original(self, rho)

    DensityMatrixSolver.band_populations = hook
    try:
        runner._transport_case(
            {**case, "Ne": 8, "Ne_brem": 8, "line_quadrature": "bin-mean"},
            transport_core="per-electron",
        )
    finally:
        DensityMatrixSolver.band_populations = original
    rho = np.array([c[0] for c in captured])
    energies = np.array([c[1] for c in captured])
    if np.ptp(energies, axis=0).max() > 0:
        raise SystemExit("zone eigenbasis changes with depth; depth averaging invalid")
    weights = np.ones(len(rho))
    weights[[0, -1]] = 0.5
    weights /= weights.sum()
    s, gamma = solver
    return dict(
        rho=np.einsum("z,zrij->rij", weights, rho),
        energies=energies[0],
        vectors=captured[0][2],
        orders=np.asarray(s.orders),
        kappa=np.asarray(s.kappa),
        period=float(s.profile.period_ang),
        gamma=float(gamma),
        depths=len(rho),
    )


class Model:
    """Plane-wave and tracked-state PXR yields of one zone at one detector ray."""

    def __init__(self, case, zone):
        from pyrite.materials.crystal import CRYSTALS, reciprocal_g_vector
        from pyrite.montecarlo.channeling.axial_case import _rotation
        from pyrite.montecarlo.geometry import tilted_geometry
        from pyrite.montecarlo.runner.planar_zone_lines import is_own_row

        self.case = case
        self.R = _rotation(case)
        self.lattice = CRYSTALS[case["crystal"]]["lattice"]
        beam, _ = tilted_geometry(
            0.0, np.deg2rad(case["tilt_deg"]), np.deg2rad(case["tilt_azim_deg"])
        )
        self.beam = np.asarray(beam, float)
        plane, _ = reciprocal_g_vector(case["channeling"]["plane_hkl"], self.lattice)
        self.xhat = self.R @ (np.asarray(plane, float) / np.linalg.norm(plane))
        surface, _ = reciprocal_g_vector(case["surface_hkl"], self.lattice)
        self.shat = self.R @ (np.asarray(surface, float) / np.linalg.norm(surface))
        if abs(self.beam @ self.xhat) > 1e-12:
            raise SystemExit("beam not in the plane: plane-wave reference would double count")
        self.L = float(case["thickness_ang"]) / abs(self.beam @ self.shat)
        self.svel = float(self.shat @ self.beam)
        self.zone = zone
        self.gamma = zone["gamma"]
        self.beta = np.sqrt(1 - 1 / self.gamma**2)
        self.G = 2 * np.pi / zone["period"]
        self.C = HBARC**2 / (2 * self.gamma * ME)
        self.cv = HBARC / (self.gamma * ME)
        self.trace = float(np.einsum("rii->", zone["rho"]).real)
        if len(case["composition"]) != 1:
            raise SystemExit("escape factor implemented for an elemental crystal only")
        self.element, self.density = case["composition"][0]
        self.pots = []
        for r, kappa in enumerate(zone["kappa"]):
            pw = kappa + zone["orders"] * self.G
            vec = zone["vectors"][r]
            self.pots.append((vec * zone["energies"][r]) @ vec.conj().T - np.diag(self.C * pw**2))
        self.reflections = []
        for hkl in [tuple(int(v) for v in h) for h in case["hkl_list"]]:
            if is_own_row(hkl, case["channeling"]["plane_hkl"]):
                continue
            g0, _ = reciprocal_g_vector(hkl, self.lattice)
            self.reflections.append((hkl, self.R @ np.asarray(g0, float)))

    def field(self, E, n, g):
        E = np.atleast_1d(E)
        k = (E / HBARC)[:, None] * n[None, :]
        Q = k + g[None, :]
        det = g @ g + 2 * k @ g
        q = -det / (2 * (Q @ self.shat))
        es = np.cross(n, g)
        es /= np.linalg.norm(es)
        ep = np.cross(es, n)
        pols = []
        for e in (es, ep):
            F = Q * (g @ e) - ((E / HBARC) ** 2)[:, None] * e[None, :]
            pols.append((self.beta * F @ self.beam, F @ self.xhat))
        return pols, det, q

    def phi(self, x):
        """``int_0^L exp(i x z) dz``."""
        return self.L * np.exp(0.5j * x * self.L) * np.sinc(x * self.L / (2 * np.pi))

    def native(self, n, g, hkl):
        """Plane-wave yield (model units) and kinematics, or None without a root.

        Validation: cr-planar-zone-pxr-modulation
        """
        from pyrite.materials.crystal import absorption_length_ang, chi_g
        from pyrite.materials.recoil import recoil_resonance_eV

        if self.beam @ g <= 0:  # only +g with v.g > 0 radiates (no +-g double count)
            return None
        E0 = float(recoil_resonance_eV(float(self.case["E0_keV"]) * 1e3, self.beam, g, n))
        if not (np.isfinite(E0) and 50.0 <= E0 <= 966203.0) or n @ self.shat <= 0:
            return None
        D = 1 - self.beta * self.beam @ n
        chi2 = abs(chi_g(self.case["crystal"], hkl, E0, self.case["B_ang2"])) ** 2
        a_abs = float(self.case["thickness_ang"]) / (
            absorption_length_ang(self.element, E0, self.density) * (n @ self.shat)
        )
        chi2 *= -np.expm1(-a_abs) / a_abs
        pols, det0, _ = self.field(E0, n, g)
        scale = chi2 * 2 * np.pi * self.L * HBARC * self.beta / D * E0 / det0[0] ** 2
        native = scale * sum(a[0] ** 2 for a, _ in pols) * self.trace
        pole = float(-HBARC * (g @ g) / (2 * n @ g) - E0)
        return dict(E0=E0, D=D, chi2=chi2, scale=scale, pols=pols, native=native, pole=pole)

    def modulated(self, n, g, base):
        """Tracked-state yield: frozen full-coherence, frozen and live diagonal.

        Validation: cr-planar-zone-pxr-modulation
        """
        z, E0, D = self.zone, base["E0"], base["D"]
        Qx = float((E0 / HBARC * n + g) @ self.xhat)
        grid = E0 + np.arange(-WINDOW_EV, WINDOW_EV + STEP_EV / 2, STEP_EV)
        live, det, q = self.field(grid, n, g)
        qs = (q * self.svel)[None, :]
        frozen_full = frozen_diag = freq_diag = lost = 0.0
        norm = sum(a[0] ** 2 for a, _ in base["pols"]) * self.trace
        for r, kappa in enumerate(z["kappa"]):
            pw = kappa + z["orders"] * self.G
            ef, uf = np.linalg.eigh(self.pots[r] + np.diag(self.C * (pw - Qx) ** 2))
            vec, rho = z["vectors"][r], z["rho"][r]
            A = uf.conj().T @ vec
            B = uf.conj().T @ ((self.cv * (pw - Qx / 2))[:, None] * vec)
            pop = rho.diagonal().real
            weight = np.zeros(A.shape)
            for a0, b0 in base["pols"]:
                M = a0[0] * A + b0[0] * B
                frozen_full += np.einsum("fi,ij,fj->", M, rho, M.conj()).real
                frozen_diag += (abs(M) ** 2 @ pop).sum()
                weight += abs(M) ** 2 * pop[None, :]
            delta = (z["energies"][r][None, :] - ef[:, None]) / D
            keep = (weight > 1e-12 * norm) & (abs(delta) < WINDOW_EV - 20)
            lost += weight[~keep].sum() / norm
            pairs = np.argwhere(keep)
            for chunk in np.array_split(pairs, max(1, len(pairs) // 64)):
                if not len(chunk):
                    continue
                f, i = chunk[:, 0], chunk[:, 1]
                mismatch = D * (grid[None, :] - (E0 + delta[f, i])[:, None]) / (HBARC * self.beta)
                inside = self.phi(mismatch) - np.exp(1j * qs * self.L) * self.phi(mismatch - qs)
                born = self.phi(mismatch) * (-qs) / (mismatch - qs)
                elastic = (abs(delta[f, i]) * D < ELASTIC_EV)[:, None]
                K = np.where(elastic, born, inside) / det
                intensity = sum(
                    abs((a[None, :] * A[f, i][:, None] + b[None, :] * B[f, i][:, None]) * K) ** 2
                    for a, b in live
                )
                freq_diag += (np.trapezoid(intensity * grid, grid, axis=1) * pop[i]).sum()
        mismatch = D * (grid - E0) / (HBARC * self.beta)
        born = self.phi(mismatch) * (-qs[0]) / (mismatch - qs[0]) / det
        free = sum(abs(a * born) ** 2 for a, _ in live)
        out = dict(
            Qx=Qx,
            free_born=base["chi2"] * np.trapezoid(free * grid, grid) * self.trace,
            frozen_full=base["scale"] * frozen_full,
            frozen_diag=base["scale"] * frozen_diag,
            freq_diag=base["chi2"] * freq_diag,
            lost_weight=float(lost),
        )
        out["estimate"] = out["freq_diag"] + out["frozen_full"] - out["frozen_diag"]
        return out


def _centres(model, m, case):
    from pyrite.materials.crystal import reciprocal_g_vector

    beam = model.beam
    bragg = m["_bragg_directions"](case, beam, 50.0, 966203.0)
    bragg.sort(key=lambda item: -item[2])
    tangent = bragg[0][1] - (bragg[0][1] @ beam) * beam
    tangent /= np.linalg.norm(tangent)
    out = [
        (
            f"{bragg[0][0]}-az polar-{a:g}",
            np.cos(np.deg2rad(a)) * beam + np.sin(np.deg2rad(a)) * tangent,
        )
        for a in POLAR_DEG
    ]
    g0, _ = reciprocal_g_vector((-1, -1, 3), model.lattice)
    g = model.R @ g0
    mirror = beam - 2 * (beam @ g) * g / (g @ g)
    mirror /= np.linalg.norm(mirror)
    out_axis = np.cross(beam, mirror)
    out_axis /= np.linalg.norm(out_axis)
    in_axis = np.cross(out_axis, mirror)
    peak = mirror - in_axis / model.gamma
    out += [("(-1 -1 3) mirror", mirror), ("(-1 -1 3) minus-in peak", peak / np.linalg.norm(peak))]
    tangent = mirror - (mirror @ beam) * beam
    tangent /= np.linalg.norm(tangent)
    out += [
        (
            f"(-1 -1 3)-az polar-{a:g}",
            np.cos(np.deg2rad(a)) * beam + np.sin(np.deg2rad(a)) * tangent,
        )
        for a in (30.0, 37.5, 45.0)
    ]
    return out


def _production_state(case, electrons):
    """Production transport and the zone source's inputs, captured once (#1166)."""
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import channeling_emission, planar_zone_pxr

    case = {**case, "Ne": electrons, "Ne_brem": electrons, "line_quadrature": "bin-mean"}
    transport = runner._transport_case(case, transport_core="per-electron")
    seen = {}
    original = planar_zone_pxr.case_zone_pxr_spectrum

    def spy(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return original(*args, **kwargs)

    planar_zone_pxr.case_zone_pxr_spectrum = spy
    try:
        # Populations come from the depth history; a two-node grid skips the
        # forward CR spectrum, which this check does not score.
        grid = np.asarray(transport["E_grid"], dtype=float)[[0, -1]]
        channeling_emission.case_channeling_spectrum(case, {**transport, "E_grid": grid})
    finally:
        planar_zone_pxr.case_zone_pxr_spectrum = original
    return dict(case=case, transport=transport, source=original, **seen)


def _production_bands(state, hkl, n):
    """``(production, native)`` zone-row and native outside bands of one reflection.

    At one ray, point detector [photons/sr/electron]. Production is the source
    for source-owned reflections and native zone rows otherwise.
    """
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.planar_zone_lines import _rows
    from pyrite.montecarlo.runner.planar_zone_pxr import source_reflections

    case, transport = state["case"], state["transport"]
    segs = transport["segs"]
    zone = np.asarray(segs["channeling_zone"], dtype=bool)
    grid = np.asarray(transport["E_grid"], dtype=float)
    point = {**case, "dtheta_obs_rad": 0.0, "domega_sr": 0.0, "hkl_list": [hkl]}
    plain = {**point, "channeling": None}

    def native(keep):
        rows = _rows(segs, np.flatnonzero(keep))
        lines = runner._cxr_for_segments(
            rows, grid, plain, n, None, None, coherent=False,
            Ne=transport["Ne_lines"], table_cache={},
        )  # fmt: skip
        return float(np.trapezoid(np.asarray(lines, dtype=float), grid))

    native_zone, outside = native(zone), native(~zone)
    if not any(row[0] == hkl for row in source_reflections(point, n)):
        return native_zone, native_zone, outside
    _case, solver, populations, path, _grid, _n = state["args"]
    band = state["source"](point, solver, populations, path, grid, n, **state["kwargs"])
    return float(np.trapezoid(band, grid)), native_zone, outside


def _score_production(state, centre, row, args):
    """Production / independent-reference band ratios at one centre (#1166)."""
    from pyrite.validation.channeling_detector import square_aperture_rule

    beam = np.asarray(state["beam"], float)
    rays, weights, _ = square_aperture_rule(
        beam, centre, args.solid_angle_sr, args.distance_mm, args.order
    )
    sums = np.zeros(4)  # production zone, native zone, reference zone, outside
    per_g = {}
    for hkl_text, info in row["reflections"].items():
        hkl = tuple(int(v) for v in hkl_text.strip("()").split(","))
        bands = np.zeros(3)
        for n, w in zip(rays, weights, strict=True):
            bands += w * np.asarray(_production_bands(state, hkl, np.asarray(n, float)))
        prod, native_zone, outside = bands
        reference = native_zone * info["ratio"]
        per_g[hkl_text] = dict(
            production_zone=prod, native_zone=native_zone, reference_zone=reference,
            outside=outside, ratio_zone=prod / reference if reference else None,
        )  # fmt: skip
        sums += (prod, native_zone, reference, outside)
    prod, native_zone, reference, outside = sums
    return dict(
        production_over_reference_zone=prod / reference,
        production_over_reference_total=(prod + outside) / (reference + outside),
        production_over_native_zone=prod / native_zone,
        zone_share_of_native=native_zone / (native_zone + outside),
        reflections=per_g,
    )


def _score_centre(model, label, centre, args):
    from pyrite.validation.channeling_detector import square_aperture_rule

    rays, weights, _ = square_aperture_rule(
        model.beam, centre, args.solid_angle_sr, args.distance_mm, args.order
    )
    rays = [np.asarray(n, float) for n in rays]
    native, energy = {}, {}
    for hkl, g in model.reflections:
        total = 0.0
        for n, w in zip(rays, weights, strict=True):
            base = model.native(n, g, hkl)
            if base is not None:
                total += w * base["native"]
                energy[hkl] = base["E0"]
        if total > 0:
            native[hkl] = total
    hard = {h for h in native if energy[h] >= HARD_EV}
    top = max(native.values())
    top_hard = max([native[h] for h in hard] or [0.0])
    kept = [
        h for h in native if native[h] > 1e-3 * top or (h in hard and native[h] > 1e-3 * top_hard)
    ]
    keys = ("native", "free_born", "frozen_full", "frozen_diag", "freq_diag", "estimate")
    totals = dict.fromkeys(keys, 0.0)
    per_g = {}
    for hkl, g in model.reflections:
        if hkl not in kept:
            continue
        acc, info = dict.fromkeys(keys, 0.0), []
        for n, w in zip(rays, weights, strict=True):
            base = model.native(n, g, hkl)
            if base is None:
                continue
            part = {**base, **model.modulated(n, g, base)}
            for key in keys:
                acc[key] += w * part[key]
            info.append(part)
        per_g[str(hkl)] = dict(
            **acc,
            ratio=acc["estimate"] / acc["native"],
            ratio_vs_free_born=acc["estimate"] / acc["free_born"],
            free_born_over_native=acc["free_born"] / acc["native"],
            hard=hkl in hard,
            E0_range_eV=[min(p["E0"] for p in info), max(p["E0"] for p in info)],
            pole_offset_range_eV=[min(p["pole"] for p in info), max(p["pole"] for p in info)],
            lost_weight_max=max(p["lost_weight"] for p in info),
        )
        for key in keys:
            totals[key] += acc[key]
    screened = sum(v for h, v in native.items() if h not in kept)
    hard_native = sum(v["native"] for v in per_g.values() if v["hard"])
    hard_estimate = sum(v["estimate"] for v in per_g.values() if v["hard"])
    return dict(
        centre=label,
        n_hat=[float(v) for v in centre],
        polar_deg=float(np.degrees(np.arccos(np.clip(centre @ model.beam, -1, 1)))),
        order=args.order,
        totals=totals,
        ratio_estimate=totals["estimate"] / totals["native"],
        ratio_vs_free_born=totals["estimate"] / totals["free_born"],
        free_born_over_native=totals["free_born"] / totals["native"],
        ratio_frozen_full=totals["frozen_full"] / totals["native"],
        ratio_live_over_frozen_diagonal=totals["freq_diag"] / totals["frozen_diag"],
        hard_ratio_estimate=hard_estimate / hard_native if hard_native else None,
        hard_share=hard_native / totals["native"],
        screened_native_fraction=screened / (screened + totals["native"]),
        reflections=per_g,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="channeling_demo")
    parser.add_argument("--material", default="silicon")
    parser.add_argument("--order", type=int, default=3, help="Gauss order per square side")
    parser.add_argument("--solid-angle-sr", type=float, default=2.05e-4)
    parser.add_argument("--distance-mm", type=float, default=5000.0)
    parser.add_argument("--centre", default="", help="score only centres whose label contains this")
    parser.add_argument("--json", default=None, help="write the record here")
    parser.add_argument(
        "--production", action="store_true", help="also score the production zone source (#1166)"
    )
    parser.add_argument("--electrons", type=int, default=50, help="production Ne = Ne_brem")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    m = runpy.run_path(str(Path(__file__).with_name("channeling_detector_components.py")))
    case = m["_profile_case"](args.profile, args.material)
    if case.get("channeling", {}).get("model") != "planar":
        raise SystemExit(f"profile {args.profile!r} is not a planar channeling case")
    model = Model(case, _capture_density(case))
    state = _production_state(case, args.electrons) if args.production else None
    results, ok = [], True
    for label, centre in _centres(model, m, case):
        if args.centre and args.centre not in label:
            continue
        row = _score_centre(model, label, centre, args)
        ok &= all(np.isfinite(v) for v in row["totals"].values())
        if state is not None:
            row["production"] = _score_production({**state, "beam": model.beam}, centre, row, args)
            p = row["production"]
            ok &= bool(np.isfinite(p["production_over_reference_total"]))
            print(
                f"{label:30s} production/reference zone {p['production_over_reference_zone']:.4f}"
                f"  total {p['production_over_reference_total']:.4f}"
                f"  production/native zone {p['production_over_native_zone']:.4f}"
                f"  zone share {p['zone_share_of_native']:.3f}",
                flush=True,
            )
        results.append(row)
        hard = row["hard_ratio_estimate"]
        print(
            f"{label:30s} polar {row['polar_deg']:7.3f}  band est/plane-wave {row['ratio_estimate']:.4f}"
            f"  vs free-Born {row['ratio_vs_free_born']:.4f}  free-Born/native {row['free_born_over_native']:.4f}"
            f"  frozen {row['ratio_frozen_full']:.4f}  hard(>=2 keV) "
            f"{'-' if hard is None else f'{hard:.4f}'} (share {row['hard_share']:.2e})",
            flush=True,
        )
        for hkl, v in sorted(row["reflections"].items(), key=lambda kv: -kv[1]["native"])[:3]:
            print(
                f"    {hkl:14s} share {v['native'] / row['totals']['native']:.3f} ratio {v['ratio']:.4f}"
            )
        for hkl, v in sorted(
            ((h, v) for h, v in row["reflections"].items() if v["hard"]),
            key=lambda kv: -kv[1]["native"],
        )[:3]:
            print(
                f"    {hkl:14s} hard, share {v['native'] / row['totals']['native']:.2e} ratio {v['ratio']:.4f}"
                f" vs free-Born {v['ratio_vs_free_born']:.4f}"
                f" E0 {v['E0_range_eV'][0]:.0f}-{v['E0_range_eV'][1]:.0f} eV"
                f" pole +{v['pole_offset_range_eV'][0]:.1f}..{v['pole_offset_range_eV'][1]:.1f} eV"
            )
    record = dict(
        profile=args.profile,
        solid_angle_sr=args.solid_angle_sr,
        distance_mm=args.distance_mm,
        order=args.order,
        zone=dict(
            depths=model.zone["depths"],
            trace_depth_mean=model.trace,
            gamma=model.gamma,
            path_ang=model.L,
        ),
        window_eV=WINDOW_EV,
        step_eV=STEP_EV,
        units="ratios of band-integrated yield; absolute columns in model units",
        results=results,
        wall_s=time.perf_counter() - start,
    )
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
