# Transverse-recoil observability screen (Shi 2024, #527 option B)

Validation: `quantum-recoil-observability`.

## Scope and independence

This is a fresh-context verification at `a415ac5ee`, with the uncommitted
#527 working-tree edits present. The order of reading was:

1. The ledger row and the source: Shi, Wong, Huang, Wong & Kaminer,
   Nat. Commun. 15, 7803 (2024), main text, Methods Eqs. (2)–(8), Fig. 2, and
   SI §S6–§S10.
2. §1–§4 below, derived and computed with the validator's own scripts. These
   use the unsquared mass shell, an independent diamond-cubic selection rule
   and a pixel-calibrated read of the Fig. 2 bitmap.
3. Only then: the research note
   (`docs/research/physics/transverse-recoil-527.md`), the check
   `checks/transverse_recoil_527.py`, and the root function
   `materials/recoil.py::recoil_resonance_eV`.

The claim is measurement-only, so no production code is under review. The
validator edited nothing except this file.

## 1. Plane-wave pathways add incoherently

Shi Eq. (3) gives the rate as

$$
\frac{d^2N}{d\omega\,d\Omega}\propto\sum_{\mathbf q_f s_f\sigma}
\Bigl|\sum_{\mathbf q_i s_i\mathbf g}\varphi_{s_i}(\mathbf q_i)\,
M_{\mathbf q_i s_i\mathbf g\to\mathbf q_f s_f,\mathbf k\sigma}\,
\delta_{\mathbf q_i+\mathbf g,\;\mathbf q_f+\mathbf k}\Bigr|^2 .
$$

For a plane wave, $\varphi$ has a single support point $\mathbf q_i$. At a
fixed photon $\mathbf k$, the Kronecker delta then fixes
$\mathbf q_f=\mathbf q_i+\mathbf g-\mathbf k$. Distinct $\mathbf g$ therefore
reach orthogonal final electron states. The inner sum has one term, and the
outer sum over $\mathbf q_f$ is a sum of $\lvert M_{\mathbf g}\rvert^2$, with
no cross terms. A cross term needs $\mathbf q_i+\mathbf g=\mathbf q_j+\mathbf h$
with $\mathbf q_i\ne\mathbf q_j$, which is the second line of Eq. (3), i.e. a
shaped state (option C).

**Finite film.** In a laterally infinite film, energy and tangential momentum
are conserved exactly. The normal component is conserved only to
$\sim2\pi/L$ (a $\operatorname{sinc}^2$ profile).

- Pathways with distinct $\mathbf g_\perp$ stay orthogonal exactly.
- Pathways with equal $\mathbf g_\perp$ and different $g_z$ could reach a common
  final state. Their lines are, however, separated by the $n_z$ spacing
  (≥ 700 eV at 30 keV, θ = 100°), against $\operatorname{sinc}$ widths of
  ~10–25 eV. The overlap is negligible.

**Line centres.** Each pathway's energy conservation, with the crystal as an
infinite mass, is Shi Eq. (6). In the PyRITE sign,
$\mathbf P_f=\mathbf P-\mathbf Q-\epsilon\hat{\mathbf n}$ with
$\mathbf Q=\hbar c\,\mathbf g$, $E=T+mc^2$ and $\mathbf P=\beta E\hat{\mathbf u}$:

$$
E-\epsilon=\sqrt{m^2c^4+\lVert\mathbf P-\mathbf Q-\epsilon\hat{\mathbf n}\rVert^2}
\;\Rightarrow\;
\epsilon\Bigl(1-\beta\hat{\mathbf u}\cdot\hat{\mathbf n}
+\frac{\mathbf Q\cdot\hat{\mathbf n}}{E}\Bigr)
=\beta\hat{\mathbf u}\cdot\mathbf Q-\frac{Q^2}{2E}.
$$

Shi writes $\mathbf q_f=\mathbf q_i+\mathbf g_{\rm Shi}-\mathbf k$ (Eq. (3)
delta; Eq. (7): $q_{fx}=g_x-(\omega/c)\sin\theta$). Hence
$\mathbf g_{\rm PyRITE}=-\mathbf g_{\rm Shi}$, and the photon lies in the
$x$–$z$ plane. Shi's label $(n_x,n_y,-2)$ is therefore PyRITE row
$(-n_x,-n_y,2)$. The unshaped spectrum is the incoherent sum, over $\mathbf g$,
of lines centred on these exact roots. With $Q^2\to0$ at fixed
$\hat{\mathbf u}\cdot\mathbf Q$, the root reduces to the classical
$\epsilon_c=\beta\hat{\mathbf u}\cdot\mathbf Q/(1-\beta\hat{\mathbf u}\cdot\hat{\mathbf n})$,
which merges all rows with equal $g_z$ (Shi Eq. (8)).

**Classical counterpart.** Classically, rows with equal $g_z$ share one
resonance. Their cross terms vanish only after averaging the impact parameter
over the cell, through $e^{-i(\mathbf g-\mathbf h)_\perp\cdot\mathbf r_0}$
(Shi: "$\mathbf r_0$ … averaged over the crystalline unit cell"). PyRITE's
per-row squaring ([`cross-reflection-coherence`](cross-reflection-coherence.md))
is that average. The classical and plane-wave quantum spectra therefore share
the same incoherent per-row structure; only the line centres (and the matrix
elements) differ. **Item 1: pass.**

Filters:

- Units: every term is in eV. **Pass.**
- Limits: $\hbar\to0$ gives the classical root; at 5 MeV the roots converge
  (§3). **Pass.**
- Sign: Shi $(2,0,-2)$ ↔ PyRITE $(-2,0,2)$. This is the line whose $g_x$
  offsets the photon's $+x$ momentum, so it has the smallest shift. **Pass.**

## 2. Rigid-shift prototype as a screen

Construction (verified in the check after §1–§4):

- PyRITE's classical PXR is computed one row at a time.
- Each row's spectrum is translated by the nominal $\epsilon-\epsilon_c$,
  computed at $\hat{\mathbf u}=\hat{\mathbf z}$ and $T_0$.
- The rows are then summed.
- On the grid this preserves each row's area and shape exactly, apart from
  edge loss. It also preserves the per-row intensity, which carries the
  classical, not quantum, matrix element.

Given the stated non-claims (no quantum $M$, no Jacobian, no
backgrounds/divergence), this is a fair screen for line positions, comb
structure and resolution. It is not a screen for absolute or relative line
heights.

**Nominal-direction approximation.** The validator evaluated the exact
$(\epsilon-\epsilon_c)$ at each transported flight's own direction and energy
(the check's seed-11 transport, 2000 electrons). It then took the deviation
from the nominal shift, weighted by flight length.

| Case | Unscattered length fraction | Median flight angle | rms (mean) shift error, leading rows | rms, $\lvert g_\perp\rvert^2\approx 20$ rows |
|---|---|---|---|---|
| Si 30 keV, 30 nm, 100° | 0.59 | 0.5° | ≈ 1 eV (analytic: ±0.9 eV at 5°) | 2–4 eV (mean ≤ 0.15) |
| Si 300 keV, 30 nm, 100° | 0.91 | 0° | < 0.5 eV | 0.4–1.2 eV |
| Si 100 keV, 100 nm, 2.5° | 0.53 | 0.45° | 1.6–5.5 eV (mean ≤ 0.5) for the $\lvert g_\perp\rvert^2=10$ shell | — |
| HOPG (004) 11.7 keV, 85 nm, 130° | 0.18 | 5.2° | 2.0 eV (mean −0.5) | — |

Shift variation as a function of tilt was also computed analytically:

- 30 keV $(0,\pm2,2)$: ±0.4 / ±0.9 / ±1.9 eV at 2° / 5° / 10°.
- Forward $(3,1,1)$: ±3 / ±8 / ±12–18 eV at the same tilts.

The errors are odd in the tilt to first order, so they broaden lines rather
than move them. The mean errors are ≤ 0.5 eV, and the peak is carried by the
unscattered majority.

The approximation is adequate for the stated peak-shift figures and for
everything at 70 eV resolution. At 0–5 eV resolution, the transported
FWHMs carry a few-eV systematic.

**Item 2: pass**, with that caveat.

## 3. Independent numbers

Si: $a=5.431$ Å, $2\pi\hbar c/a=2282.9$ eV, θ = 100°, photon direction
$(\sin\theta,0,\cos\theta)$. Rows use the diamond-cubic rule: all odd, or all
even with $h+k+l\equiv0\pmod 4$. PyRITE sign.

| $T$ | classical $n_z=2$ / $3$ (eV) | $(-2,0,2)$ | $(0,\pm2,2)$ | $(2,0,2)$ |
|---|---|---|---|---|
| 30 keV | 1418.42 / 2127.63 | 1394.86 (−23.55) | 1383.88 (−34.54) | 1373.07 (−45.35) |
| 300 keV | 3124.19 / 4686.28 | 3119.46 (−4.72) | 3104.21 (−19.98) | 3089.10 (−35.08) |
| 5 MeV | 3875.97 / 5813.96 | 3875.92 (−0.05) | 3873.22 (−2.75) | 3870.53 (−5.44) |

The unsquared mass-shell residuals are ≤ $10^{-9}$ eV.

**Forward case.** 100 keV, θ = 2.5°, PyRITE $g_z=+1$ (Shi $n_z=-1$). The
classical root is 2767.03 eV. Exact shells:

- 2717.30 / 2715.36 eV for $(\mp1,\pm1,1)$, shift −49.7 / −51.7 eV;
- 2644.35 / 2642.46 / 2640.57 / 2638.68 eV for the $\lvert g_\perp\rvert^2=10$
  shell, shift −122.7 to −128.4 eV;
- 2569.45 / 2563.95 eV for $(\pm3,\pm3,1)$.

**HOPG.** (004), $c=6.71$ Å, normal incidence ($g_\perp=0$), 11.7 keV, 130°:
classical 1369.79 eV, exact 1334.44 eV, shift −35.35 eV.

**Agreement.** All of these match the research note §2/§3.2 tables and the
check output (e.g. 1383.9, 3104.3, 2767.0 and 1334.3 eV on the grid). The
§2 table lists rows by Shi $(n_x,n_y)$ under a header that defines
$\mathbf g=-\mathbf n\,2\pi/a$, so "(2,0) 1394.9" is consistent.

**Check re-run.** `checks/transverse_recoil_527.py 2000` gives rc 0, `PASS`.
It reproduces the §5.1 and §5.2 tables to the printed digit:

- 30 keV: −35 / −35 / −38.5 eV, centroid −101, FWHM 13 / 18 → 81.5 / 125.5 eV,
  height ratio 0.40 / 0.41 / 0.59;
- 300 keV: −19.5 / −19.5 / −27.5 eV, centroid −65;
- forward: −125.5 / −125.5 / −121 eV, centroid −239, FWHM 15 / 90 → 83 / 282.5 eV,
  height ratio 0.27 / 0.27 / 0.34;
- HOPG: −35.5 / −35 / −35 eV, centroid −34.

Straight-segment rows from the same run, used in §5:

| Case, detector | straight peak shift | transported | straight FWHM CL / QM | transported |
|---|---|---|---|---|
| 30 keV, 70 eV | −38.5 | −38.5 | 72 / 111 | 81.5 / 125.5 |
| 300 keV, 70 eV | −27.5 | −27.5 | 75 / 104 | 77 / 106.5 |
| forward, 0 eV | −125.5 | −125.5 | 13 / 86.5 | 15 / 90 |
| forward, 70 eV | **−123.5** | **−121.0** | 72.5 / **255** | 83 / **282.5** |
| HOPG, 0 eV | −35.5 | −35.5 | 2 / 2 | 13.5 / 13.5 |

**Forward resolvability.** These are local maxima of the 70 eV prototype.

- Straight: 2644 eV (1.00) and 2705 eV (0.74), so the first two shells are
  separate.
- Transported: 2645 eV (1.00), 2493, 2265 and 2045 eV. The 2716 eV shell has
  no maximum of its own and is a shoulder.

## 4. Shi Fig. 2 reads

Method: the validator calibrated the bitmap axes with the tick marks.

- $x$ scale: 3.295 eV/px (2f), 3.571 eV/px (2d) and 3.279 eV/px (2b).
- $y$ scale: per-panel tick spacing, left and right axes separately.
- Validation of the calibration: the black CL peak positions read 1418.1, 3123.2
  and 3875.4 eV, against the classical roots 1418.4, 3124.2 and 3876.0 eV.
  Positions are therefore good to ~1 eV (±1 px ≈ ±3.5 eV).

| Panel | Quantity | Validator read | Claim/check | Result |
|---|---|---|---|---|
| 2b, 5 MeV | CL $n_z=-2$ / $-3$ | 4.41e-11 / 3.81e-11 | 4.4e-11 / 3.8e-11 | match |
| 2b | QM $n_z=-2$ / $-3$ | 3.72e-11 / 3.59e-11 | — | — |
| 2d, 300 keV | CL $n_z=-2$ / $-3$ | 5.86e-11 / 5.95e-11 | 5.8e-11 / 5.5e-11 | match (−8 % on $n_z=-3$) |
| 2d | tallest QM $n_z=-2$ | 5.35e-11 at **3105.4 eV** | ≈ 5.3e-11 | height match |
| 2f, 30 keV | CL $n_z=-2$ | 1.16e-9 | 1.15e-9 | match |
| 2f | CL $n_z=-3$ (right axis, ×1e-9) | **1.21e-9** | **0.9e-9** | **differs, −26 %** |
| 2f | tallest QM $n_z=-2$ | 1.10e-9 at **1383.5 eV** | ≈ 1.05e-9 | height match |

The derived ratios:

- PyRITE/Shi CL for $n_z=-2$: 1.84 at 30 keV and 2.53 at 300 keV. The claimed
  1.9× / 2.6× is confirmed.
- PyRITE/Shi CL for $n_z=-3$ at 30 keV: 1.22/1.21 ≈ **1.0**, not the note's
  1.36.
- At 5 MeV: 0.95 and 0.95, against the claimed 0.96 / 0.95.

**Identification of Shi's tallest QM line (item 4).** In both panels, the
tallest dotted peak lies at the **$(0,\pm2,2)$ root**, not at the $(-2,0,2)$
root:

- 2f: 1383.5 eV, against 1383.9 for $(0,\pm2,2)$ and 1394.9 for $(-2,0,2)$, a
  3.3 px separation.
- 2d: 3105.4 eV, against 3104.2 and 3119.5, a 4.3 px separation.

At 10× zoom of 2f, no separate tall peak sits at 1394.9 eV; that position is
on the tallest peak's high-energy flank. The doublet perpendicular to the
photon plane is the tallest line for either in-plane photon azimuth. It is
also the row PyRITE weights highest.

Shi's own peak shifts are therefore ≈ −35 eV at 30 keV and ≈ −18 eV at
300 keV. The prototype's straight peak shifts are −35.0 and −19.5 eV. The
sign convention (PyRITE $\mathbf g=-$Shi label) is correct, but the claimed
identification "Shi's sits at $(-2,0,2)$" is **wrong**. So is the note's
"with Shi's weighting the tallest line is the $(-2,0,2)$ root, −23.5 eV at
30 keV and −4.7 eV at 300 keV".

What remains unvalidated is the relative weighting:

- Shi's tallest QM line is ≈ 0.9–0.95 × CL; the prototype's is 0.36–0.41 × CL.
- The per-row PyRITE weights differ from Shi's.

## 5. Over-statement review (item 5)

1. **Tallest-line identification.** The claim's *Not claimed* sentence
   ("PyRITE's per-reflection weights put the tallest line at (0,±2,2) where
   Shi's sits at (−2,0,2), so peak-shift magnitudes at large angle depend on
   the unvalidated weights") is wrong as stated. The same error appears in
   note §2 ("where the $(2,0,-2)$ line falls"), §5.1 and §5.2.
   - Both sit at $(0,\pm2,2)$, and the large-angle peak shifts agree with
     Shi's figure to ≈ 1–2 eV.
   - The caveat that should remain is the relative height: prototype QM/CL
     0.36–0.41 against Shi ≈ 0.9.
2. **"Multiple scattering changes peak shifts by < 1 eV and widths by
   ≤ 15 eV."** This fails in the forward case at 70 eV resolution:
   - the peak shift changes from −123.5 to −121.0 eV (2.5 eV);
   - the prototype FWHM changes from 255 to 282.5 eV (+27.5 eV).

   All other cases are within the bound; the largest is the 30 keV 70 eV
   case, 111 → 125.5 eV (+14.5). A correct statement is "< 3 eV and ≤ 30 eV",
   or the bound can be restricted to the 0/5 eV detectors.
3. **"Resolvable with a 70 eV detector" (forward).** True of the
   ≈ −121 eV envelope shift, which is 1.7 detector FWHM. It is not true of
   individual shells after transport: the 2716 eV shell becomes a shoulder
   (§3).
   - The forward peak shift (−125 eV, at the $\lvert g_\perp\rvert^2=10$
     shell) depends on PyRITE's unvalidated relative shell weights.
     $\lvert g_\perp\rvert^2=2$ is at 0.74 of the peak, so a modest
     reweighting would give ≈ −50 eV.
   - The claim restricts weight dependence to "large angle". The forward
     case has no external height reference (Shi Supplementary Fig. 8b data
     were not available).
   - Note §5.2's "sixfold broadening survive[s] a 70 eV detector" is also
     imprecise. At 70 eV the broadening is 3.4×; sixfold is the 0 eV figure.
4. **Shi Fig. 2f CL $n_z=-3$ read.** The read is 0.9e-9 against 1.21e-9 (§4).
   It is not gated and appears only in note §5.1 and the check's `SHI_CL`. It
   changes the 30 keV $n_z=-3$ ratio from 1.36 to ≈ 1.0.
5. **HOPG.** "A rigid −35 eV shift" assumes normal incidence (beam along
   $\hat{\mathbf c}$). That is different from Huang's tilted geometry
   (12.45°), where $g_\perp\ne0$ relative to the beam. Both the row and the
   note say "$g_\perp=0$", so this is consistent, but the case is
   "Huang-like", not Huang's geometry.
6. **Nominal-direction shift.** The disclosure is adequate (§2).
7. **Rendering defect.** Note §5 contains `$\lvert hvert,\lvert kvert\le 8$`,
   where `\rvert` has lost its backslash-r. It renders as "hvert".
8. **No change needed.** The 5 MeV heights (0.96 / 0.95), the classical-peak
   placement (≤ 0.7 eV from the root), the area preservation and the HOPG
   shift all reproduce.

## 6. Verdict

- **Claim**: `quantum-recoil-observability` — measurement only,
  `checks/transverse_recoil_527.py` against
  `materials/recoil.py::recoil_resonance_eV` and `mc_cxr_spectrum` — Shi et al.
  2024 Eqs. (3), (6)–(8), Fig. 2, SI §S6, §S10.
- **Filters**:
  - units: pass;
  - limits: pass ($\hbar\to0$ gives the classical root; 5 MeV collapse);
  - signs/conventions: pass ($\mathbf g_{\rm PyRITE}=-\mathbf g_{\rm Shi}$,
    photon in the $x$–$z$ plane);
  - reference data: **failure**, as two items:
    - the Fig. 2f/2d tallest-QM identification (it is $(0,\pm2,2)$, not
      $(-2,0,2)$);
    - the 30 keV $n_z=-3$ CL read.
- **Re-derivation**: `matches` for the physics:
  - incoherent per-row sum for plane waves;
  - exact roots;
  - rigid-shift construction;
  - every table number in note §5.1–§5.2.

  `differs` in two statements:
  - First divergent item: Shi's tallest QM line at 30/300 keV is the
    $(0,\pm2,2)$ root (1383.5 / 3105.4 eV read), not $(-2,0,2)$ (1394.9 /
    3119.5 eV).
  - Second: the multiple-scattering bound fails for the forward case at
    70 eV (Δ peak shift 2.5 eV > 1 eV; Δ FWHM 27.5 eV > 15 eV).
- **Verdict**: `discrepancy`, confined to claim wording and one reference read.
  No numerical result of the check is wrong. After the text corrections below,
  the expected verdict is `rederived`.
- **Write-up**: `docs/validation/radiation-physics/quantum-recoil-observability.md`
- **Suggested ledger change** (owner applies; not a sign-off):
  - In *Fig. 2 large-angle geometry*, append: "Shi's tallest QM line sits at
    the same (0,±2,2) root (Fig. 2f/2d pixel reads 1383.5 / 3105.4 eV), so the
    peak positions agree; the prototype's QM/CL height (0.36–0.41) is below
    Shi's (≈ 0.9)."
  - In *SI §S10 forward case*, replace "resolvable with a 70 eV detector"
    with: "with a 70 eV detector the envelope peak moves −121 eV and the
    FWHM grows 3.4× (83 → 283 eV); after transport the leading (1̄,±1) shell
    is a shoulder, not a separate peak. The −125 eV peak is the
    |g⊥|² = 10 shell and depends on PyRITE's relative shell weights
    (|g⊥|² = 2 at 0.74 of it)."
  - Replace *Multiple scattering* with: "at these thicknesses it changes peak
    shifts by ≤ 2.5 eV and widths by ≤ 30 eV (largest: forward, 70 eV
    detector); the nominal-direction recoil shift errs by ≤ 5.5 eV rms on
    scattered flights (mean ≤ 0.5 eV)."
  - In *Not claimed*, replace "and PyRITE's per-reflection weights put the
    tallest line at (0,±2,2) where Shi's sits at (−2,0,2), so peak-shift
    magnitudes at large angle depend on the unvalidated weights" with "and
    PyRITE's per-reflection weights differ from Shi's (prototype tallest
    line 0.36–0.41× CL against Shi ≈ 0.9×), so relative line heights, and
    any peak shift that depends on them (notably the forward case), are
    unvalidated".
  - Add this write-up to *Notes* and to
    `docs/validation/radiation-physics/index.md`.
  - Research note: correct §2 bullet 2, §5.1 bullet 2 and the §5.2
    large-angle bullet the same way. Change the 30 keV $n_z=-3$ Shi CL to
    1.2e-9 (ratio ≈ 1.0) in §5.1 and in the check's `SHI_CL`. Fix "sixfold"
    → "3.4× at 70 eV". Fix the `\lvert h\rvert` markup in §5.

### Re-check after owner corrections (2026-11-11)

The owner applied the §6 corrections without committing them. The same fresh
context re-checked them and edited only this write-up.

- **Ledger row.**
  - The large-angle bullet now places Shi's tallest QM line at the $(0,\pm2,2)$
    root (1383.5 / 3105.4 eV) and keeps the height caveat: 0.36–0.41 against
    ≈ 0.9.
  - The forward bullet gives −121 eV, 3.4× (83 → 283 eV), the leading-shell
    shoulder and the shell-weight dependence.
  - The multiple-scattering bullet reads ≤ 2.5 eV / ≤ 30 eV and ≤ 5.5 eV rms
    (mean ≤ 0.5 eV).
  - *Not claimed* uses the replacement sentence.
  - *Notes* link this write-up.
  - Status stays `unverified` for the owner to set.
- **Research note.**
  - §2 bullet 2, §5.1 (table row 30 keV $n_z=-3$: Shi 1.2e-9, ratio 1.0; tallest
    line) and the §5.2 large-angle, forward and multiple-scattering bullets now
    match §3–§5 here.
  - The §5 markup now reads $\lvert h\rvert,\lvert k\rvert\le 8$.
  - One non-blocking residue: §3.2 still says "the shells are resolvable with a
    ~70 eV EDS, in agreement with SI §S10". That is Shi's statement for
    unscattered (straight) emission, where the prototype also shows separate
    maxima at 2644 and 2705 eV. The same subsection already says that elastic
    scattering smears scattered flights. §5.2 states the transported shoulder.
    Adding "on unscattered segments" there would remove the apparent tension.
- **Check.** `SHI_CL[30.0]` is `(1.15e-9, 1.2e-9)`. Re-run with 2000 electrons:
  - rc 0, `PASS`;
  - 30 keV $n_z=-3$ ratio 1.02;
  - forward 70 eV straight / transported −123.5 / −121.0 eV, FWHM 255 / 282.5 eV;
  - all other numbers unchanged from §3.
- **Toctree.** `quantum-recoil-observability` is in
  `docs/validation/radiation-physics/index.md`.

**Final verdict: `rederived`.** The derivation, the source reads and the
corrected claim, note and check agree. The owner applies the status; human
sign-off remains pending (#277).
