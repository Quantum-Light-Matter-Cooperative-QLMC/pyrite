# In-crystal line recoil (Huang 2023 harmonic defect)

Validation: `line-recoil-in-crystal`.

## Scope and independence

Fresh-context verification at `2ee11cf13`. The ledger row, the
`recoil_resonance_eV` signature and docstring, and the source (Huang et al.,
Nat. Photon. 17, 224 (2023), main text and SI) were read first. Sections 1–4
were derived and computed before the research note
(`docs/research/physics/quantum-recoil-in-crystal-527.md`) and
`checks/recoil_in_crystal_527.py` were opened. The claim is measurement-only:
no production code is under review. The validator's own scripts used
`mpmath` (50 digits) for an unsquared mass-shell solve, and independent
Cromer–Mann carbon form factors for the Bloch couplings.

## 1. Why the zero-phonon line centre is the vacuum root

Let $M=mc^2$, $E=T+M$, $\mathbf P=c\mathbf p_i=\beta E\hat{\mathbf u}$,
photon energy $\epsilon$ in direction $\hat{\mathbf n}$, and
$\mathbf Q=\hbar c\,\mathbf g$ with the PyRITE sign
$\mathbf P_f=\mathbf P-\mathbf Q-\epsilon\hat{\mathbf n}$.

**Static lattice.** The crystal Hamiltonian is invariant under time
translation and under lattice translations. A zero-phonon (Debye–Waller)
transition leaves the lattice state unchanged. It therefore takes no energy and
transfers momentum only in units of $\hbar\mathbf g$; it acts as an infinite
mass. Phonon-assisted transfers change the lattice state, so they do not add
intensity at this centre.

**Free-like Bloch branches.** Far from electron Bragg conditions, each
electron state is a Bloch wave whose dominant plane wave has the free
relativistic dispersion, plus the common mean inner potential $-U_0$ and
second-order branch shifts $\sum_h U_h^2/d_h$. With those set to zero, energy
conservation $E=\epsilon+E_f$ and crystal-momentum conservation are the
vacuum statements:

$$
E-\epsilon=\sqrt{M^2+\lVert\mathbf P-\mathbf Q-\epsilon\hat{\mathbf n}\rVert^2}.
$$

Squaring once cancels $\epsilon^2$ and gives the linear relation

$$
\epsilon\left(1-\beta\hat{\mathbf u}\cdot\hat{\mathbf n}
+\frac{\mathbf Q\cdot\hat{\mathbf n}}{E}\right)
=\beta\hat{\mathbf u}\cdot\mathbf Q-\frac{Q^2}{2E}.
$$

With $\mathbf g=-\mathbf g_{\rm paper}$ this is Huang Eq. (3) at $n=1$. The
validator rederived Eq. (3) from $\mathbf p_f=\mathbf p_i+\hbar\mathbf
g_{\rm paper}-\hbar\mathbf q$. Eq. (3) is therefore an exact root, not an
expansion; Eq. (4) is its expansion. The recoil terms come from the curvature
of the electron's own dispersion, not from lattice recoil, so the lattice
cannot remove them.

**Finite slab.** A slab with laterally infinite faces conserves the tangential
momentum and the energy exactly. The normal-momentum mismatch $\Delta q_z$
enters only through $\int_0^L e^{i\Delta q_z z}dz$, giving a
$\operatorname{sinc}^2(\Delta q_z L/2)$ profile. Its maximum lies at
$\Delta q_z=0$, which is the infinite-crystal root. The map from $\Delta q_z$
to $\epsilon$ is linear to $O(\Gamma^2\partial^2\epsilon)$ over the
$\Gamma\approx20$ eV intrinsic width (Huang Eq. (6)). The profile therefore
broadens the line without moving its centre at this order.

Filters: every term in the root has units of eV. As $\hbar\to0$, or
$\mathbf Q\to0$ at fixed $\beta\hat{\mathbf u}\cdot\mathbf Q/\lVert\mathbf
Q\rVert$, the root reduces to the classical $\epsilon_c=\beta\hat{\mathbf
u}\cdot\mathbf Q/(1-\beta\hat{\mathbf u}\cdot\hat{\mathbf n})$. The sign
convention agrees with the paper after the stated label negation. **Pass.**

## 2. Harmonic defect algebra

For collinear harmonics $\mathbf Q_m=m\mathbf Q_1$, $\epsilon_c$ is linear in
$\mathbf Q$, so for every tilt, energy and observation angle

$$
D_c=2\epsilon_c(\mathbf Q_2)-\epsilon_c(\mathbf Q_4)=0.
$$

An energy-dispersive (EDS) calibration error $\epsilon_{\rm meas}=a+b\epsilon$
maps any defect to $2(a+b\epsilon_2)-(a+b\epsilon_4)=a+bD$. A pure gain error
($a=0$) leaves $D_c=0$ and rescales a true defect by $b\approx1$; an offset
enters additively as $+a$. In general the calibration error enters as
$2e(\epsilon_2)-e(\epsilon_4)$. That reduces to $a$ only when $e$ is linear
over 600–1550 eV. If $e$ is only bounded, $\lvert e\rvert\le0.5$ eV, then
$\lvert2e_2-e_4\rvert\le1.5$ eV.

To first order in the transfer, the root is
$\epsilon\simeq\epsilon_c(1+s)$ with
$s=-Q^2/(2E\beta\hat{\mathbf u}\cdot\mathbf Q)-(\mathbf Q\cdot\hat{\mathbf
n}/E)/D$. The correction $\epsilon_c s$ scales as $m^2$, so

$$
D\simeq-2\,\epsilon_c(\mathbf Q_2)\,s(\mathbf Q_2)>0.
$$

The defect is minus twice the (002) recoil shift, or about 19 eV here. **Pass.**

## 3. Source and pinned-data checks

| Item | Source | Pinned / claimed | Result |
|---|---|---|---|
| Table S4, 13 rows ($V$, $E_2$, $E_4$, $\theta_{\rm til}$, $\phi_{\rm til}$) | SI PDF p. 17 (SI p. 16) | `tests/data/recoil/huang-2023-table-s4.json` | all 13 rows and 65 values match |
| Set assignment | Table S4 labels; Fig. 1d axis 10.5–13 keV; Fig. S5a/b axes 13.5–14.5 keV | fig1d ×7, figS5a ×3, figS5b ×3 | match |
| $\theta_{\rm obs}$ | Methods, X-ray measurements: 130° JXA-8530F (Fig. 1, Fig. 2a–c), 119° JSM-7800F Prime (Fig. 2d–f) | fig1d 130°; S5a, S5b 119° | match. The S5 caption puts both S5 panels on the JSM-7800F Prime; S5b is the Fig. 2d–f hBN set |
| Calibration | Methods; SI §S5 (Eq. (S12), Table S3, residuals −0.5…+0.5 eV) | ±0.5 eV | match |
| $c$ spacing | Fig. 2 labels: graphite 6.71 Å, hBN 6.62 Å | 6.71, 6.62 | match. Fig. 3 lists graphite as 6.70 Å, a different context |
| Reflection sign | Eq. (5): $\mathbf g_{(00\bar m)}=-m(2\pi/d)\hat{\mathbf c}$; Eq. (3) | PyRITE $\mathbf g=+m(2\pi/c)\hat{\mathbf c}$ | consistent (§1) |
| Thicknesses | Methods; SI §S6 | 85, 55, 95 nm | match |

Minor source notes:

- For the 11.7 keV Fig. 1c case, the main text gives $\phi_{\rm til}=0.3^\circ$
  and Table S4 gives $0.4^\circ$. The pinned file uses the table; the change in
  the defect is below 0.01 eV.
- The paper does not state the detector azimuth $\phi_{\rm obs}$; the pin
  assumes $0$. Fig. 1c's inset is consistent with that choice. Fig. S15
  instead draws $\hat{\mathbf c}$ tilted away from the detector, which
  corresponds to $\phi_{\rm obs}=180^\circ$. The predicted mean defect is
  19.10 eV at $\phi_{\rm obs}=0$ and 18.04 eV at $180^\circ$. Either value
  supports the conclusion; $\phi_{\rm obs}=0$ fits the absolute lines better
  (rms 3.2 against 3.8 eV for $E_2$).
- Tilts were fitted with the authors' quantum-plus-scattering Monte Carlo
  model, so absolute line energies are circular.
- SI §S11 uses $\hbar\omega_p\approx13$ eV (valence). At 600–1500 eV all six
  carbon electrons contribute, so $\delta$ is about five times larger
  (§4.2).

## 4. Independent numbers

### 4.1 Harmonic defect

The validator solved the unsquared mass shell per row and cross-checked it
against the closed form and `recoil_resonance_eV`; all three agree to
$<10^{-12}$ eV.

| Quantity | Validator | Claim |
|---|---|---|
| mean measured $2E_2-E_4$ (13 rows) | 18.192 eV (s.d. 1.656) | 18.19 |
| mean straight exact root | 19.101 eV | 19.10 |
| classical root | 0 for every row (to roundoff) | 0 |
| residual (measured − predicted) | −0.909 eV, s.e.m. 0.436 | −0.91 |
| $\sigma$ = residual s.e.m. ⊕ 0.5 eV offset | 0.663 eV | 0.66 |
| residual / $\sigma$ | −1.37 | within 2σ |
| classical exclusion, measured mean / $\sigma$ | 27.4 (26.8 if the measured-mean s.e.m. is used) | 27σ |
| classical exclusion, uncorrelated ±0.5 eV per line ($\sigma$ = 1.56) | 11.7 | > 10σ gate holds |

Per-set means of measured / predicted defect: Fig. 1d 18.19 / 19.15 eV,
Fig. S5a 18.47 / 18.17 eV, Fig. S5b 17.92 / 19.93 eV.

Sensitivity at 11.7 keV: $\partial D/\partial\theta_{\rm til}=+0.14$ eV per
degree, and $\partial D/\partial\theta_{\rm obs}\approx-0.16$ eV per degree.
The predicted defect is therefore insensitive to the fitted tilts. A symmetric
detector acceptance averages out to second order.

### 4.2 In-medium corrections at graphite 11.7 keV, $\theta_{\rm obs}=130^\circ$, $\theta_{\rm til}=12.45^\circ$, $\phi_{\rm til}=0.4^\circ$

The validator used the independent-atom $U_0=13.5$ eV (Cromer–Mann). Its
$\delta$ comes from $f_1\simeq6$ at $\rho=2.26$ g cm⁻³: $1.08\times10^{-3}$
at 659 eV and $2.8\times10^{-4}$ at 1299 eV. The values the check takes from
`refractive_index` (via `hopg`) are $1.14\times10^{-3}$ and $2.91\times10^{-4}$.

| Correction | $\Delta E_2$ (eV) | $\Delta E_4$ (eV) | $\Delta D$ (eV) | Claim |
|---|---|---|---|---|
| Mean inner potential, both states (root at $T+\lvert U_0\rvert$) | +0.330 | +0.664 | −0.004 | cancels |
| Entrance refraction at the (001) face (tangential $\mathbf p$ kept; direction 12.450° → 12.443°) | +0.031 | +0.062 | +0.0002 | ≤ 0.02 |
| Photon refraction, bulk index only, $\Delta\mathbf K=-\delta\epsilon\hat{\mathbf n}$ | +0.084 | +0.042 | +0.125 | ≤ 0.14 |
| Photon refraction with exit-face tangential continuity | +0.27 to +0.29 | +0.13 to +0.15 | **+0.41 to +0.44** | not evaluated |

The first-order photon shift follows from the mismatch
$\Phi=E-\epsilon-E_f(\mathbf P-\mathbf Q-\mathbf K)$ with
$\partial\Phi/\partial\mathbf K=c\boldsymbol\beta_f$:

$$
\Delta\epsilon=\frac{\boldsymbol\beta_f\cdot\Delta\mathbf K}{1-\boldsymbol\beta_f\cdot\hat{\mathbf n}}.
$$

The in-crystal wavevector must carry the exterior tangential momentum. With
$\hat{\mathbf N}$ the exit-face normal on the photon's side,
$K_N=\sqrt{n^2\epsilon^2-K_t^2}\simeq\epsilon\,\hat{\mathbf n}\cdot\hat{\mathbf
N}-\delta\epsilon/(\hat{\mathbf n}\cdot\hat{\mathbf N})$, so

$$
\Delta\mathbf K=-\frac{\delta\,\epsilon}{\hat{\mathbf n}\cdot\hat{\mathbf N}}\hat{\mathbf N},
\qquad
\Delta\epsilon=-\frac{\epsilon\,\delta\,\boldsymbol\beta\cdot\hat{\mathbf N}}
{(\hat{\mathbf n}\cdot\hat{\mathbf N})(1-\boldsymbol\beta\cdot\hat{\mathbf n})}.
$$

This is the repository's own `xray-in-medium-resonance` form,
$\mathbf k_{\rm eff}=\omega(\hat{\mathbf n}+\delta\mathbf h)$ with
$\mathbf h=-\mathbf e/(\mathbf e\cdot\hat{\mathbf n})$. Here the backward
photon leaves through the entrance face, 62.4° from its normal at
$\phi_{\rm obs}=0$ (37.6° at $180^\circ$). The exit-face form is 3.3 times the
bulk form at $\phi_{\rm obs}=0$; the result is +0.44 eV at $0$ and +0.26 eV at
$180^\circ$. The hBN 14.0 keV / 119° case gives +0.44 eV as well. Over all 13
rows the mean is +0.41 eV ($\phi_{\rm obs}=0$; range +0.34 to +0.49). That
moves the predicted mean defect to 19.51 eV and the residual to
$-1.32\pm0.66$ eV, or −1.99σ: just inside the 2σ gate. At
$\phi_{\rm obs}=180^\circ$ the residual is −0.19 eV.

### 4.3 Bloch branch shifts

The validator built its own AB-graphite reciprocal lattice ($a=2.464$ Å,
$c=6.71$ Å, $\Omega=35.28$ Å³) with Mott–Bethe
$U_h=(2\pi\hbar^2/m\Omega)\lvert F_e(h)\rvert$.

- Using the claim's per-coupling sum of two-beam shifts
  $-d/2+\operatorname{sign}(d)\sqrt{d^2/4+U_h^2}$ in the check's crystal frame
  reproduces its numbers exactly: $+0.101$ / $+0.327$ / $-0.125$ eV, and the
  nearest initial coupling at $d=+0.464$ eV with $U_h=0.350$ eV.
- However, the in-plane azimuth $\psi$ of the crystal about $[001]$ is not
  reported by Huang. The check's frame ($\hat{\mathbf x}_c\propto\hat{\mathbf
  y}\times\hat{\mathbf c}$) is an arbitrary choice.
- Scanning $\psi\in[0,60^\circ)$ in 0.25° steps, the free-like-branch defect
  shift has median $\lvert\Delta D\rvert\approx0.38$–$0.51$ eV. It exceeds
  0.14 eV for 82–87 % of azimuths and reaches 6–7 eV when a strong low-order
  coupling ($\lvert U_h\rvert$ of a few eV) sits at Bragg. These ranges cover
  Debye–Waller $B=0$ or 0.5 Å² and reflection cutoffs from
  $\lvert h\rvert,\lvert k\rvert\le3,\lvert l\rvert\le4$ to
  $\le10,\le16$.
- At the check's own azimuth, the result also depends on the cutoff:
  $-0.125$, $+0.122$ and $-0.326$ eV for the three cutoffs at $B=0$.
- The note's statement that strong low-order couplings are "tens of eV
  detuned at these geometries" therefore holds only for the assumed azimuth.

The observable centroid is protected by a sum rule. For one coupling, the
two branches $\alpha$ have excitations $\lvert C_{0\alpha}\rvert^2$ and shifts
$\lambda_\alpha$, and

$$
\sum_\alpha\lvert C_{0\alpha}\rvert^2\lambda_\alpha=\langle\mathbf k\rvert
V-U_0\lvert\mathbf k\rangle=0 .
$$

In the far-detuned limit, this is $(1-U^2/d^2)(U^2/d)+(U^2/d^2)(-d)\to0$.
At EDS resolution (52–72 eV FWHM), an unresolved split line keeps its
centroid at the vacuum root to the extent that branch emission matrix
elements are equal. The free-like-branch shift the claim tabulates is an upper
estimate of the line-peak displacement, not the centroid. So a Bloch effect
at the eV level is credible only as broadening. That argument, not the
single-azimuth number, supports neglecting Bloch shifts.

### 4.4 Scattering transport

Huang's Fig. 1d plots their scattering-inclusive Monte Carlo predictions. The
validator read the values against the red experimental points, which lie at
known Table S4 values:

- quantum ≈ 13.9, 15.0, 15.8, 16.7, 16.6, 15.6, 16.3 eV (mean ≈ 15.7 eV);
- classical ≈ −2 to −3.5 eV.

The straight exact root gives 19.15 eV for the same set. Huang's transport
model therefore lowers the defect by about 3 eV, roughly 5σ of the claim's
uncertainty. Peak-fitting broadened, asymmetric distributions with EDS
resolutions that do not scale with $m$ would produce this; their classical
value is not 0. This bias is not among the claim's listed omitted
corrections. The research note's statement that the straight-path agreement is
"consistent with their statement that scattering moves the recoil measurement
little" contradicts the plotted values.

Classical exclusion survives, and is stronger: −2.5 eV against 18.2 eV. The
interpretation that the straight-path root "reproduces" the measured defect
needs the caveat that Huang's own transport model would predict about 15.7 eV
for Fig. 1d, with the measurement 2.5 eV above it.

### 4.5 Shentcis consistency numbers

For WSe₂ (002) with $c=12.98$ Å, beam along $[001]$ and
$\theta_{\rm obs}=121^\circ$, the exact-minus-classical shift is −1.63 eV at
60 keV and −0.69 eV at 300 keV, matching the claim's ≤ 1.6 eV. The validator
did not re-read the Shentcis source geometry.

## 5. Implementation comparison (check script and anchor)

`checks/recoil_in_crystal_527.py` and the anchor
`tests/materials/test_recoil.py::test_huang_2023_harmonic_defect_matches_in_crystal_lines`
were read after §1–4. Both exit green:

```text
harmonic defect: measured 18.19 eV, exact recoil 19.10 eV, classical 0
residual -0.91 +- 0.66 eV (sem 0.44, calibration 0.5); classical excluded at 27 sigma
hopg m=2: line   659.2; U0 -13.40 -> +0.328; delta 1.14e-03 -> +0.090; entry bend -> -0.031
hopg m=4: line  1299.0; U0 -13.40 -> +0.659; delta 2.91e-04 -> +0.045; entry bend -> -0.062
graphite Bloch (largest |U_h| 9.79 eV): line m=2 +0.101, m=4 +0.327, defect -0.125
largest in-medium change of 2E2-E4: 0.134 eV
PASS
1 passed (anchor node)
```

The table comparison, $\sigma$ construction and anchor logic match §4.1
exactly. The anchor's ±0.5 eV offset model is the linear-calibration
assumption of §2. Differences from the derivation:

1. **Photon refraction (first divergent term).** The check uses the bulk
   form `-e0 * delta * beta * u_n / (1 - beta * u_n)`, i.e.
   $\Delta\mathbf K=-\delta\epsilon\hat{\mathbf n}$. That form violates the
   exit-face tangential continuity which the ledger's own "first-order Snell
   decrement" and `xray-in-medium-resonance` impose. The correct term is
   $\Delta\mathbf K=-\delta\epsilon\hat{\mathbf N}/(\hat{\mathbf
   n}\cdot\hat{\mathbf N})$, which gives +0.41 to +0.44 eV on the defect, not
   +0.13 eV. The claimed bound "≤ 0.14 eV on the defect" fails; the check's
   < 1 eV gate still passes.
2. **Entrance-bend sign.** The check computes
   `bent = cos(bend)*BEAM - sin(bend)*cross(cross(axis, BEAM), BEAM)/sin_in`
   with `bend < 0`. Since $(\hat{\mathbf c}\times\hat{\mathbf
   u})\times\hat{\mathbf u}=-\hat{\mathbf c}_\perp$, this rotates the electron
   *away* from the normal: 12.4573° against the correct 12.4427°. The
   magnitude is right, but the sign of the line term flips: −0.031 should be
   +0.031 eV. The effect on the defect is negligible. The code comment
   "toward the normal" is wrong.
3. **Bloch frame.** The in-plane azimuth is an undocumented arbitrary choice
   (§4.3), and the result is not converged in the reflection cutoff.
4. **Mean inner potential.** This term matches: +0.328/+0.659 eV, defect
   −0.004 eV.

## 6. Verdict

- **Claim**: `line-recoil-in-crystal` — `checks/recoil_in_crystal_527.py`
  against `materials/recoil.py::recoil_resonance_eV` — Huang et al. 2023
  Eq. (3)/(5), Methods, SI Table S4, §S5, §S11.
- **Filters**: units pass; limits pass ($\hbar\to0$ gives the classical
  root with $D_c\equiv0$; a linear gain error leaves it at zero); signs and
  conventions pass ($\mathbf g=-\mathbf g_{\rm paper}$ reproduces Eq. (3)).
  Reference data pass (13/13 rows, angles, set assignment, calibration,
  spacings).
- **Re-derivation**: `matches` for the central statement: the zero-phonon
  line centre is the vacuum root; measured 18.19 eV against predicted
  19.10 eV; residual −0.91 ± 0.66 eV; classical excluded. `differs` for the
  quantified correction budget. The first divergent term is the photon-
  refraction wavevector, bulk $-\delta\epsilon\hat{\mathbf n}$ against
  exit-face $-\delta\epsilon\hat{\mathbf N}/(\hat{\mathbf n}\cdot\hat{\mathbf
  N})$, which moves the defect by +0.41–0.44 eV, beyond the stated 0.14 eV.
  The Bloch figure holds for one arbitrary azimuth. The ~3 eV
  scattering/peak-fit bias in Huang's own model is not listed.
- **Verdict**: `discrepancy`, confined to the stated bounds and caveats. The
  measured-versus-exact-root comparison and the classical exclusion rederive.
  After the text corrections below, the expected verdict is `rederived`.
- **Suggested ledger change**: the owner applies this; it is not a sign-off.
  Replace the claim sentence beginning "At 10–15 keV …" through "… two-beam
  Bloch branch shifts." with:

  > At 10–15 keV graphite/hBN (002)/(004) geometries, the omitted in-medium corrections move the harmonic defect `2E2 - E4` by ≤ 0.6 eV: mean inner potential (cancels in the defect, linear in `g`; single lines +0.3 to +0.7 eV), photon refraction with exit-face tangential continuity (`k_eff = omega (n + delta h)`, +0.25 to +0.58 eV on the defect over the 13 rows and both detector-azimuth conventions, ≤ 0.3 eV per line at the sized geometries) and entrance refraction (< 0.001 eV). Two-beam Bloch branch shifts of the free-like branch depend on the unreported in-plane crystal azimuth (median ≈ 0.4–0.5 eV on the defect, up to several eV near a strong Bragg condition); the excitation-weighted branch centroid is unshifted at two-beam order, so at EDS resolution they broaden rather than move the line.

  Replace "The straight-path exact root reproduces Huang et al.'s measured defect, `18.19 ± 0.66` eV measured against `19.10` eV predicted; the classical root gives `0` and is excluded at 27σ." with:

  > The straight-path exact root reproduces Huang et al.'s measured defect, `18.19 ± 0.66` eV measured against `19.10` eV predicted (19.5 eV with exit-face photon refraction, −2.0σ); the straight classical root gives `0` and is excluded at 27σ (≥ 11σ if the ±0.5 eV calibration error is uncorrelated between the two lines). Huang's own scattering Monte Carlo predicts ≈ 15.7 eV (quantum) and ≈ −2.5 eV (classical) for Fig. 1d, so the straight-path agreement does not test their transport/peak-fit bias.

  Change Checks "first-order Snell decrement from `refractive_index`" to
  "first-order exit-face Snell decrement from `refractive_index`". Add
  "peak-fit bias from multiple scattering" to *Not claimed*. Add this
  write-up to the Notes and to `docs/validation/radiation-physics/index.md`.
  The check also needs the photon term and entrance-bend sign fixed (§5).
  Research note §1.2 ("consistent with their statement that scattering moves
  the recoil measurement little") and §2's correction table and Bloch caveat
  need the same corrections.

### Re-check after owner corrections (2026-11-11)

The owner applied the corrections above without committing them, on top of
`2ee11cf13`. The same fresh context re-checked them without editing the
ledger, the check or any code.

- **Ledger row.** The row carries the replacement claim text. Checks reads
  "first-order exit-face Snell decrement" and marks the Bloch estimate
  azimuth-dependent, illustrative and not gated, with the centroid sum rule.
  Not claimed adds peak-fit bias from multiple scattering and the unreported
  $\phi_{\rm obs}$ (18.04 eV at 180°). Notes link this write-up. The status
  stays `unverified` for the owner to set, and the generated
  `domain-inventories.md` view matches the row.
- **Check script.** The photon term is now
  `-e0*delta*beta*u_e/(e.n)/(1-beta*u_n)`, with the outward exit normal
  $\pm\hat{\mathbf c}$ chosen by the sign of $\hat{\mathbf n}\cdot\hat{\mathbf c}$.
  This is identical to the §4.2 exit-face form. The entrance bend now uses
  `+sin(bend)`. The Bloch estimate is excluded from the gate and documented as
  one arbitrary azimuth. Re-run results, all reproducing §4:
  - graphite 11.7 keV: photon +0.294 / +0.148 eV, defect +0.440 eV;
  - entrance bend +0.031 / +0.062 eV;
  - hBN 14.0 keV photon +0.294 / +0.149 eV;
  - largest gated change of the defect 0.440 eV;
  - result `PASS`, rc 0.
- **Research note.** §1.2 carries the Huang Monte Carlo caveat (≈ 15.7 / −2.5
  eV), the correlated-calibration (≥ 11σ) and the $\phi_{\rm obs}$ bullets.
  §2's table and Bloch paragraph match §4.2–4.3 here. §5 states the sum-rule
  limit and the peak-fit bias.
- **Toctree.** `line-recoil-in-crystal` is in
  `docs/validation/radiation-physics/index.md`.
- **Tests.** `tests/dev/test_validation_ledger.py` and
  `tests/materials/test_recoil.py` (including the Huang anchor): 61 passed.

**Final verdict: `rederived`.** The independent derivation, the source checks
and the corrected correction budget now agree with the ledger claim, the check
and the research note. The owner applies the status; human sign-off remains
pending (#277).
