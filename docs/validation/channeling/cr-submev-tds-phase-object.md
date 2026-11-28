# `cr-submev-tds-phase-object` re-derivation

Claim: `checks/channeling_submev_w001_tds_phase.py::{phase_moments,rates}`.
Recorded in the {ref}`strong-phase TDS section <sec-submev-w001-tds-phase>`.
Research record for #585; it does not change the production kernel.

Fresh context, 2026-11-16, commit `57853df9c`. The derivation below was
written from the ledger row, the doc section and the `axial_dissipator`
module docstring before the check's function bodies were read. It builds on
the secular, flat-band coherence equation rederived in
`cr-submev-axial-coherence-width`.

## One displaced atom as a quantum channel

An atom of string $s$ at $\mathbf R$ with in-plane displacement $\mathbf u$
adds the eikonal phase $\phi_{\mathbf u}(\mathbf r)=-V_s(\mathbf r-\mathbf R-\mathbf u)/\hbar v$
when the electron crosses its slice. The continuum Hamiltonian already holds
the thermal mean $\langle\phi\rangle(\mathbf r)=E_{\mathbf u}\phi_{\mathbf u}(\mathbf r)$.
The residual transmission is therefore the multiplication operator
$T_{\mathbf u}=e^{i\delta\phi_{\mathbf u}}$, with
$\delta\phi_{\mathbf u}=\phi_{\mathbf u}-\langle\phi\rangle$ and
$\mathbf u\sim\mathcal N(0,a_s^2\mathbf 1)$. Averaged over the Einstein
displacement, one atom applies the completely positive, trace-preserving map

$$
\Phi_s(\rho)=E_{\mathbf u}\bigl[T_{\mathbf u}\,\rho\,T_{\mathbf u}^\dagger\bigr].
$$

**Matrix elements.** $T_{\mathbf u}$ is multiplicative. In the string-localized
flat-band (Wannier) basis of `cr-submev-axial-coherence-width`, its diagonal
element is

$$
\langle n\vert T_{\mathbf u}\vert n\rangle
=\int\lvert w_n\rvert^2e^{i\delta\phi_{\mathbf u}}\,d^2r\equiv M_n(\mathbf u).
$$

Off-site elements $\langle n,\mathbf S\vert T-1\vert n,\mathbf R\rangle$ are
overlaps of two different-site orbitals weighted by the atom-local
$T-1$, so they are of tunnelling size. That is the same neglect as in the
second-order record.

**Secular projection.** For $\sigma=\langle f\vert\rho\vert i\rangle$,
$\langle f\vert\Phi(\rho)\vert i\rangle=\sum_{mn}E[\langle f\vert T\vert m\rangle\rho_{mn}\langle i\vert T\vert n\rangle^*]$.
Keep only the pair $(m,n)=(f,i)$. Other pairs either oscillate at a
different frequency, or are the $2p_x/2p_y$ partner. The partner term
$E[\langle 2p_y\vert T\vert 2p_x\rangle\,M_{1s}^*]$ is odd under the
reflection $x\to-x$ applied to $\mathbf u$ and to the integration variable.
The Gaussian is even, so it vanishes for every source, not just after the
$C_{4v}$ sum. Then

$$
\sigma\to E_{\mathbf u}\bigl[M_f M_i^*\bigr]\,\sigma .
$$

Likewise $\rho_{nn}\to E\lvert M_n\rvert^2\rho_{nn}$ plus feeding from other
populations, which is secular and is *not* included. So this is the
retained population, as the ledger says. The coherent (mean) wave
$\psi\to E_{\mathbf u}[T_{\mathbf u}]\psi$ has amplitude factor
$E_{\mathbf u}M_n$ on level $n$ and intensity factor $\lvert E_{\mathbf u}M_n\rvert^2$.

**Composition.** Atoms of one string sit at spacing $1/\lambda_s$ along the
axis. Their Einstein displacements are independent, and the coherent
evolution between atoms is diagonal in the eigenbasis. Under the secular
projection each factor is a scalar on the tracked element, so
$N=\lambda_s z$ atoms give the factor $c_s^N$. The order of atoms from
different strings does not matter. The rate per unit depth of the modulus is
therefore exactly

$$
\boxed{\;
G_n=-2\sum_s\lambda_s\ln\lvert E_{\mathbf u}M_n\rvert,\quad
P_n=-\sum_s\lambda_s\ln E_{\mathbf u}\lvert M_n\rvert^2,\quad
\kappa=-\sum_s\lambda_s\ln\bigl\lvert E_{\mathbf u}M_fM_i^*\bigr\rvert\;}
$$

as claimed. The $-\lambda\ln$ form is right for periodic atoms. A Poisson
(random-depth) medium would give $\lambda(1-\mathrm{Re}\,c)$ instead. The two
agree to second order but differ by 15 % already at $1-c=0.25$, which is the
scale of the 1s factors at 30 keV ($G_{1s}/\lambda\approx0.57$ summed over
strings). The string geometry fixes $-\lambda\ln$.

**Dropped phase.** $\arg E_{\mathbf u}M_n$ and $\arg E_{\mathbf u}M_fM_i^*$ are
level and line shifts. They are removed by the modulus. The split
$\delta\phi=\phi-\langle\phi\rangle$ puts the first cumulant in $H$. The
exact coherent-wave potential also has a real part from the odd cumulants
($E e^{i\delta\phi}=\exp(-\tfrac12\kappa_2-\tfrac i6\kappa_3+\dots)$ pointwise).
That is third order in $\delta\phi$ and absent from the second-order
dissipator, so the split is consistent with it. In the strong-phase regime
the shift is not small, though (numerical section below).

## Second-order expansion

Write $A_n=\int\lvert w_n\rvert^2\delta\phi_{\mathbf u}$ and
$B_n=\int\lvert w_n\rvert^2\delta\phi_{\mathbf u}^2$. Then
$M_n=1+iA_n-\tfrac12B_n+O(\delta\phi^3)$, and $E_{\mathbf u}A_n=0$ because
$E_{\mathbf u}\delta\phi_{\mathbf u}(\mathbf r)=0$ pointwise.

- $E M_n=1-\tfrac12E B_n$, so $G_n=\sum_s\lambda_sE B_n=\int\lvert w_n\rvert^2C_\mathrm{grid}(\mathbf r,\mathbf r)=\Gamma_n^\mathrm{grid}$.
- $E\lvert M_n\rvert^2=1-E B_n+E A_n^2$, so $P_n=\Gamma_n^\mathrm{grid}-J_{nn}$.
- $E M_fM_i^*=1+iE(A_f-A_i)-\tfrac12E(B_f+B_i)+E A_fA_i$, and the imaginary
  part vanishes. So
  $\kappa=\tfrac12(\Gamma_f+\Gamma_i)-J_{fi}$ with
  $J_{fi}=\sum_s\lambda_sE[A_fA_i]$. That is $\kappa_\mathrm{TDS}$ of
  `cr-submev-axial-coherence-width`, with $L_{\mathbf u}=\delta\phi_{\mathbf u}$
  and measure $\lambda_sp(\mathbf u)d^2u$ as in the dissipator docstring.

All three reproduce the dissipator's in-grid TDS rates. The large-angle sink
$C-C_\mathrm{grid}$ is absent on both sides, because `tds_phase` is
band-limited. **Units:** $M_n$ is dimensionless, $\lambda_s$ is in Å⁻¹, and the
rates are in Å⁻¹.

## Comparison with the implementation

| Item | Derivation | Code | Result |
| --- | --- | --- | --- |
| $\delta\phi_{\mathbf u}$ | band-limited $\phi_{\mathbf u}-E_{\mathbf u}\phi_{\mathbf u}$ | `tds_phase`: spectrum $\times(e^{-i\mathbf k\cdot\mathbf u}-e^{-a^2k^2/2})$, real part | match |
| measure | $\mathcal N(0,a_s^2\mathbf 1)$ | `displacement_quadrature`: `hermegauss` outer product $/2\pi$, nodes $\times\sqrt{a^2}$ (as `lw.tds_gain`) | match |
| $M_n$ | $\int\lvert w_n\rvert^2e^{i\delta\phi}$ | `einsum(exp(1j*dphi), densities) * area`; densities sum to 1 | match |
| $G_n$ | $-2\sum\lambda\ln\lvert EM_n\rvert$ | `exact_coherent` | match |
| $P_{1s}$ | $-\sum\lambda\ln E\lvert M_{1s}\rvert^2$ | `exact_retained[0]` | match |
| $\kappa$ | $-\sum\lambda\ln\lvert EM_{1s}M_{2p}^*\rvert$ | `cross` $=E\,M_0\bar M_n$; `exact_kappa[1]` | match |
| 2nd order | $\Gamma_n$, $\Gamma_{1s}-J_{1s1s}$, $\tfrac12(\Gamma_{1s}+\Gamma_{2p})-J$ | `var`, `var[0]-lin[0]`, `0.5*(var[0]+var)-lin` | match |
| sources | every string and supercell image | loop over `tds_sources` (9 strings) | match |
| widths | FWHM $2\hbar v\kappa$ | `2.0 * hbar_v * kappa` | match |

**Chunking.** Nodes and weights are sliced together (`CHUNK = 64`) and every
moment is accumulated as `w @ (...)`, so chunking is exact.
**Columns.** `densities[:2]` is $(1s, 2p_x)$. Column 0 of `cross` is
$E\lvert M_{1s}\rvert^2$, and the code comment notes that
`exact_kappa[0]` is the 1s population loss. `screen_energy` reads
`exact_kappa[1]`, `linear_kappa[1]`, `exact_population_loss[0]` and
`linear_population_loss_1s`. All are consistent with $0=1s$, $1=2p_x$.
`linear_population_loss_1s` $=$ `var[0]-lin[0]` $=\Gamma^\mathrm{grid}_{1s}-J_{1s1s}$.
That is the second-order counterpart of $P_{1s}$, not a loss of the
coherent wave. Using $2p_x$ alone is justified by the reflection argument
above.

## Analytic tests

`tests/check/test_channeling_submev_w001_tds_phase.py` (3 passed) uses point
densities on two pixels and a fake `tds_phase`.

- **Linear phase** $\delta\phi=c_nu_x$. $M_n=e^{ic_nu_x}$ is Gaussian, so
  $E M_n=e^{-c_n^2a^2/2}$ and $G_n=\lambda c_n^2a^2=$ `var`. $\lvert M_n\rvert=1$
  gives $P_n=0$. $\kappa=\tfrac12\lambda(c_0-c_1)^2a^2$, and the
  second-order form gives the same value. The expected values are correct.
  The test checks the $\sqrt{a^2}$ scaling, the weight normalization, the
  `cross` and `lin` column pairing, and the $\tfrac12$ in $\kappa$.
- **Quadratic uniform phase** $\epsilon(u_x^2-a^2)$. $u_x^2/a^2\sim\chi^2_1$,
  so $Ee^{i\delta\phi}=e^{-i\epsilon a^2}(1-2i\epsilon a^2)^{-1/2}$, and
  $G=\tfrac12\lambda\ln(1+4\epsilon^2a^4)$ against
  $\lambda\,\mathrm{Var}=2\lambda\epsilon^2a^4$. A uniform phase gives
  $M_0=M_1$ and $\lvert M\rvert=1$, so $\kappa=P=0$. The test is correct, and
  at $\epsilon=30$ ($\epsilon a^2=1.2$) it tests genuine saturation of $G$
  (ratio 0.33).
- **Gap (minor).** The strong-phase forms of $P$ and $\kappa$ are tested only
  where they are trivial (zero) or Gaussian (equal to second order). No
  test has a non-Gaussian, spatially varying phase. One closed form is a
  density split equally over two pixels with $\delta\phi=\pm cu_x$:
  $M=\cos cu_x$ and $E\lvert M\rvert^2=\tfrac12(1+e^{-2c^2a^2})$, so
  $P=-\lambda\ln\tfrac12(1+e^{-2c^2a^2})$, while second order gives
  $\lambda c^2a^2$. That would pin `mean_m2` beyond the Gaussian case.

## Numerical spot check

Own probe in `/tmp` (not committed). It calls `phase_moments` and `rates`,
adds an independent Monte Carlo over $\mathbf u$ (seeded normal draws,
which avoid Gauss–Hermite aliasing), and reports
$\hbar v\lambda\arg E[M_{1s}M_{2p}^*]$.

| Case | $\kappa$ exact (Å⁻¹) | $\kappa$ 2nd (Å⁻¹) | $P_{1s}$ exact / 2nd (Å⁻¹) | Note |
| --- | --- | --- | --- | --- |
| 300 keV, 128 pts, GH 24 | 0.035917 | 0.042071 | 0.059388 / 0.070604 | record exactly |
| 300 keV, 128 pts, MC 1000 per source | 0.03577 | — | 0.05926 | agrees with GH to 0.4 % |
| 30 keV, 128 pts, GH 24 | 0.097454 | 0.173796 | 0.165846 / 0.316617 | record exactly |
| 30 keV, 96 pts, GH 24 / 36 / 48 / 64 | 0.0975 / 0.0940 / 0.0957 / 0.0966 | 0.1722 | 0.167 / 0.161 / 0.163 / 0.165 | oscillates, ±2 % |
| 30 keV, 96 pts, MC 2000 per source | 0.0953 | — | 0.1631 | within that spread |

The second-order quadrature reproduces the dissipator's in-grid loss
(`quadrature_vs_grid_loss_rel` below $5\times10^{-7}$ at 128/24). The full
second-order $\kappa_\mathrm{TDS}$, in-grid plus sink, is grid-independent:
0.1738 + sink here, against 0.1759 in `w001-linewidth.json` at 25 points.
That cross-checks the two records.

**Grid logic.** The phase object only sees band-limited phase. The
out-of-band part is a sink whose second-order loss is 1.1–1.2 % of
$\Gamma_{1s}$ at 128 points. For a perturbation that small, its strong-phase
effect is at most about its second-order size. So the in-grid result misses
at most about $\tfrac12\Gamma^\mathrm{sink}$ in $\kappa$: $+0.0019$ Å⁻¹, or
+2 %, at 30 keV. Refining to about 1 % is adequate for a 1.8× claim. Note
that neither the exact nor the second-order $\kappa$ in the record includes
that sink.
**96 points.** The non-convergence is in the displacement quadrature, not
the grid. At fixed 96 points, κ oscillates non-monotonically with the
Gauss–Hermite order, and Monte Carlo falls inside the spread. The 96-point
GH-24 value equals the 128-point one to 0.1 %. The doc's wording "the values
are not converged" is correct. "Under-resolved" in the record `meta` is the
less accurate word: the 96-point grid is resolved, but its quadrature is
noisy. A plausible mechanism is band-edge (Gibbs) content of the truncated
core phase. It puts high-frequency $\mathbf u$ dependence into
$e^{i\delta\phi}$, which Gauss–Hermite aliases. At 128 points the band edge
sits lower on the spectrum. The `quadrature_vs_grid_loss_rel` values of
about $10^{-3}$ at 96 points, against $\le3\times10^{-4}$ at 128, fit this
picture. The conclusion does not depend on it, because every 96-point
variant lies within 3.5 % of the adopted value.

**Doc table and bullets** (against the record's 128/24 rows and
`w001-linewidth.json`). Electronic $\kappa_\mathrm{el}=\kappa-\kappa_\mathrm{TDS\,only}$
is 0.00839, 0.00309 and 0.00152 Å⁻¹. It equals
$\tfrac12(\Gamma^\mathrm{el}_{1s}+\Gamma^\mathrm{el}_{2p})-J_\mathrm{el}$ from
the same record.

| T | ratio$^{-1}$ | FWHM total (eV) | /gap | photon FWHM / line (eV) | $1/P_{1s}$ exact / 2nd (Å) |
| --- | --- | --- | --- | --- | --- |
| 30 keV | 1.78 | 137.2 | 0.606 | 204 / 337 | 6.03 / 3.16 |
| 100 keV | 1.34 | 117.0 | 0.499 | 259 / 519 | 11.3 / 8.19 |
| 300 keV | 1.17 | 114.7 | 0.460 | 513 / 1117 | 16.8 / 14.2 |

These all match the doc table, the "1.8×/1.3×/1.2×" ratios, the 6.0 Å and
3.2 Å lengths, and "about 205 eV" and "510 eV". The 2p per-atom variance
"below 0.1" is 0.076 (`per_atom_tds_loss`). At 128 points the in-grid 1s loss
is $1.03\lambda$, so "per-atom phase variance near 1 rad²" is accurate at this
grid.

## Caveats

- **Line shift dropped (finding, minor).** The phase-object coherence factor
  has $\hbar v\sum_s\lambda_s\lvert\arg E[M_{1s}M_{2p}^*]\rvert\approx28$ eV at
  30 keV (128/24) and 14 eV at 300 keV. That is 12 % and 6 % of the
  $k=0$ gap. It is a strong-phase (third-cumulant) shift of the 2p–1s
  spacing that the second-order kernel cannot produce. Dropping it via the
  modulus is legitimate for the width. But the FWHM/gap ratio then uses an
  unshifted gap: at 30 keV it would be about 0.54 or 0.70, depending on
  the sign, which I did not pin. Report the shift in the record, and state
  in the doc that FWHM/gap carries this ±12 % uncertainty at 30 keV. The
  "unresolved" conclusion holds either way.
- **Sink not added (minor).** The adopted $\kappa$ omits
  $\tfrac12\Gamma^\mathrm{sink}\approx0.0019$ Å⁻¹ at 30 keV (+2 % on the
  total FWHM; 0.606 to about 0.62). Add it as pure loss, which is valid
  per `cr-submev-axial-coherence-width`, or say it is omitted.
- **Fresnel estimate (wording).** With $u_\mathrm{rms}=\sqrt{a^2}=0.0449$ Å
  per axis ($a^2=0.002016$ Å²), $k=90.0$ Å⁻¹ at 30 keV and $d=0.3$ Å, I get
  $q^2d/2k=0.83$ rad, not 0.7. With the 2D $u_\mathrm{rms}$ it is 0.41.
  The point is the same, O(1): the eikonal per atom is not safe at 30 keV.
  Quote 0.8 or name the $u_\mathrm{rms}$ convention.
- **Untested inputs.** Eikonal per atom, the projected Mott–Bethe potential
  and the Einstein model are listed as untested in the check docstring, the
  doc and the ledger Notes. That is honest. The ledger also says electronic
  excitation is not included. The doc's "+ electronic" column adds it
  linearly from the second-order record. That is consistent, because the
  electronic phase per event is small (`cr-submev-axial-coherence-width`).
- **Retained population.** $P_n$ excludes refilling from other levels. The
  doc calls it "the population retained in $n$" and compares it with the
  occupation run's 7 Å, "which also includes refilling". That is honest.
- **Conclusion.** "The second-order kernel overstates the TDS width by
  1.8×/1.3×/1.2×" follows from the record within the stated model (ratio
  stable to 0.1 % in GH, Monte Carlo agrees, sink about 1 %). "FWHM/gap
  0.46–0.61, line still unresolved" follows too. Even with the dropped
  shift and the sink, the ratio stays between about 0.45 and 0.72. It is
  conditional on the eikonal per atom, which is the outstanding test.

## Verdict

`rederived`. The three phase-object rates follow exactly from the
displacement-averaged per-atom channel. The pieces are: diagonal flat-band
matrix elements, secular projection (the $2p_x/2p_y$ cross term vanishes by
reflection per source), and periodic independent atoms. Their second-order
expansions are $\Gamma^\mathrm{grid}_n$, $\Gamma^\mathrm{grid}_n-J_{nn}$ and
$\kappa_\mathrm{TDS}$. The code's measure, columns, chunking and moment
algebra match. The record reproduces exactly at 300 keV and 30 keV, and an
independent Monte Carlo agrees. The findings are minor: the dropped line
shift is unreported, the sink is not added to the adopted width, the Fresnel
estimate is 0.83 rather than 0.7, "under-resolved" in the record meta should
read quadrature noise, and the strong-phase population test is missing. No
implementation change is required for the claim.
