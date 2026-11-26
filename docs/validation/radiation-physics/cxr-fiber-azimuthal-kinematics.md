# Fiber azimuthal kinematics and graphite non-basal structure factors

Validation: `cxr-fiber-azimuthal-kinematics`, `graphite-nonbasal-structure-factor`.

Independent fresh-context verification for issue #1480 (Phase I research; no
production change). The derivations below were written from the ledger rows,
Shi et al. (2024) Eq. (6), International Tables Vol. A (No. 194) and the
production signatures before reading `checks/fiber_nonbasal_cxr_1480.py` or
the [research note](../../research/physics/fiber-nonbasal-cxr-1480.md).
Code comparison, numerics and mutation tests follow. Scratch evidence:
`scratch/v1480_verify.py`, `scratch/v1480_m6.py` (not committed).

## Conventions

$\mathbf g$ has $\lvert\mathbf g\rvert=2\pi/d$ (production
`_reciprocal_basis` carries the $2\pi$). Momentum convention
$\mathbf p_i=\mathbf p_f+\hbar\mathbf g+\hbar\mathbf k$, so the electron gives
up $\hbar\mathbf g$; this is Shi et al. with the reciprocal vector negated.
$E=T+m_ec^2$, $\beta=Pc/E$, $\mathbf Q=\hbar c\,\mathbf g$ (eV),
$D=1-\beta\,\hat{\mathbf u}\cdot\hat{\mathbf n}$. The sample frame has
$\hat{\mathbf z}$ along reciprocal $(001)$; a crystallite at azimuth $\phi$ is
rolled by $R_z(\phi)$ about that axis, so
$\mathbf g=(g_\perp\cos(\phi+\phi_0),\,g_\perp\sin(\phi+\phi_0),\,g_z)$ with
$g_z=2\pi l/c$ (hexagonal: $\mathbf c^*\parallel\mathbf c$).

## `cxr-fiber-azimuthal-kinematics`

### Exact root

Momentum conservation gives $c\mathbf p_f=\mathbf P-\mathbf Q-\epsilon\hat{\mathbf n}$
and energy conservation $E_f=E-\epsilon$. Squaring the mass shell,

$$
(E-\epsilon)^2=M^2+\lVert\mathbf P-\mathbf Q\rVert^2
-2\epsilon(\mathbf P-\mathbf Q)\cdot\hat{\mathbf n}+\epsilon^2
\;\Rightarrow\;
\epsilon=\frac{\mathbf P\cdot\mathbf Q-Q^2/2}{E-\mathbf P\cdot\hat{\mathbf n}+\mathbf Q\cdot\hat{\mathbf n}}
=\frac{\beta\,\hat{\mathbf u}\cdot\mathbf Q-Q^2/2E}{D+\mathbf Q\cdot\hat{\mathbf n}/E}.
$$

This is the claimed root, including the $+\mathbf Q\cdot\hat{\mathbf n}/E$ sign.
For $\hat{\mathbf u}=+\hat{\mathbf z}$ a line exists only for
$\hat{\mathbf u}\cdot\mathbf g=g_z>0$, i.e. $l>0$ with production's
$(001)\mapsto+\hat{\mathbf z}$ orientation (confirmed: production
$\mathbf g_{001}=(0,0,0.936)$ Å⁻¹).

### (i) Axial degeneracy

With $\hat{\mathbf u}=\hat{\mathbf z}$ the classical root
$\epsilon_{\rm cl}=\hbar c\,\beta g_z/D$ depends on $\mathbf g$ only through
$g_z=2\pi l/c$, and $D$ does not involve $\mathbf g$. Every reflection with
the same $l$, at every $\phi$, shares the $(00l)$ classical root.

### (ii) Recoil shift and spread

With $Q^2=(\hbar c)^2(g_\perp^2+g_z^2)$ and
$\mathbf Q\cdot\hat{\mathbf n}=\hbar c\,(g_zn_z+g_\perp n_\perp\cos\Delta)$,
$\Delta=\phi+\phi_0-\psi_n$, expanding to first order in $\hbar c\,g/E$:

$$
\frac{\epsilon}{\epsilon_{\rm cl}}\simeq1
-\frac{\hbar c\,(g_\perp^2+g_z^2)}{2E\beta g_z}
-\frac{\hbar c\,g_zn_z}{ED}
-\frac{\hbar c\,g_\perp n_\perp\cos\Delta}{ED}.
$$

The $(00l)$ line keeps the second and third terms with $g_\perp=0$, and the
two lines share $\epsilon_{\rm cl}$. Relative to its basal partner, a
non-basal line therefore has a $\phi$-independent fractional shift and a
fractional half-range

$$
\boxed{\frac{\delta\epsilon}{\epsilon_{\rm cl}}=-\frac{\hbar c\,g_\perp^2}{2E\beta g_z}},\qquad
\boxed{\frac{\Delta\epsilon_{1/2}}{\epsilon_{\rm cl}}=\frac{\hbar c\,g_\perp\lvert n_\perp\rvert}{ED}}.
$$

The cross term $g_zn_z$ in $\mathbf Q\cdot\hat{\mathbf n}$ is common to the
non-basal line and $(00l)$, so it cancels from the claimed shift. Both claimed
expressions are dimensionless: they are fractions of $\epsilon_{\rm cl}$, not
energies. In eV the shift is $-(\hbar c\,g_\perp)^2/(2ED)$ and the half-range
$\epsilon_{\rm cl}\,\hbar c\,g_\perp\lvert n_\perp\rvert/(ED)$.

### (iii) Oblique incidence

For $\hat{\mathbf u}=(\sin\alpha,0,\cos\alpha)$ the classical numerator is
$\hat{\mathbf u}\cdot\mathbf g=g_z\cos\alpha+g_\perp\sin\alpha\cos(\phi+\phi_0)$
while $D$ is $\phi$-independent. Half-range over mean is exactly
$(g_\perp/g_z)\tan\alpha$ (valid while $g_z\cos\alpha>g_\perp\sin\alpha$).

### (iv) Vanishing recoil

As $\hbar c\,g/E\to0$ the numerator tends to $\beta\hbar c\,\hat{\mathbf u}\cdot\mathbf g$
and the denominator to $D$: the classical root. Ultra-relativistically
($\beta\to1$) both first-order fractions scale as $1/E$.

### Comparison with production

- `materials/recoil.py::_coefficients` divides the conservation polynomial by
  $E$: numerator $\beta\hat{\mathbf u}\cdot\mathbf Q-Q^2/2E$, denominator
  $(M/E)^2/(1+\beta)+\tfrac12\beta\lVert\hat{\mathbf u}-\hat{\mathbf n}\rVert^2+\mathbf Q\cdot\hat{\mathbf n}/E$.
  The first two terms equal $1-\beta\hat{\mathbf u}\cdot\hat{\mathbf n}$
  identically (cancellation-safe form). Matches.
- `montecarlo/spectrum/cxr/_recoil.py::_line_recoil_terms` returns
  $r_{\rm num}=\hbar c\,g^2/2E$, $r_{\rm den}=\hbar c\,\hat{\mathbf n}\cdot\mathbf g/E$;
  the route forms $\omega=(\mathbf v\cdot\mathbf g-r_{\rm num})/(D+r_{\rm den})$ in Å⁻¹
  (e.g. `_recoil.py:139-140`, `_kernels.py::_cxr_kin_core`). Multiplying by
  $\hbar c$ gives the same root. Setting `_RECOIL_HBARC_EV_ANG = 0` zeroes
  both terms and leaves the classical root exactly.
- `materials/crystal.py::_orientation_R(..., surface_hkl=(0,0,1), azimuth_rad=phi)`
  rotates $\mathbf c^*$ onto $+\hat{\mathbf z}$, then applies $R_z(\phi)$:
  a roll about the fiber axis, as assumed.

Numbers (independent script, own $\hbar c$, own root; production orientation):

| quantity | independent | claimed / check |
|---|---:|---:|
| degeneracy, all 4 polar × 2 azimuthal angles, 37 $\phi$, 30 keV and 1 MeV | $4.4\times10^{-16}$ | $4.4\times10^{-16}$ |
| own exact root vs `recoil_resonance_eV`, 2000 random geometries 10 keV–10 MeV | $2.4\times10^{-14}$ | — |
| 30 keV (102), 119°: exact (mean $-\epsilon_{00l}$)/$\epsilon_{\rm cl}$ vs first order | $-0.02579$ vs $-0.02578$ | — |
| 30 keV (112), 119°: same | $-0.07737$ vs $-0.07733$ | — |
| 1 MeV (102)/(112): same | $-0.00322$/$-0.00965$ vs $-0.00322$/$-0.00966$ | — |
| 30 keV (102) half-range, exact vs first order | $0.00787$ vs $0.00811$ | $0.0079$ vs $0.0081$ |
| 30 keV (112) half-range, exact vs first order | $0.01290$ vs $0.01405$ | $0.0129$ vs $0.0141$ |
| oblique, (102), 1° / 5° | $0.0275$ / $0.1377$ | $0.0275$ / $0.1377$ |

The (112) half-range differs from its first-order value by 8 % at 30 keV
($\hbar c\,g/E\approx0.02$); this is second-order and the claim says
"first-order".

### Findings on the claim text and check

1. **Units of (ii).** The claim gives the shift $-\hbar c\,g_\perp^2/(2E\beta g_z)$
   and half-range $\hbar c\,g_\perp\lvert n_\perp\rvert/(ED)$ without saying
   they are fractions of the classical root; read as energies they are
   dimensionally wrong. Also state that the shift is relative to the $(00l)$
   line (the common $g_z^2$ and $g_zn_z$ recoil terms cancel only there).
2. **The check's printed "shift" is a different quantity.** `cmd_kinematics`
   compares (mean exact)/$\epsilon_{\rm cl}-1$ with the full first-order
   expression $-\hbar c\,g^2/(2E\beta g_z)-\hbar c\,g_zn_z/(ED)$ (total recoil
   shift vs classical, $-0.0334$ vs $-0.0333$). That contains the claimed
   term, but the ledger "Checks" text presents it as the claimed shift. The
   claimed term itself is confirmed above to $4\times10^{-5}$.
3. **Shift is printed, not asserted.** Mutation M6 (flip the sign of
   $\mathbf Q\cdot\hat{\mathbf n}$ in both `recoil_resonance_eV` and the route)
   passes every check: the shift moves from $-0.0333$ to $-0.0389$ at 30 keV
   but only the half-range, which is symmetric to first order, is asserted.
   The sign is still covered here and by `line-recoil-resonance`, not by
   this anchor.
4. **Degeneracy loop scope.** In section 1 `worst = 0.0` is reset inside the
   `psi` loop, so each `check` covers only the last angle
   ($\theta=160^\circ$, $\psi=0.7$). Mutation M3 (1 % break at $\theta=20^\circ$
   only) passes. The ledger's "four polar and two azimuthal observation
   angles" overstates the asserted coverage. The physics holds at all angles
   (table above).

Mutations that are detected: M1, flip the $\mathbf Q\cdot\hat{\mathbf n}$ sign
in the route only (route-vs-recoil parity fails, $9.7\times10^{-2}$); M2,
double $Q^2/2E$ in the route ($1.7$); M4, roll about $\hat{\mathbf x}$ instead
of $\mathbf c^*$ (12 failures).

## `graphite-nonbasal-structure-factor`

### Basis is Bernal graphite

ITA No. 194 graphite: C at 2b $(0,0,\tfrac14),(0,0,\tfrac34)$ and 2c
$(\tfrac13,\tfrac23,\tfrac14),(\tfrac23,\tfrac13,\tfrac34)$. Shifting the origin by
$-\mathbf c/4$ gives $(0,0,0),(0,0,\tfrac12),(\tfrac13,\tfrac23,0),(\tfrac23,\tfrac13,\tfrac12)$,
which is exactly the `hopg.cif` P1 basis. An origin shift changes only the
phase of $S$. $a=2.461$, $c=6.711$ Å agree with graphite's accepted
$a\approx2.46$, $c\approx6.71$ Å.

### Geometric structure factor

With $\omega=e^{2\pi i/3}$ and $m=h-k$: $h+2k\equiv m$ and $2h+k\equiv-m$
(mod 3), so

$$
S=1+\omega^{m}+(-1)^l\left(1+\omega^{-m}\right).
$$

For $m\equiv0$, $S=2+2(-1)^l$: $\lvert S\rvert^2=16$ (even $l$), $0$ (odd $l$).
For $m\not\equiv0$, $1+\omega^{\pm1}=e^{\pm i\pi/3}$, so
$S=2\cos(\pi/3)=1$ (even $l$) or $S=\pm2i\sin(\pi/3)$, $\lvert S\rvert^2=3$
(odd $l$). This matches the claim. It reproduces the graphite-2H powder
pattern: 100, 101, 102, 103, 004, 110, 112 present; 001, 003, 111, 113
absent (the $hh\bar{2h}l$, $l=2n$ condition of the $c$ glide, and $00l$,
$l=2n$ from $6_3$). An exhaustive scan $\lvert h\rvert,\lvert k\rvert\le4$,
$\lvert l\rvert\le6$ finds no violation for either basis.

### Fiber families and $g_\perp$

$\lvert\mathbf a^*\rvert=4\pi/(a\sqrt3)=2.948$ Å⁻¹; the $\{11\}$ and $\{20\}$ rows
are $\sqrt3$ and $2$ times that: $5.106$ and $5.896$ Å⁻¹. $g_z=2\pi l/c$
(production $\lvert\mathbf c^*\rvert=0.93625$ Å⁻¹). The 60° orbits
$\{10\}$, $\{11\}$, $\{20\}$ each have six members with one $m$ class, so
equal $\lvert S\rvert$, $g_z$, $g_\perp$. **Scope caveat:** for general
$(h,k)$ with $h\ne k$, $hk\ne0$ (first case $\{21\}$, $g_\perp=7.80$ Å⁻¹) the
mirror-related orbits $\{hk\}$ and $\{kh\}$ share $l$, $g_\perp$ and
$\lvert S\rvert$, so a fiber family grouped that way has 12 members. "Every
non-basal fiber family has six members" holds only for
$\lvert g_\perp\rvert<7.8$ Å⁻¹, including the claimed $\lvert\mathbf g\rvert\le6$ Å⁻¹ table.

Independent $\lvert S\rvert^2$ with ITC Vol. C Table 6.1.1.4 Cromer–Mann carbon
$f_0$: (002) 299.48, (102) 7.17, (112) 54.23, (101) 26.84, (103) 16.20,
(100) 9.74, (110) 59.29. These agree with the ledger values to 0.3 %; the
residual is the production form-factor parameterization.

### Anisotropic Debye–Waller

For a harmonic site with $\langle u_x^2\rangle=\langle u_y^2\rangle=U_\perp$,
$\langle u_z^2\rangle=U_\parallel$, the amplitude factor is
$e^{-W}$, $W=\tfrac12\langle(\mathbf g\cdot\mathbf u)^2\rangle=\tfrac12(U_\perp g_\perp^2+U_\parallel g_z^2)$.
Production `debye_waller(g, B)` returns $\exp[-B(g/4\pi)^2]$, the amplitude
form with $B=8\pi^2U$ in the isotropic case. Equating the exponents,

$$
B_{\rm eff}=\frac{8\pi^2\left(U_\perp g_\perp^2+U_\parallel g_z^2\right)}{g^2},
$$

as claimed. This is exact per $\mathbf g$ when all sites share one tensor.
Numerically, `debye_waller(g, B_eff)` equals $e^{-W}$ to machine precision
for (112), (100), (002). For $g\parallel c$, $B_{\rm eff}=8\pi^2U_\parallel=1.303$ Å²,
consistent with the catalog $B_{33}=1.30$ Å². $U_\perp=0.0031$,
$U_\parallel=0.0165$ Å² match `data/channeling_thermal.toml` `[hopg.C]`. That
provenance states $U_{11}$ is fixed from Ludsteck (1972) in Trucano & Chen,
not refined, and $U_\parallel$ is the mean of 0.016(2) and 0.017(2). The
research note's attribution "Trucano & Chen 1975" should carry that
qualification. The primary paper was not re-read here.

### Check behaviour

`structure hopg --g-max 6` reproduces every value above. It has **no
assertion**: mutation M5 (layer-B atom moved from $(\tfrac23,\tfrac13,\tfrac12)$
to $(\tfrac13,\tfrac23,\tfrac12)$) breaks the 16/0/1/3 rule on (100), (200), (101),
(102)… and the command still exits 0. It is a diagnostic table, as the ledger
says, not a gate.

## Verdict

- `cxr-fiber-azimuthal-kinematics`: **rederived**. Every physics item
  matches. The claim text should label the (ii) expressions as fractions of the
  classical root relative to $(00l)$. The anchor should assert the shift and
  fix the section-1 loop scope before the claim is promoted to `anchored`.
- `graphite-nonbasal-structure-factor`: **rederived**. Add the six-member
  scope ($\lvert g_\perp\rvert<7.8$ Å⁻¹) and the $U_{11}$ provenance note.
  The check is not an asserting anchor.

Residual scope: vacuum roots only (the Snell correction is common to all
reflections); no intensity or yield claim is verified; Trucano & Chen's paper
was not re-read; h-BN is not covered.
