# `brem-lattice-coherent-fraction`

Fresh-context independent verification for issue #1250. The sections through
"Suggested ledger change" verified the proposal; the
[production verification](#production-verification-2026-11-17) at the end
checks the production estimator against it.

## Claim and scope

- **Claim:** for a randomly oriented crystal, the orientation-averaged CBS line
  yield of the explicit reflections $\lvert\mathbf g\rvert\le G$ equals, to first
  Born order in $U_{\mathbf g}$, the Debye–Waller-weighted (lattice-coherent) part
  of the screened-nucleus Born bremsstrahlung below momentum transfer $G$.
  Dropping $e^{-2W}$ and the cutoff gives the amorphous Born bremsstrahlung from
  the same amplitude. Explicit lines up to $G$ therefore double-count a fraction
  $R(G)$ of a tabulated amorphous bremsstrahlung (BremsLib).
- **Code:** proposed only. The evidence is the author's draft
  (`scratch/p2-derivation.md`) and its quadrature (`scratch/shell1250.py`,
  `scratch/share1250.py`), which call production helpers from
  `montecarlo/spectrum/cxr/_recoil.py`.
- **Source:** the production line weight (`coherent-line-spectrum`), the
  production CBS amplitude $A_{\rm CBS}=(U_{\mathbf g}/mc^2)f_{\rm CBS}$
  (`cbs-amplitude`), and the Ter-Mikaelian coherent/incoherent split of crystal
  bremsstrahlung by $e^{-2W}$.

**Independence.** The derivation below was written from the two upstream ledger
rows and their validated write-ups (amplitude forms and line weight only),
before the author's draft or scripts were read. Implementation bodies were read
afterwards, only to diagnose the numerical bias.

## Independent derivation

Gaussian units with $c=1$. Wavenumbers are in Å$^{-1}$, $\hbar c=1973.27$ eV·Å,
$\omega=E/\hbar c$, the electron velocity is $\boldsymbol\beta=\beta\hat{\mathbf v}$,
and the photon direction is $\hat{\mathbf n}$, with $\mathbf k=\omega\hat{\mathbf n}$.

### 1. Energy-integrated line yield of one segment

From `coherent-line-spectrum`, one straight segment of duration $t_L=L/\beta$
and one reflection give

$$
\frac{d^2N_{\mathbf g}}{dE\,d\Omega}
=\frac{\alpha\,\omega_{\rm res}}{4\pi^2\hbar c}\,\Sigma_A\,
t_L^2\operatorname{sinc}^2\!\left(\frac{\Phi'\,t_L\,(\omega-\omega_{\rm res})}{2\pi}\right),
\qquad
\Sigma_A=\sum_{s=1,2}\lvert A_s\rvert^2 ,
$$

where $\Phi'$ is the derivative of the phase-mismatch rate with respect to $\omega$.
The classical phase rate is $\omega D_0-\mathbf v\cdot\mathbf g$, with
$D_0=1-\boldsymbol\beta\cdot\hat{\mathbf n}$, so $\Phi'=D_0$. With exact
plane-wave kinematics the mismatch is $(E_i-E_f-\hbar\omega)/\hbar$, with
$\mathbf p_f=\mathbf p_i-\hbar(\mathbf g+\mathbf k)$, and

$$
\Phi'=1-\mathbf v_f\cdot\hat{\mathbf n}
=\frac{E\big(1-\boldsymbol\beta\cdot\hat{\mathbf n}+\hbar c\,\hat{\mathbf n}\cdot\mathbf g/E\big)}{E-\hbar\omega}
\equiv\kappa D,
\qquad
\kappa=\frac{E}{E-\hbar\omega},\quad
D=1-\boldsymbol\beta\cdot\hat{\mathbf n}+\frac{\hbar c\,\hat{\mathbf n}\cdot\mathbf g}{E}.
$$

This is the golden-rule density-of-final-states Jacobian. Here $E$ is the total
electron energy. Energy conservation $E_f=E-\hbar\omega$ gives the exact
resonance

$$
\omega_{\rm res}=\frac{\mathbf v\cdot\mathbf g-\hbar c\,g^2/(2E)}{D},
$$

because $k^2=\omega^2$ cancels in vacuum. With $\int\operatorname{sinc}^2u\,du=1$,
the time factor becomes
$t_L^2\operatorname{sinc}^2(\cdot)\to(2\pi t_L/\kappa D)\,\delta(\omega-\omega_{\rm res})$, so

$$
\boxed{\;
\frac{dN_{\mathbf g}}{d\Omega}=\frac{\alpha\,\omega_{\rm res}\,t_L}{2\pi\,\kappa D}\,\Sigma_A,
\qquad
\frac{d^3N_{\mathbf g}}{dE\,d\Omega\,dL}
=\frac{\alpha\,\omega\,\Sigma_A}{2\pi\,\beta\,\kappa D}\,\delta(E-E_{\rm res})\;}
$$

per unit path length $L=\beta t_L$. The result is dimensionless per sr, or per eV per sr per Å.

### 2. Orientation average at fixed $\lvert\mathbf g\rvert$

For a randomly oriented crystal, $\hat{\mathbf g}$ is uniform on the sphere relative
to the fixed pair $(\hat{\mathbf v},\hat{\mathbf n})$. Write
$\mu=\hat{\mathbf g}\cdot\hat{\mathbf v}$ and let $\phi$ be the azimuth about
$\hat{\mathbf v}$.

**Classical limit** ($\hbar\to0$: $\kappa=1$, $D=D_0$,
$E_{\rm res}=\hbar c\,\beta g\mu/D_0$):

$$
\Big\langle\delta(E-E_{\rm res})\Big\rangle_{\hat{\mathbf g}}
=\frac12\int_{-1}^{1}d\mu\,\delta\!\Big(E-\frac{\hbar c\,\beta g\mu}{D_0}\Big)
=\frac{D_0}{2\hbar c\,\beta g}\,\Theta\big(\beta g-\omega D_0\big),
$$

$$
\boxed{\;
\Big\langle\frac{d^3N_{\mathbf g}}{dE\,d\Omega\,dL}\Big\rangle_{\hat{\mathbf g}}
=\frac{\alpha\,\omega\,\langle\Sigma_A\rangle_\phi}{4\pi\,\beta^2\,\hbar c\,g},
\qquad g\ge q_{\min}=\omega D_0/\beta .\;}
$$

The classical amplitude is the braced $A_{\rm CBS}$ of `cbs-amplitude`. With
$\mathbf G=(\mathbb 1-\boldsymbol\beta\boldsymbol\beta)\cdot\mathbf g$ and
$\mathbf v\cdot\mathbf g=\omega D_0$ on shell,

$$
\Sigma_A=\frac{\lvert U_{\mathbf g}\rvert^2}{(mc^2)^2}\,
\frac{\big\lvert\hat{\mathbf n}\times\big[(\hat{\mathbf n}-\boldsymbol\beta)\times\mathbf G\big]\big\rvert^2}
{\gamma^2\,\omega^2 D_0^4},
$$

which is the Liénard–Wiechert transverse structure for an acceleration along
$(\mathbb 1-\boldsymbol\beta\boldsymbol\beta)\cdot\mathbf F$. For
$g\gg q_{\min}$, $\mu\to0$ and $\langle\Sigma_A\rangle_\phi\propto g^2\lvert U_{\mathbf g}\rvert^2$.

**Recoil-exact case.** $E_{\rm res}(\mu,\phi)$ follows from the exact root and
the $\delta$ is integrated numerically over $(\mu,\phi)$, with weight
$1/(\kappa D)$ and the production $f_{\rm CBS}$. Energy conservation with an
on-shell final electron bounds the transfer at
$\lvert\mathbf g+\mathbf k\rvert\le(p+p_f)/\hbar$, which is close to $2p/\hbar$.

### 3. Lattice sum, structure factor, amorphous limit

Write the sample potential energy as a sum of atomic potentials $u_j$ at
$\mathbf R_a=\mathbf R^0_a+\mathbf u_a$. The radiated amplitude is linear in the
potential (first Born order), so the intensity at fixed
$(\omega,\hat{\mathbf n},\mathbf v)$ is
$\int\frac{d^3q}{(2\pi)^3}\lvert u(q)\rvert^2X(\mathbf q)\,\langle\lvert\rho(\mathbf q)\rvert^2\rangle/\mathcal V$,
with $\rho(\mathbf q)=\sum_a e^{-i\mathbf q\cdot\mathbf R_a}$. Here $X$ is the
per-unit-potential kernel of §§1–2, written below as $h$ after the orientation
average. Cross terms between different $\mathbf q$ at the same $\omega$ vanish
under the $\delta$ of §1. Independent Gaussian displacements give

$$
\frac{\langle\lvert\rho(\mathbf q)\rvert^2\rangle}{\mathcal V}
=\underbrace{n_{\rm at}\big(1-e^{-2W(q)}\big)}_{\text{thermal diffuse}}
+\underbrace{e^{-2W(q)}\,\frac{\lvert\rho^0(\mathbf q)\rvert^2}{\mathcal V}}_{\text{lattice-coherent}},
\qquad
\frac{\lvert\rho^0\rvert^2}{\mathcal V}\to\frac{(2\pi)^3}{V_c^2}\sum_{\mathbf g}\lvert S_{\mathbf g}\rvert^2\delta(\mathbf q-\mathbf g),
$$

while an amorphous (uncorrelated) solid has $\langle\lvert\rho\rvert^2\rangle/\mathcal V=n_{\rm at}$
exactly. With
$U_{\mathbf g}=V_c^{-1}\sum_j u_j(g)e^{i\mathbf g\cdot\mathbf r_j}e^{-W_j}$
(Mott–Bethe $u=4\pi e^2(Z-f_0)/g^2$, as in `cbs-amplitude`), the coherent term
integrates to exactly $\sum_{\mathbf g}\lvert U_{\mathbf g}\rvert^2h(\mathbf g)$. That is
the production line sum, with reflections added in intensity. The amorphous
term is $n_{\rm at}\int\frac{d^3q}{(2\pi)^3}\lvert u\rvert^2h$. In the
continuum limit, $\sum_{\mathbf g}\to V_c\int d^3g/(2\pi)^3$ and the shell average
$\langle\lvert\sum_j e^{i\mathbf g\cdot\mathbf r_j}\rvert^2\rangle=N_b$ hold. For an
orientation-averaged (isotropic) $h$ and $d^3g\to4\pi g^2dg$,

$$
\sum_{\mathbf g}\lvert U_{\mathbf g}\rvert^2h(g)\;\to\;
n_{\rm at}\int_0^\infty\frac{g^2\,dg}{2\pi^2}\,\lvert u(g)\rvert^2e^{-2W(g)}h(g),
\qquad 2W=\frac{B g^2}{8\pi^2}.
$$

Coherent (all $\mathbf g$) plus thermal diffuse equals the amorphous Born
bremsstrahlung. This is the sum rule behind "explicit lines double-count part of
an amorphous table". The basis average is needed only for this continuum
replacement. The ratio

$$
R(G)=\frac{\sum_{\lvert\mathbf g\rvert\le G}\lvert U_{\mathbf g}\rvert^2h(g)}
{n_{\rm at}\int\frac{g^2dg}{2\pi^2}\lvert u\rvert^2h}
$$

uses the exact discrete numerator and the exact amorphous denominator, with no
basis approximation.

### 4. Convergence of the denominator: recoil is load-bearing

In the classical limit $h(g)\propto g$ at large $g$ (§2), and with
$\lvert u\rvert^2\propto g^{-4}$ the integrand
$g^2\lvert u\rvert^2h\propto 1/g$. **The classical amorphous Born integral therefore
diverges logarithmically.** It is finite only with the recoil-exact
kinematics, which cut it at $q_{\max}\approx(p+p_f)/\hbar$. This is the usual
$\ln(q_{\max}/q_{\min})$ of bremsstrahlung. The DW-weighted numerator
converges without recoil, since $e^{-2W}$ cuts at
$g\sim\sqrt{8\pi^2/B}\approx13$ Å$^{-1}$ for $B=0.46$ Å$^2$. So $R(G)$ is
meaningful only with recoil-exact $h$, and the phrase "dropping $e^{-2W}$ and the
cutoff gives the amorphous Born brem" needs that qualifier.

### 5. Cheap filters

| Check | Result |
|---|---|
| Units of $dN_{\mathbf g}/d\Omega$ | $\alpha\,\omega t_L\,\Sigma_A/(\kappa D)$: Å$^{-1}\cdot$Å, dimensionless per sr. **pass** |
| Units of §2 boxed result | $\omega/(\hbar c\,g)$ is 1/(eV·Å): per eV per sr per Å of path. **pass** |
| $\kappa D=1-\mathbf v_f\cdot\hat{\mathbf n}$ | exact identity (§1). **pass** |
| $B\to\infty$ | $e^{-2W}\to0$ for every $\mathbf g\ne0$, so $R\to0$; everything becomes thermal diffuse. **pass** |
| $G\to0$ | empty sum, $R=0$; also $R=0$ for $G<q_{\min}(\omega)$. **pass** |
| $G\to\infty$, $B\to0$ | $R\to1$ up to the low-shell discreteness of $\sum$ vs $\int$, and only with recoil-exact $h$ (§4). **pass with qualifier** |
| $\pm\mathbf g$ | both enter $\sum_{\mathbf g}$ and the full-sphere $\int d^3q$; $\mathbf g=0$ never radiates, since $q\ge q_{\min}>0$. **pass** |
| DW exponent | $\lvert U_{\mathbf g}\rvert^2\propto e^{-2W}$, $W=Bs^2$, $s=g/4\pi$, so $2W=Bg^2/(8\pi^2)$. **pass** |

## Comparison with the author's draft

| Draft item | Independent result | Verdict |
|---|---|---|
| (1) $dN/d\Omega=\alpha\omega\lvert A\rvert^2t_L/(2\pi\kappa D)$ | §1 boxed | **matches** (factor, $\kappa$, $D$, root) |
| line weight, $P=\kappa D(\omega-\omega_{\rm res})/2$, $r_{\rm num}=\hbar c g^2/2E$, $r_{\rm den}=\hbar c\,\hat{\mathbf n}\cdot\mathbf g/E$ | §1 root and Jacobian | **matches**; $\kappa D$ shown to be $1-\mathbf v_f\cdot\hat{\mathbf n}$ |
| (2) $(1/\beta)\int\frac{d\Omega_g}{4\pi}\alpha\omega\lvert A\rvert^2/(2\pi\kappa D)\,\delta(E-E_{\rm res})$ | §1 per-path form, averaged | **matches** |
| (3) $\alpha\omega\langle\lvert A\rvert^2\rangle_\phi/(4\pi\beta^2\hbar c\,g)$, $g\ge\omega D/\beta$ | §2 boxed | **matches** |
| (4) lattice sum to $n_{\rm at}\int g^2dg/(2\pi^2)\lvert u\rvert^2e^{-2W}h$ | §3 | **matches** |
| (5) $R(G)$, limits | §3–§5 | **matches**; the $G\to\infty$ limit needs recoil-exact $h$ (§4), which the draft does not state |
| polarization basis $\hat{\mathbf e}_s\propto\hat{\mathbf n}\times\mathbf g$, $\hat{\mathbf e}_p=\hat{\mathbf e}_s\times\hat{\mathbf n}$ | orthonormal, so $\Sigma_A$ is basis-free | **matches** |
| script DW `exp(-B g**2/(8 pi**2))` | $e^{-2W}$ | **matches** |

No divergent factor, sign, exponent or unit was found in Eqs. (1)–(5).

## Numerical checks run for this verification

All runs were local and capped (`systemd-run --user --scope -p MemoryMax=8G`),
one process at a time, on Si with $B=0.46$ Å$^2$, $G=4$ Å$^{-1}$ (44 reflections),
and the draft's 0.5–10 keV, 500 eV band grid.

1. **Quadrature vs Monte Carlo over the production line weight.** Over 20 000
   random orientations, I summed `line_weights` from `scratch/shell1250.py`
   (production `_quantum_line_factors`, `_line_recoil_terms`,
   `_recoil_slope_factor` and `_recoil_line_allowed`) inside the band. I compared
   the result with the draft's $(\mu,\phi)$ quadrature (300 keV, 45°). The
   quadrature/MC ratio was $0.9981\pm0.0018$ with the script's QR sampler and
   $0.9983\pm0.0018$ with sign-corrected (Haar) QR. **The quadrature is right.**
2. **Sampler.** `np.linalg.qr` of a Gaussian matrix without the
   $\operatorname{sign}(\operatorname{diag}R)$ correction is not Haar: $q_{0,x}\le0$ in
   100 % of draws, and $\langle q_{1,y}\rangle=-0.48$. The cubic $m\bar3m$ point
   group plus the enumerated $\pm\mathbf g$ make it harmless here (item 1), but it
   would bias a lower-symmetry or non-centrosymmetric average.
3. **Per-line production vs closed form.** I compared `mc_cxr_spectrum`
   (`components=True`, `bin-mean`, a 5 eV grid wide enough to hold the whole
   line) with the §1 boxed yield for single reflections and orientations at
   $L=10^3,10^4,10^5$ Å. Lines above about 1.5 keV agree to $\le5\times10^{-4}$.
   Lines at 0.7–0.85 keV are low by 0.3–5 %, and their peaks are displaced
   by about 15 eV. (`node` quadrature is only valid when the line is resolved, at
   $L=10^3$ Å here.)
4. **Grazing-exit split** (same seed and $K=2000$ orientations as the author's
   `production()`, paired per orientation with the closed form):

   | Arm | all | $\lvert n_z\rvert<0.02$ | $[0.02,0.1)$ | $[0.1,0.3)$ | $\ge0.3$ |
   |---|---|---|---|---|---|
   | 30 keV, 45° prod/closed | 0.9700 | 0.060 | 0.873 | 1.0008 | 1.0008 |
   | 300 keV, 45° prod/closed | 0.9794 | 0.253 | 0.961 | 0.9976 | 0.9997 |
   | share of closed-form yield | — | 2.2 % | 7.6 % | 18.8 % | 71.4 % |

   $1/0.9700=1.031$ and $1/0.9794=1.021$ reproduce the logged 1.030 and 1.019.
5. **Denominator support.** In the author's Born $g$-profile (`share_*.json`),
   $g\times$integrand is flat, a $dg/g$ tail as in §4. It ends at
   627 Å$^{-1}$ (300 keV; $2p/\hbar=638$) and at 173 Å$^{-1}$ (30 keV; $2p/\hbar=180$).
   Below 100 Å$^{-1}$ lies 54 % of the 300 keV Born yield and 83 % of the
   30 keV yield.
6. **Electron–electron factor.** Every staged BremsLib table carries
   $1+\eta/Z$ (`bremslib-electron-electron`). For Si this is 1.042–1.047 across
   the band at 300 keV and 1.003–1.006 at 30 keV. Dividing it out of the
   author's BremsLib arm gives the nucleus-only ratio:

   | Arm | Born/BremsLib (draft) | Born/(BremsLib$/(1+\eta/Z)$), band | per bin, 0.5 → 9.5 keV |
   |---|---|---|---|
   | 300 keV, 45° | 0.967 | 1.010 | 1.03 → 1.009 (bins after the first 1.001–1.010) |
   | 300 keV, 90° | 0.991 | 1.036 | 1.06 → 1.018 |
   | 30 keV, 45° | 0.941 | 0.946 | 0.98 → 0.87 |
   | 30 keV, 90° | 0.946 | 0.951 | 0.99 → 0.86 |

7. **Share table.** Recomputing from `scratch/share_*.json` reproduces every
   draft-table entry. The 300 keV/90° JSON (written 23:56) postdates the
   broken-geometry log (23:51) and gives sane ratios, so the log row is stale,
   not the table.

## Diagnosis of the 2–3 % quadrature/production bias

**Cause: harness geometry, not the derivation and not the line weight.**
`shell1250.production()` places the segment in a 1 Å slab with a fixed lab
normal $\hat{\mathbf z}$ and randomizes $(\hat{\mathbf v},\hat{\mathbf n})$. Some
draws therefore have near-grazing exit, $\lvert n_z\rvert\lesssim0.1$. Production's
in-medium kinematics apply tangential continuity at the exit face,
$\mathbf k_{\rm eff}/\omega=\hat{\mathbf n}+\delta\,\nabla L$ with
$\nabla L=-\hat{\mathbf e}/(\hat{\mathbf e}\cdot\hat{\mathbf n})$
(`_setup.py`, `segment_escape_gradient`, `xray-in-medium-resonance`).
That correction diverges as $1/n_z$. It shifts the root (the ~15 eV peak
displacement at 0.7 keV), can fail the in-medium convergence guard, and comes
with a Beer–Lambert path of $0.5/\lvert n_z\rvert$ Å. All of it grows toward low
energy, since $\delta\propto E^{-2}$ and $\mu$ is large there. The yield-weighted
band sum is dominated by low-energy bins, so the deficit concentrates in the
2–10 % of orientations with grazing exit. The 30 keV arm, whose band ends at
about 3 keV, shows the largest bias. Excluding $\lvert n_z\rvert<0.1$ in both arms
leaves agreement at $1.0008$ (30 keV) and $0.9997$ (300 keV). The vacuum, surface-free
quadrature models the bulk, as the claim intends, so the bias is real slab-surface
physics that the comparison should exclude.

Ruled out:

- sinc$^2$ integration vs binning: `bin-mean` matches the closed form to $<5\times10^{-4}$ on resolved lines;
- $\kappa$: production `_recoil_slope_factor` is the same Jacobian, and §1 proves it is $1-\mathbf v_f\cdot\hat{\mathbf n}$;
- polarization basis: the sum is basis-free;
- recoil root: the shared helper matches the §1 root;
- bulk refractive index in $D$: $\delta\,\beta\cos\theta/D\sim10^{-3}$ at 0.7 keV, too small and not $n_z$-dependent;
- the sampler: item 1.

**Suggested harness fix (not applied).** Pair production with the closed form per
orientation and cut $\lvert n_z\rvert\ge0.3$ in both arms. Alternatively, keep
$\hat{\mathbf n}$ along the slab normal and randomize the lattice instead. Use a
sign-corrected QR or `scipy.spatial.transform.Rotation.random`.

## Is Born-without-DW "the same physics as BremsLib"?

**Yes at 300 keV, approximately at 30 keV, once electron–electron
bremsstrahlung is removed.** The draft's 0.94–0.99 compares nucleus-only
(screened-nucleus) Born with BremsLib's electron–atom table, which includes
$1+\eta/Z$. Electron–electron bremsstrahlung comes from inelastic scattering on
the target electrons. It has no lattice-coherent part and is not double-counted
by CBS lines. Against BremsLib's electron–nucleus part:

- **300 keV:** Born/e–n is 1.00–1.01 at 45° and 1.02–1.04 at 90°. The residual is
  of the expected size for the spinless convective current
  (`cbs-amplitude`: Dirac magnetic terms at $O(\eta^2)$) and for Born vs partial
  waves, with $\alpha Z/\beta\approx0.13$. The same Mott–Bethe amplitude in its
  amorphous, recoil-exact limit reproduces the tabulated electron–nucleus
  DDCS toward $\hat{\mathbf n}$ to 1–4 %.
- **30 keV:** Born/e–n falls from 0.98 to 0.87 toward the tip ($k/T\to0.33$).
  This is the known Born underestimate at low $T$ ($\alpha Z/\beta\approx0.31$), the
  Coulomb/Elwert correction that the partial-wave table carries. The physics is
  the same, but Born is accurate there only to about 13 %.

The draft line "reproduces the BremsLib DDCS toward $\hat{\mathbf n}$ to 2 %" is
true only after dividing out $1+\eta/Z$. The raw 300 keV/45° per-bin ratio is
0.957–0.985.

## Denominator recommendation

The double-counted quantity is, by §3, exactly the orientation-averaged
Born-DW CBS yield that production lines emit. Recommendation:

1. **Primary: absolute subtraction** of the Born-DW coherent share
   $S(G)=\sum_{\lvert\mathbf g\rvert\le G}\lvert U_{\mathbf g}\rvert^2h$, computed with
   the production amplitude and recoil-exact $h$. Then the orientation-averaged
   model total (lines plus residual continuum) equals BremsLib by construction.
   This needs no assumption that non-Born (Coulomb, spin) corrections factorize,
   and it never subtracts electron–electron bremsstrahlung. $S(G)$ costs the
   same to tabulate as $R(G)$, since $R=S/{\rm Born}$.
2. **If a fraction is preferred,** apply $R(G)$ to the electron–nucleus part
   $\text{BremsLib}/(1+\eta/Z)$ only. Applied to the full table it over-subtracts by
   the factor $1+\eta/Z$, about 4.5 % of $R$ at 300 keV.
3. With that correction the two choices agree at 300 keV/45°:
   $R\cdot\text{BremsLib}_{\rm en}=0.1067/1.045=0.102$ vs $S/\text{BremsLib}=0.103$ of
   brem. At 30 keV they differ by about 0.008 of brem (0.141 vs 0.133), because
   Born is about 6 % below the partial-wave value there. Carry that difference as the
   systematic.
4. Subtract only the $\lvert A_{\rm CBS}\rvert^2$ component. PXR and the
   PXR×CBS interference term have no counterpart in an amorphous bremsstrahlung
   table, and `components=True` is not an additive split
   (`coherent-line-spectrum`).

## Discrepancies and flags (none in Eqs. (1)–(5))

1. **Ledger Checks, "unexplained 2–3 % bias":** explained (above). It is a
   slab-exit (grazing $\hat{\mathbf n}$) artifact of the comparison harness. Paired
   agreement for $\lvert n_z\rvert\ge0.1$ is $1.0008$ / $0.9997$, and the quadrature vs
   MC of the production line weight is $0.998\pm0.002$.
2. **Ledger Claim, amorphous limit:** "dropping $e^{-2W}$ and the cutoff gives
   the amorphous Born brem" holds only with recoil-exact kinematics. The
   classical integral diverges as $\ln q_{\max}$ (§4). Half the 300 keV Born yield
   lies at $q>100$ Å$^{-1}$.
3. **Ledger Checks, "no-DW Born vs BremsLib 0.94–0.99":** BremsLib includes
   $1+\eta/Z$. Nucleus-only ratios are 1.00–1.04 at 300 keV and 0.98–0.87 at
   30 keV (above).
4. **Draft, BremsLib agreement "to 2 %":** holds only after the $1+\eta/Z$ correction.
5. **Harness sampler:** non-Haar QR. Harmless for Si; fix it before any
   lower-symmetry crystal.
6. **Minor:** `share1250.py` sets $f_0=0$ for $s>6$ Å$^{-1}$, a step in $Z-f_0$.
   At $g>75$ Å$^{-1}$ that is a sub-percent change in $\lvert u\rvert^2$ in a region
   with no DW weight; the denominator effect is negligible. The `share-arms.log`
   300/90 row is stale relative to `share_300_90.json`.

## Verdict

**Re-derivation matches.** Draft Eqs. (1)–(5), the line weight, the exact root,
the $\kappa D$ Jacobian, the orientation average, the structure-factor continuum
limit and the DW exponent all agree term for term. The quadrature reproduces a
Monte Carlo average of the production line weight to $0.2\%$. The claim
needs the qualifiers above: recoil-exact $h$, the electron–electron exclusion,
and the CBS-only component.

## Suggested ledger change

The task owner applies this, never as `signed-off`. Status `rederived`. Replace
the Checks bias clause with: "quadrature vs MC of the production line weight
0.998 ± 0.002 (20k orientations); the earlier 2–3 % gap is a 1 Å-slab
grazing-exit artifact ($\lvert n_z\rvert<0.1$), paired agreement 1.0008/0.9997
outside it". Replace "no-DW Born vs BremsLib 0.94–0.99" with "no-DW Born vs
BremsLib electron–nucleus part (÷(1+η/Z)): 1.00–1.04 at 300 keV, 0.98–0.87 at
30 keV (Born, low-T tip)". Add "recoil-exact kinematics required for a finite
denominator" to Assumptions. Resolve the denominator note in favour of
absolute Born-DW subtraction of the CBS-only component; $R(G)$, if used,
multiplies BremsLib/(1+η/Z). Link this write-up and add it to the
`radiation-physics/index.md` toctree.

## Production verification (2026-11-17)

Fresh context, branch `issue-1250-cbs-brem-double-count` at `9c72bd04c`
(`BREM_LATTICE_SHARE_MODEL = born-dw-cbs-share-v2/tables-brem-lattice-share-1/scored-set`).
The production estimator was derived below from the proposal above and from the
BremsLib table convention (`bremslib-angular-model`) before the bodies of
`montecarlo/spectrum/brem_lattice_share.py` and `montecarlo/runner/lattice_share.py`
were read.

### Expected production form

**Share density.** From §§1–3, the orientation-averaged CBS-only yield of the
scored set per unit path is

$$
S(T,\theta,k)=\sum_{\mathbf g\in\mathcal G}\lvert U_{\mathbf g}\rvert^2\,
h(\lvert\mathbf g\rvert;T,\theta,k)
\qquad[\text{photons}/(\text{eV sr Å})],
$$

with $h$ per unit $\lvert U_{\mathbf g}\rvert^2$ (eV$^{-2}$, $U_{\mathbf g}$ in eV).
Because $h$ depends on $\mathbf g$ only through $\lvert\mathbf g\rvert$, $S$ is an exact
shell sum for any scored set $\mathcal G$, complete star or not. In the
resonance variable

$$
x=\frac{k\,(1-\beta\cos\theta)}{\hbar c\,g\,\beta},
\qquad \frac{dx}{dk}=\frac{1-\beta\cos\theta}{\hbar c\,g\,\beta},
$$

which equals $\mu=\hat{\mathbf g}\cdot\hat{\mathbf v}$ on the classical root, a kernel
stored as a mass $M_i$ per $x$ bin of width $\Delta x_i$ gives
$h(k)=(M_i/\Delta x_i)\,dx/dk$. Interpolation must act on the density per unit
$x$, followed by the Jacobian at the actual $g$.

**Element split.** The continuum is scored as $\sum_j n_j\,d\sigma_j$, so any split
$S=\sum_j S_j$ removes exactly $S$ in total. The incoherent weights
$S_j\propto n_j\lvert u_j(g)\rvert^2/\sum_i n_i\lvert u_i(g)\rvert^2$ per shell are a
convention. They become load-bearing only where a node clamps.

**Table subtraction.** BremsLib stores
$\chi=(k/Z^2)\,d^2\sigma/dk\,d\Omega$ in mb/sr, and the scorer returns
$d^2\sigma/dk\,d\Omega=\chi Z^2/k$ with $k$ in eV. Removing $S_j$ from
$n_j\,d^2\sigma_j/dk\,d\Omega$ (with $n_j$ in atoms/Å$^3$) requires

$$
\Delta\!\left(\frac{d^2\sigma_j}{dk\,d\Omega}\right)=\frac{S_j}{n_j}
\;[\text{Å}^2/(\text{eV sr})],
\qquad
\Delta\chi_j=\frac{k\,S_j}{n_j Z_j^2}\times10^{11}\;\text{mb/sr},
$$

since 1 Å$^2$ = $10^{-16}$ cm$^2$ = $10^{11}$ mb. Here $k=\kappa T$ at each node,
including the node's actual top $\kappa$. The SDCS must follow from the same
solid-angle rule the scorer uses on the DDCS.

**Line-electron floor.** Let $\Lambda(T)$ be the expected track length per primary
at energy $T$ that the continuum scorer weights, and $\Lambda_\ell(T)$ the same
for the line scorer. The expected double count is
$\int dT\,\Lambda_\ell(T)\,\overline{S}(T)$, where $\overline S$ is the in-band,
angle-resolved share. The continuum removes
$\int dT\,\Lambda(T)\,\sum_j n_j\,\Delta\sigma_j(T)$. These are equal for every
scene only if

$$
\sum_j n_j\,\Delta\sigma_j(T)=\overline S(T)\,\frac{\Lambda_\ell(T)}{\Lambda(T)} .
$$

In production, both scorers share one transport. The line route keeps
segments with $E\ge E_{\rm cut,lines}$ and clips them at the cutoff
(`_clip_segments_to_cutoff`), and the continuum keeps segments down to
$E_{\rm cut,brem}\le E_{\rm cut,lines}$. Per primary,
$\Lambda_\ell/\Lambda=\Theta(T-E_{\rm cut,lines})$, so the exact condition is
$\Delta\sigma_j(T)=(S_j/n_j)\,\Theta(T-E_{\rm cut,lines})$. **The v2 rule is the
correct consistency condition.** The v1 rule ($\Theta\equiv1$ down to the
kernel floor) over-subtracted the share of every electron below the line
cutoff. The selector's `share_only_tables` must use the same floor, so its
$S_k$ is the yield the correction actually removes. If
$E_{\rm cut,brem}>E_{\rm cut,lines}$ (allowed uncoupled), the continuum has no
track below $E_{\rm cut,brem}$ and nothing is removed there, which is also
consistent.

### Implementation comparison

| Item | Code | Result |
|---|---|---|
| Line weight per unit path and $\lvert U\rvert^2$ | `brem_lattice_share.py:123` $\alpha\omega\lvert f\rvert^2/[(mc^2)^2\,2\pi\,\kappa D\,\beta]$ | **matches** §1 boxed (per path $L=\beta t_L$) |
| Orientation average | `:157–165`, $\mu\in(0,1]$ Gauss–Legendre, weight $\tfrac12 w\cdot\tfrac{2\pi}{n_\phi}/4\pi$ | **matches**; $\mu\le0$ cannot resonate ($\mathbf v\cdot\mathbf g>\hbar c g^2/2E>0$) |
| $x$ | `kernel_x`, `:172–182` | **matches** |
| Jacobian | `:312` density $=M/\Delta x$; `:315` scale $=\hbar c g\beta/(1-\beta\cos\theta)=dk/dx$; `:327` divides by it | **matches** |
| $\lvert g\rvert$, $\theta$, $T$ interpolation | density linear in $\ln g$ at fixed $x$ (`:307–326`); $\theta$ at kernel nodes then linear to table $\theta$ (`:375`); $T$: log total, linear shape (`:272–288`) | **matches** the stated scheme |
| Shell weights | `reflection_shell_weights`, `:247–269`: production `U_g` (static Mott–Bethe, DW, bonding charge; no $f'$, $f''$), split by $n_j\lvert u_j\rvert^2$ | **matches**; $\sum_j$ weights $=\lvert U_{\mathbf g}\rvert^2$ |
| Subtraction | `:373` `1e11 * k * S / (n * Z**2)`, $k=\kappa T\cdot10^3$ eV with top $\kappa$ (`:369–371`) | **matches** $\Delta\chi_j$ above |
| Clamp, SDCS | `:377–383`: clamp $\ge0$; SDCS × ratio of `solid_angle_integral` (linear in $\theta$, the scorer's rule) | **matches**; table stays self-consistent |
| Floor | `:365` `lo = max(kernel T0, T_floor_keV)`; nodes with $T<$ `lo` untouched | **matches** at nodes (see finding 1) |
| Emission only | `emission.py:110`, `:195`: corrected tables passed to the scorers only; coupled transport reads `runner._case_bremslib_tables` | **matches** |
| Scored set | `scored_reflections`, `runner/lattice_share.py:68–105`: explicit `hkl_list`, or rebuilt with `iter_reflection_shells` from the selection record's cutoff and shell width, with a count guard; without a record (streamed and directional routes, which never run the selector and score `hkl_list`) it subtracts `hkl_list`'s share. *(Owner note after verification: the earlier refusal of a selection case without its record was replaced by this rule once CI showed the directional route reaching it; `rebrem` now passes the record.)* | **matches** |
| Per layer | `_radiator`, `:58–65`; layer composition `abs_layers[L][2]`; amorphous layers skipped | **matches** |
| Exclusions | `_uncorrected_reason`, `:126–136`: disabled, EEDL (warned, `:170–190`), channeling, amorphous; elements without a table (`:209`) | **matches** the ledger |
| Floor value | `_line_cutoff_keV`, `:193–195`, `E_cut_lines_keV` default 5 keV, the same default as `case_driver.py:303` | **matches** |

### Numerical checks run for this verification

All checks were run locally under `systemd-run --user --scope -p MemoryMax=8G`,
one at a time, against the released kernel (`require_verified("brem-lattice-share")`).

1. **Absolute units and Jacobian, independent classical limit.** I computed a
   direct Liénard–Wiechert line yield with no recoil. For one reflection with
   $\lvert U\rvert^2=1$ eV$^2$,
   $dN/d\Omega\,dL=\alpha\lvert\hat{\mathbf n}\times[(\hat{\mathbf n}-\boldsymbol\beta)\times\mathbf G]\rvert^2/[2\pi\,\omega\,(\gamma mc^2)^2D_0^5\,\beta]$,
   with $\mathbf G=(\mathbb 1-\boldsymbol\beta\boldsymbol\beta)\cdot\mathbf g$. I averaged it over
   $4\times10^6$ uniform $\hat{\mathbf g}$ and histogrammed it in $E=\hbar c\,\mathbf g\cdot\mathbf v/D_0$
   ($\theta=45°$, 0.5–10 keV, 500 eV bins). The ratio of `lattice_share_density`
   to this yield was:

   | $T$, $\lvert g\rvert$ | band total | per bin |
   |---|---|---|
   | 300 keV, 3.0 Å$^{-1}$ (off-node) | 0.9988 | 0.987–1.009 |
   | 300 keV, 8.0 Å$^{-1}$ | 1.0018 | 0.987–1.013 |
   | 30 keV, 3.0 Å$^{-1}$ | 0.9871 | 1.003–1.005, then 0.945 and 0 in the last two bins, at the classical endpoint $x\to1$ where recoil lowers the line |

   The absolute normalization, the $dk/dx$ Jacobian and the off-node $\lvert g\rvert$
   interpolation are confirmed independently of the author's quadrature. The
   30 keV deficit is the expected recoil shift.
2. **Cited evidence** (`pyrite-output/validation/issue-1250/2026-11-17-p5a/units-released-kernel.log`):
   released-kernel $S$/P3 1.0030 (per bin 0.979–1.030), and the production
   brem drop with the corrected table $/S$ is 0.990 (per keV 0.969–1.02).
   These reproduce the ledger figures. Every cited artifact exists.
3. **Cutoff-bracket size.** For Si, $G=8$ Å$^{-1}$ (238 reflections), $\theta=45°$,
   I weighted the in-band share by a CSDA (Bethe) track length. The share
   removed by the ramp in finding 1 is $3.5\times10^{-3}$ of the share removed
   above 5 keV. The full 1–5 keV share that v1 removed is $7.2\times10^{-2}$ in
   the same model (measured: 2.8 % on the MoS₂/Si transport). v2 therefore
   removes about 95 % of the v1 inconsistency.

### Findings

1. **Floor edge is a ramp, not a step (minor; systematic, not a discrepancy).**
   `lattice_share_table` floors at nodes (`brem_lattice_share.py:365–367`), but
   the scorer interpolates the DDCS linearly in $\ln T$ between staged nodes
   (`brem_bremslib.py`, `_incident_bracket`). The staged 5 keV node is
   `5.000000000000002`, so it is corrected. Continuum segments with
   $T\in(4.729,5)$ keV, which score no lines, therefore lose a fraction
   $w=\ln(T/4.729)/\ln(5/4.729)$ of $S(5\,\text{keV})$. This over-subtraction is
   about $3\times10^{-3}$ of $S$ (check 3), or a few $10^{-4}$ of brem. For a
   cutoff off the staged grid, the ramp spans the bracketing interval instead
   and splits between under- and over-subtraction. Making it exact needs a
   per-segment indicator at evaluation, or a duplicated node at
   $E_{\rm cut,lines}$. Recommendation: record it as a systematic in the ledger;
   no code change is needed for `anchored`.
2. **Interpolation figures not reproduced by the cited artifact (provenance).**
   The ledger Checks and the release-builder docstring
   (`scripts/release_brem_lattice_share_table.py:17–19`) state "drop-one-node
   interpolation: $\lvert g\rvert\le1$ % per bin, $\theta\le0.7$ % (2.7 % at
   10°), $T\le0.4$ % total". The P5a logs on the released grid
   (`loo.log`, `loo-weighted.log`) report mass-weighted L1 0.049 ($\lvert g\rvert$),
   0.012 ($\theta$) and 0.046 ($T$), p95 per bin 5–7 %, and a $T$-total maximum
   of 0.99 at the 1.49 and 4.02 keV nodes. Leave-one-out doubles the spacing, so
   these overstate the error at the released spacing by about 4×, but they do
   not match the quoted figures either. The scene-level checks (item 2;
   check 1 per bin $\le1.3$ %) bound the actual error at a few percent per bin
   and 0.3 % in the band total. Recommendation: cite the logs' metric, or the
   probe that produced the quoted numbers, with its path.
3. **Silent $\lvert g\rvert$ range drop (minor scope).** `lattice_share_density`
   drops reflections with $\lvert g\rvert<0.25$ Å$^{-1}$ ($d>25$ Å) or above
   75.4 Å$^{-1}$ (`:303`) without recording it. The selector caps at 75.4, so
   only very large-cell crystals reach the lower edge. There, the continuum
   would keep that share, and the run record would not say so.
4. **Test gaps (minor).** `test_no_share_is_removed_below_the_line_electron_cutoff`
   checks the node rule with a 30 keV floor, which is correct as far as it
   goes. The runner test asserts only `record["line_cutoff_keV"] == 5.0`, not
   that `layer_emission_tables` leaves $T<E_{\rm cut,lines}$ nodes intact. No
   test pins the bracket ramp of finding 1.

### Is `anchored` justified?

The anchor pins the released dataset at one scene: $S$ to $10^{-6}$, within
0.5 % of the independent P3 quadrature (measured 0.30 %), and the production
drop/brem and drop/$S$. The $B\to\infty$ and $G\to0$ limits and the exact
$k S/(nZ^2)$ subtraction are pinned too. All 16 tests in the two modules pass
locally. The production equations match the re-derivation term for term, and
check 1 confirms the absolute scale independently. **`anchored` is justified
once the anchor tests are green in CI**, which the methodology requires and the
ledger records as pending.

### Production verdict

- **Claim**: `brem-lattice-coherent-fraction` —
  `montecarlo/spectrum/brem_lattice_share.py::{shell_kernel_x,kernel_x,reflection_shell_weights,lattice_share_density,lattice_share_table}`
  plus the runner wiring — production line weight (`coherent-line-spectrum`,
  `cbs-amplitude`) with the Ter-Mikaelian $e^{-2W}$ split.
- **Filters**: units pass (photons/(eV sr Å) → Å$^2$/(eV sr) → mb/sr with
  $10^{11}$); limits pass ($B\to\infty$, $G\to0$, floor); signs and conventions pass
  ($\chi=k\,d^2\sigma/Z^2$, no $\beta^2$).
- **Re-derivation**: `matches`. The v2 floor is the exact consistency
  condition at table nodes; the residual is the $\ln T$ ramp of finding 1.
- **Verdict**: `rederived` for the production form. The anchor supports
  `anchored` once CI is green.
- **Suggested ledger change**: keep `anchored` (CI green required). Add to
  Notes: "floor applied at staged `T` nodes; the `ln T` interpolation ramps it
  over one staged interval below `E_cut_lines_keV` (≈3e-3 of `S` at Si 30 keV,
  over-subtraction)". Replace the drop-one-node clause with the P5a
  `loo-weighted.log` figures, or cite the probe that produced the quoted ones.
