# High-order reflection kinematic scope

Validation: `cxr-high-order-kinematic-scope`.

Independent fresh-context verification for issue #1173 (P1). The derivation
sections below were written from the ledger row, the cited sources and the
sibling claim `kinematic-validity-envelope` before the implementation bodies
were read. Code comparison and numerics follow in later sections.

## Claim, sources and conventions

The claim is a scope statement for the incoherent first-order PXR/CBS line
model over every non-forbidden reflection with $\lvert\mathbf g\rvert\le
30$ Å⁻¹ in the #642 scenes. It has three order-dependent parts: (i) a photon
Born screen, (ii) a harmonic isotropic Debye–Waller factor, and (iii) the
atomic-form-factor fit range. It adds a statement that electron recoil is exact
in every line route.

Conventions: $\mathbf g$ is a reciprocal-lattice vector with
$\lvert\mathbf g\rvert=2\pi/d$; $k=\omega/c=E/(\hbar c)$ is an inverse length;
$s=\sin\theta_B/\lambda=1/(2d)=\lvert\mathbf g\rvert/(4\pi)$. Sources:
Feranchuk et al., Phys. Rev. E 62, 4225 (2000), photon eigenmode as
reproduced in Zhai et al. (2025) SI Eq. (3); International Tables Vol. C
§6.1.1.3 for $W=B\sin^2\theta/\lambda^2$; Waasmaier & Kirfel, Acta Cryst. A51,
416 (1995), fit range $0\le s\le 6$ Å⁻¹.

## Independent derivation

### (i) Born parameter scaling at the line resonance

With the cell-averaged susceptibility

$$
\chi_{\mathbf g}=-\frac{r_e\lambda^2}{\pi V}F_{\mathbf g}e^{-W_{\mathbf g}}
=-\frac{4\pi r_e}{k^2V}F_{\mathbf g}e^{-W_{\mathbf g}},
$$

the coupling scale is

$$
k^2\lvert\chi_{\mathbf g}\rvert=\frac{4\pi r_e}{V}\lvert F_{\mathbf g}\rvert
e^{-W_{\mathbf g}},
$$

independent of $k$ except through the anomalous parts $f'(E)$, $f''(E)$ of
$F_{\mathbf g}$. For the Si lines here ($E\gg$ the 1.84 keV K edge) these are
small and slowly varying; the statement holds for the $f_0$ part exactly.

The vacuum detuning of the virtual wave is
$\Delta=\lvert\mathbf k+\mathbf g\rvert^2-k^2=g^2+2k\hat{\mathbf n}\cdot\mathbf g$.
The classical line resonance paired with this branch is
$\omega=(\mathbf k+\mathbf g)\cdot\mathbf v$, i.e.

$$
k=\frac{\beta\,\mathbf g\cdot\hat{\mathbf v}}{1-\beta\,\hat{\mathbf n}\cdot\hat{\mathbf v}},
\qquad \mathbf g\cdot\hat{\mathbf v}>0.
$$

Substituting, with $\hat{\mathbf g}=\mathbf g/g$,

$$
\frac{\Delta}{g^2}=\Gamma\equiv
1+\frac{2\beta(\hat{\mathbf g}\cdot\hat{\mathbf v})(\hat{\mathbf g}\cdot\hat{\mathbf n})}
{1-\beta\,\hat{\mathbf n}\cdot\hat{\mathbf v}}.
$$

$\Gamma$ depends only on directions and $\beta$, not on $\lvert\mathbf g\rvert$.
It is bounded. Multiplying by $1-\beta\hat{\mathbf n}\cdot\hat{\mathbf v}>0$,

$$
\Gamma\,(1-\beta\hat{\mathbf n}\cdot\hat{\mathbf v})
=1-\beta\,\hat{\mathbf n}\cdot\bigl[\hat{\mathbf v}-2(\hat{\mathbf g}\cdot\hat{\mathbf v})\hat{\mathbf g}\bigr]
=1-\beta\,\hat{\mathbf n}\cdot\hat{\mathbf v}_{\rm mirror},
$$

where $\hat{\mathbf v}_{\rm mirror}$ is the unit mirror image of
$\hat{\mathbf v}$ in the reflecting plane. So $\Gamma=(1-\beta c_m)/(1-\beta c_v)$ with
$c_m=\hat{\mathbf n}\cdot\hat{\mathbf v}_{\rm mirror}$ and
$c_v=\hat{\mathbf n}\cdot\hat{\mathbf v}$. Both factors lie in
$[1-\beta,1+\beta]$, so
$(1-\beta)/(1+\beta)\le\Gamma\le(1+\beta)/(1-\beta)$; the lower end is
attained by back-reflection ($\hat{\mathbf g}\parallel\hat{\mathbf v}$,
$\hat{\mathbf n}=-\hat{\mathbf v}$). The exact Bragg direction
$\hat{\mathbf n}=\hat{\mathbf v}_{\rm mirror}$ gives
$\Gamma=(1-\beta)/(1-\beta c_v)$, which is the smallest detuning for a given
emission angle. For $\beta<1$ the detuning never vanishes at the line resonance:
the "near-Bragg blow-up" of $g^2/\lvert\Delta\rvert$ is capped at

$$
\frac{g^2}{\lvert\Delta\rvert}\le\frac{1+\beta}{1-\beta}=\gamma^2(1+\beta)^2 .
$$

Hence

$$
D=\frac{k^2\lvert\chi_{\mathbf g}\rvert}{\lvert\Delta\rvert}
=\frac{4\pi r_e}{V}\,\frac{\lvert F_{\mathbf g}\rvert e^{-W_{\mathbf g}}}{g^2}\,
\frac{1}{\Gamma},
\qquad
D\le\frac{4\pi r_e}{V}\frac{\lvert F_{\mathbf g}\rvert e^{-W_{\mathbf g}}}{g^2}
\frac{1+\beta}{1-\beta}.
$$

The claim's factorization ($\lvert F\rvert e^{-W}/g^2$ times a
$\lvert\mathbf g\rvert$-independent geometric factor) is exact. The geometric
factor is $O(1)$ only for non-ultrarelativistic beams: at 300 keV
($\beta=0.7765$) the bound is $7.95$; at 30 keV ($\beta=0.3284$) it is $1.98$;
at 5 MeV it would be $\approx 4\gamma^2\approx 452$. Since $\lvert F\rvert\le
\sum Z$ and $e^{-W}\le1$, $D\to0$ as $g\to\infty$ for fixed $\beta<1$.
The ranking metric $\lvert F\rvert e^{-W}/g^2$ orders $D$ up to that factor,
not exactly.

### (ii) Debye–Waller sensitivity

Amplitude $\propto F_{\mathbf g}e^{-W}$ with $W=B s^2=B(g/4\pi)^2$. PXR and CBS
line intensities are quadratic in the reflection amplitude (PXR $\propto
\lvert\chi_{\mathbf g}\rvert^2$, CBS $\propto\lvert U_{\mathbf g}\rvert^2$,
interference $\propto$ their product), so $I_{\mathbf g}\propto e^{-2W}$ and

$$
\frac{\partial\ln I_{\mathbf g}}{\partial B}=-2\Bigl(\frac{g}{4\pi}\Bigr)^2
=-\frac{g^2}{8\pi^2},
\qquad g=30\ \text{Å}^{-1}:\ -\frac{900}{78.957}=-11.40\ \text{Å}^{-2}.
$$

For Si, $B=0.46$ Å² gives $W(30)=2.62$, $e^{-2W}=5.3\times10^{-3}$; a 10 %
error in $B$ moves that line's intensity by $\approx 41$ % (linearized
$-0.52$, exactly $e^{\mp0.524}$).

### (iii) Form-factor fit range

$s\le6$ Å⁻¹ maps to $g\le24\pi=75.40$ Å⁻¹. The g30 set reaches
$s=2.39$ Å⁻¹, inside the fit range.

## Implementation comparison

Read after the derivation above.

| Item | Independent | Code | Result |
|---|---|---|---|
| Resonance | $k=\beta\,\mathbf g\cdot\hat{\mathbf v}/(1-\beta\hat{\mathbf n}\cdot\hat{\mathbf v})$ | `omega = beta * (g @ beam) / (1.0 - beta * float(beam @ direction))` in `campaign/kinematic_validity.py::_radiator_optics` | match |
| Detuning | $g^2+2k\hat{\mathbf n}\cdot\mathbf g$ | `np.einsum("ij,ij->i", g, g) + 2.0 * omega * (g @ direction)` | match |
| $D$ | $k^2\lvert\chi_{\mathbf g}\rvert/\lvert\Delta\rvert$ | `materials/crystal.py::kinematic_validity` | match |
| Debye–Waller | amplitude $e^{-W}$, $W=B(g/4\pi)^2$ | `debye_waller` returns `exp(-B_ang2 * s**2)`, `s = g/(4 pi)`, applied once to each atom in `structure_factor` and `_reflection_families` | match |
| $f_0$ | Waasmaier–Kirfel at $s=g/4\pi$ | `materials/atomic.py::cromer_mann_f0` calls `xraydb.f0(element, g/(4 pi))` (WK despite the legacy name) | match |
| Ranking metric | $\lvert F(E_{\rm line})\rvert e^{-W}/g^2\times\Gamma^{-1}\propto D$ | `abs(S) / g_mag**2` with `S` evaluated at `E_ref_eV` (default 1000 eV), `exp(-W)` inside `S` | differs by $\Gamma^{-1}$ and by $\lvert F(E_{\rm line})\rvert/\lvert F(E_{\rm ref})\rvert$; see below |

Si is in `_EDGE_PRONE`, and the screen passes `use_henke=True`, so $F$
includes Chantler $f'(E)+if''(E)$ at the line energy.

## Numerical reproduction

Anchor `tests/campaign/test_kinematic_validity.py::test_high_order_silicon_set_stays_kinematic_and_ranks_with_born_parameter`:
1 passed (2.6 s). It covers Si at 300 keV for $\lvert\mathbf g\rvert\le22$ Å⁻¹
only (5408 reflections, max $D=6.54\times10^{-5}$); the g30 numbers in the
claim are not pinned by any test.

Full g30 screen, rebuilt from `checks/defaults_spectral_gate_642.py::scene_sweep`
with every layer's `hkl_list` replaced by
`dominant_reflections(..., n_families=10**9, g_max_invang=30)` and passed to
`case_kinematic_validity`:

| Scene / layer | Reflections | Evaluated | No positive resonance | `outside_optical_data` | Max $D$ | $R\le1$ | Mixing |
|---|---|---|---|---|---|---|---|
| Si, 300 keV, 10 µm | 13688 | 6839 | 6844 | 5 | $6.54\times10^{-5}$ (220) | 132 | 0 |
| MoS₂ film, 30 keV | 40198 | 20086 | 20099 | 13 | $2.81\times10^{-4}$ (002) | 47 | 0 |
| Si substrate, 30 keV | 13688 | 6839 | 6844 | 5 | $3.88\times10^{-5}$ (111) | 275 | 0 |

Evaluated total $6839+20086+6839=33764$, max $D=2.81\times10^{-4}$: both
match the claim, as do the extinction counts. Central-ray line energies stay
$\le45.5$ keV (Si 300 keV) and $\le19.4$ keV (30 keV scene).

**Mislabelled drop counts.** All 23 `outside_optical_data` records (5 Si,
13 MoS₂, 5 Si substrate) have central-ray energies between
$2.1\times10^{-13}$ and $3.3\times10^{-12}$ eV. They are reflections with
$\mathbf g\cdot\hat{\mathbf v}=0$ whose resonance is positive only by
floating-point roundoff, which fall *below* the 10 eV start of the optical
tables. They are not resonances past the optical-table ceiling and are
unrelated to `line-grid-emission-ceiling`. The claim's "5 Si and 13 MoS₂"
also omits the 5 substrate records.

### First-principles spot check

Own $\chi_{\mathbf g}$: diamond basis summed directly, $f_0$ from
`xraydb.f0`, $f'+if''$ from `xraydb.f1_chantler`/`f2_chantler` at the line
energy, $e^{-W}$ with $B=0.46$ Å², $V=a^3$ ($a=5.4309$ Å), CODATA $r_e$;
$D=4\pi r_e\lvert F\rvert/(V\lvert\Delta\rvert)$. Only the lab rotation and
the tilt geometry were taken from the library.

| hkl | $g$ (Å⁻¹) | $E$ (eV) | $\Gamma$ | $D$ (own) | $D$ (code) | rel. diff. |
|---|---|---|---|---|---|---|
| (2 2 0) | 3.272 | 3545.52 | 0.2235 | $6.5426\times10^{-5}$ | $6.5427\times10^{-5}$ | $-7.9\times10^{-6}$ |
| (1 1 −1) | 2.004 | 2659.14 | 0.6117 | $5.3759\times10^{-5}$ | $5.3759\times10^{-5}$ | $1\times10^{-8}$ |
| (9 −1 15) | 20.271 | 62.99 | 0.9986 | $9.1404\times10^{-8}$ | $9.1404\times10^{-8}$ | $1\times10^{-8}$ |
| (−1 17 3) | 20.005 | 241.16 | 0.9888 | $7.3697\times10^{-8}$ | $7.3697\times10^{-8}$ | $1\times10^{-8}$ |
| (20 4 −16) | 29.991 | 45483.48 | 1.0982 | $1.4361\times10^{-9}$ | $1.4361\times10^{-9}$ | $1\times10^{-8}$ |

(2 2 0) sits exactly on the central-ray Bragg direction: $\Gamma=1-\beta=0.2235$
(the $c_v=1$ case above), which is why it carries the largest Si $D$. The
$10^{-8}$ residual is CODATA rounding; (2 2 0)'s $8\times10^{-6}$ is the
Chantler interpolation at 3.5 keV.

### Orientation-independent bound

Using the bound $g^2/\lvert\Delta\rvert\le(1+\beta)/(1-\beta)$ and the
triangle inequality per element with $f_0(s)+\max_E\lvert f'+if''\rvert$
(maximum over 30 eV–300 keV), every g30 reflection satisfies, for any
emission direction at the nominal $\beta$ (lower $\beta$ after energy loss
only shrinks the bound):

| Scene | $(1+\beta)/(1-\beta)$ | $\max D$ bound | $\max D$ bound, $g\ge10$ Å⁻¹ |
|---|---|---|---|
| Si 300 keV | 7.95 | $1.6\times10^{-3}$ (111) | $5.8\times10^{-5}$ |
| Si 30 keV | 1.98 | $4.1\times10^{-4}$ (111) | $1.4\times10^{-5}$ |
| MoS₂ 30 keV | 1.98 | $1.1\times10^{-3}$ (002) | $1.3\times10^{-5}$ |

So $D<0.01$ holds over all directions, not just the central ray, in the
vacuum classical-resonance model. That is a stronger statement than the claim
makes and could replace its central-ray number.

### Ranking metric and the anomalous terms

$k^2\lvert\chi_{\mathbf g}\rvert$ is independent of $\omega$ for $f_0$ only.
At high order $f_0(s)$ is small, so the anomalous terms matter. Over the
evaluated Si reflections, $\lvert F(E_{\rm line})\rvert/\lvert
F(1\ \text{keV})\rvert$ (the ratio between $D\Gamma$ and the ranking metric):

| Scene | $g$ band (Å⁻¹) | n | min / median / max | fraction with $\lvert f'+if''\rvert>f_0$ |
|---|---|---|---|---|
| Si 300 keV | (0, 6] | 48 | 0.79 / 1.21 / 1.58 | 0.04 |
| Si 300 keV | (6, 18] | 1380 | 0.22 / 1.59 / 10.9 | 0.08 |
| Si 300 keV | (18, 30] | 5411 | 0.53 / 1.39 / 13.1 | 0.10 |
| Si substrate 30 keV | (18, 30] | 5411 | 0.49 / 1.57 / 19.6 | 0.24 |

The ranking therefore orders $D$ only up to $\Gamma^{-1}$ (factor $\le7.95$
at 300 keV) and this anomalous ratio (up to $\sim20$). Neither moves any
reflection near the threshold; the median-based anchor still passes.

### Debye–Waller and form-factor range

Finite difference of $\ln[\exp(-B s^2)]^2$ at $g=30$ Å⁻¹:
$-11.39863$ Å⁻², equal to $-g^2/(8\pi^2)=-11.39863$ Å⁻². $24\pi=75.398$ Å⁻¹.
Both match the claim.

## Electron recoil route check

`montecarlo/spectrum/cxr/_recoil.py::_line_recoil_terms`,
`_recoil_slope_factor` and `_recoil_line_allowed` are called by every CPU
incoherent line route (`cxr/_per_hkl.py` at the per-piece Snell resonance,
`cxr/_batched.py`, `cxr/_coverage.py`, `spectrum/reflection_rows.py`). The CuPy
float32 fused root (`cxr/_fused.py::in_medium_root`) receives the already
shifted `denom_vac + r_den` and `v.g - r_num` from `_in_medium_kinematics`. The
CUDA coherent prologue carries the device twin. The root is exact at
$\delta=0$ and drops only mixed $\delta\,\hbar\omega/E$ terms, per
`line-recoil-in-medium-root`. The kinematic statement holds.

The line amplitudes use `_quantum_line_factors` (`cbs-amplitude`): exact
second-order plane-wave elements with a **spinless** convective current; the
docstring states that Dirac magnetic-moment terms enter the intensity at
$O(h^2)$. The relevant small parameters grow with order:
$\hbar g/(mc)=0.116$ at $g=30$ Å⁻¹ and $\hbar\omega/(mc^2)=0.40$ for the
206 keV kinematic-bound line ($205.7$ keV $=\hbar c\beta g/(1-\beta)$). "No
small-recoil approximation" is true of the root, not of the amplitude.

## "Not claimed" audit

Present and correct: site-resolved/anisotropic ADPs (MoS₂ scalar $B$),
thermal-diffuse emission, anharmonicity, dynamical diffraction off the
central ray, central-ray and vacuum scope.

Missing, order-dependent at $d<0.4$ Å:

1. **Spinless amplitude.** Magnetic-moment (spin) terms omitted at $O(h^2)$
   in intensity (`cbs-amplitude`). They scale with $(\hbar g/mc)^2$ and
   $(\hbar\omega/mc^2)^2$, so they grow with order and line energy. This
   contradicts "no order-dependent approximation beyond three" as written.
2. **$q$-independent dispersion corrections.** $f'(E)+if''(E)$ is added to
   $f_0(s)$ without a core-shell form factor and with the same $e^{-W}$. At
   high $s$, where $f_0$ is about 1 electron, this term exceeds $f_0$ for 10 %
   (Si 300 keV) to 24 % (Si 30 keV) of high-order reflections. It is
   larger at soft line energies near the L edges.
3. **Ranking and frequency independence.** "$\omega^2\chi_g$ is independent
   of $\omega$" holds for $f_0$ only. The ranking matches $D$ only up to
   $\Gamma^{-1}\in[(1-\beta)/(1+\beta),(1+\beta)/(1-\beta)]$ and
   $\lvert F(E_{\rm line})\rvert/\lvert F(E_{\rm ref})\rvert$. The
   geometric factor is $O(1)$ only for $\gamma\sim1$. It reaches 7.95 at
   300 keV and grows as $\approx4\gamma^2$ at MeV energies.
4. **Static disorder and the independent-atom model** are not
   order-dependent model terms here and are adequately covered by
   "converged within the model".

## Verdict

The three scaling statements (i)–(iii), the arithmetic ($-11.4$ Å⁻²,
75.4 Å⁻¹), the 33,764-reflection screen, $D\le2.8\times10^{-4}$, the
extinction counts and the recoil route coverage all rederive. The
implementation $D$ matches an independent $\chi_{\mathbf g}$ to $10^{-8}$.

`discrepancy` (claim text only; no code or numeric fault). The first exact
divergence is the attribution of the "5 Si and 13 MoS₂" dropped reflections:
these are $\mathbf g\cdot\hat{\mathbf v}=0$ roundoff resonances at
$\le3.3\times10^{-12}$ eV, below the table start. They are not "resonances past
the optical tables" (`line-grid-emission-ceiling`). Second, "no
order-dependent approximation beyond three" omits the spinless amplitude and
the $q$-independent $f'+if''$. Correcting both wordings, and stating the
$g^2/\lvert\Delta\rvert\le(1+\beta)/(1-\beta)$ bound, would make the claim
`rederived` as it stands numerically.
