# Fiber-texture CXR intensity average

Validation: `cxr-fiber-intensity-average`.

Independent fresh-context verification for issue #1480 (opt-in HOPG
full-fiber CXR model). Sections 1–5 were written from the ledger row, the
[research note](../../research/physics/fiber-nonbasal-cxr-1480.md#model), the
[production note](../../research/physics/fiber-production-1480.md#model-and-limits),
the cited sibling claims (`cr-uniaxial-thermal-projection`,
`brem-lattice-coherent-fraction`, the final-state orthogonality section of
[cross-reflection-coherence](cross-reflection-coherence.md#final-state-orthogonality-1480))
and International Tables Vol. A (No. 194), before reading the implementation
bodies. The code comparison, numerics and test run follow in
[Source-to-code comparison](#source-to-code-comparison).

## Conventions

$\lvert\mathbf g\rvert=2\pi/d$. The crystal frame has $\hat{\mathbf c}$ along
reciprocal $(001)$ (hexagonal: $\mathbf c^*\parallel\mathbf c$). A crystallite
at roll $\phi$ presents the lab vector $R_\phi\mathbf g$, with $R_\phi$ a
rotation by $\phi$ about $\hat{\mathbf c}$, applied before any mosaic tilt or
miscut. $I_{\mathbf g}(\phi)$ is the per-reflection spectral intensity
(PXR, CBS and their interference) of one electron track crossing one
crystallite at roll $\phi$. $D$ is the finite scored reflection domain.

## 1. Intensity addition and the fiber average

For an unshaped plane-wave electron, reflection $\mathbf g$ leaves the
electron in $\mathbf p_f=\mathbf p_i-\hbar\mathbf g-\hbar\mathbf k$; distinct
$\mathbf g$ give orthogonal final states, so the trace over the electron
removes $A_{\mathbf g}A_{\mathbf g'}^*$ and

$$
I(\phi)=\sum_{\mathbf g\in D}\lvert A_{\mathbf g}(\phi)\rvert^2
=\sum_{\mathbf g\in D}I_{\mathbf g}(\phi).
$$

Interference between PXR and CBS of the *same* $\mathbf g$ is kept (same final
state). With one roll per coherent region and a uniform roll distribution,
the expected intensity is linear in the orientation distribution:

$$
I_{\rm fiber}=\frac1{2\pi}\int_0^{2\pi}I(\phi)\,d\phi .
$$

Basal $\mathbf g\parallel\hat{\mathbf c}$ is invariant under $R_\phi$, so
$I_{\mathbf g}(\phi)$ is constant and is scored once.

## 2. Sixty-degree closure and the sector quadrature

Let $C=R_{\pi/3}$. Suppose (a) $C D=D$ and (b) every coupling datum entering
$I_{\mathbf g}$ — $\chi_{\mathbf g}$, $U_{\mathbf g}$ moduli and their relative
phase, $B(\mathbf g)$ — is equal at $\mathbf g$ and $C\mathbf g$. Because
$I_{\mathbf g}(\phi)$ depends on $\phi$ only through the lab vector
$R_\phi\mathbf g$ and those data,
$I_{\mathbf g}(\phi+\pi/3)=I_{C\mathbf g}(\phi)$, and

$$
I(\phi+\pi/3)=\sum_{\mathbf g\in D}I_{C\mathbf g}(\phi)
=\sum_{\mathbf g'\in CD}I_{\mathbf g'}(\phi)=I(\phi).
$$

Hence $I$ has period $\pi/3$ and

$$
I_{\rm fiber}=\frac3\pi\int_{\phi_0}^{\phi_0+\pi/3}I(\phi)\,d\phi
\approx\sum_{\mathbf g\parallel c}I_{\mathbf g}
+\frac1N\sum_{i=0}^{N-1}\sum_{\mathbf g\not\parallel c}I_{\mathbf g}(\phi_i),
\qquad
\phi_i=\phi_0+\frac\pi3\,\frac{i+\tfrac12}N .
$$

This is the equally spaced (midpoint) rule on one full period of a periodic
function: it integrates every Fourier harmonic $e^{6ik\phi}$ exactly unless
$N\mid k$, so the error is the aliasing of harmonics $6kN$ and falls faster
than any power of $N$ for smooth $I(\phi)$. On a binned photon axis a moving
narrow line makes $I(\phi)$ only piecewise smooth per bin, so convergence is
algebraic in practice; the claim says "to quadrature error", which covers both.
Equally spaced nodes $2\pi i/N$ on the full circle are a different rule: mod
$\pi/3$ they occupy only $N/\gcd(N,6)$ distinct residues, so they waste nodes
(the ledger's aliasing remark is correct).

**Is (b) true for graphite and h-BN?** Both are P6₃/mmc. The $6_3$ screw
$\{C\,|\,0,0,\tfrac12\}$ is a space-group operation, so for every species $j$,
$S_j(C\mathbf g)=e^{-i\mathbf g\cdot\mathbf t}S_j(\mathbf g)$ with one phase
common to all species. Any species-weighted sum ($\chi_{\mathbf g}$ with $f_j$,
$U_{\mathbf g}$ with $Z_j-f_j$) acquires the same phase, so $\lvert\chi\rvert$,
$\lvert U\rvert$ and $\chi U^*$ are invariant. This includes h-BN's two-species
odd-$l$ terms ($\propto f_B-f_N$) and graphite's $\lvert S\rvert^2\in\{1,3\}$
classes. $B(\mathbf g)$ depends only on $\cos^2(\mathbf g,\hat{\mathbf c})$
and is roll-invariant (section 3). Condition (a) is automatic for a
$\lvert\mathbf g\rvert\le G$ ball of a primitive hexagonal lattice (its point
group is 6/mmm); it would fail for an R-centred (rhombohedral) set. Condition
(b) is the substantive one: a primitive hexagonal cell with a trigonal basis
(point group $\bar 3m$, $3m$, $\bar 6m2$ without the screw, ...) passes (a) but
not (b), so a closure test on $\mathbf g$ vectors alone is insufficient in
general. Expected implementation: refuse non-closed domains *and* verify (or
restrict to structures guaranteeing) equal coupling data across $C$-orbits.

## 3. Projected uniaxial Debye–Waller factor

For a site covariance $U=u_b I+(u_c-u_b)\hat{\mathbf c}\hat{\mathbf c}^\mathsf T$
(Å², $u_b,u_c$ the per-axis mean-square displacements), the amplitude factor
is $e^{-W}$ with

$$
W=\tfrac12\mathbf g^\mathsf TU\mathbf g
=\tfrac12\lvert\mathbf g\rvert^2\bigl[u_b+(u_c-u_b)\cos^2(\mathbf g,\hat{\mathbf c})\bigr].
$$

In the crystallographic form $W=B s^2$ with $s=\sin\theta/\lambda=1/(2d)=\lvert\mathbf g\rvert/4\pi$,

$$
B(\mathbf g)=\frac{16\pi^2W}{\lvert\mathbf g\rvert^2}
=8\pi^2\bigl[u_b+(u_c-u_b)\cos^2(\mathbf g,\hat{\mathbf c})\bigr]\ \text{Å}^2,
\qquad
e^{-W}=\exp\!\Bigl(-\frac{B(\mathbf g)\lvert\mathbf g\rvert^2}{16\pi^2}\Bigr).
$$

So "$8\pi^2$" pairs with $s=\sin\theta/\lambda$ (equivalently $\lvert g\rvert^2/16\pi^2$
with $\lvert g\rvert=2\pi/d$); the "$16\pi^2$" convention would be the
intensity exponent $2W$ written per $s$, or $B$ referred to $1/d$ without the
$1/4$. Limits: $u_c=u_b=u\Rightarrow B=8\pi^2u$ (isotropic); basal
$\cos^2=1\Rightarrow B=8\pi^2u_c=B_{33}$; in-plane $\cos^2=0\Rightarrow
8\pi^2u_b$. Graphite (#535): $u_b=0.0031$, $u_c=0.0165$ Å², so $8\pi^2u_c=1.303$
Å² versus the catalog $B_{33}=1.30$ Å² (0.2 % difference in $B$; at $(004)$,
$2W$ changes by $\approx 5\times10^{-4}$). Because $\chi_{\mathbf g}$,
$U_{\mathbf g}$, the CBS amplitude ($\propto U_{\mathbf g}$) and the PXR amplitude
($\propto\chi_{\mathbf g}$) are all linear in the thermally averaged structure
factor, the *same* $e^{-W(\mathbf g)}$ must multiply both PXR and CBS
amplitudes and appear squared in the shell weight $\lvert U_{\mathbf g}\rvert^2$.
For a single species with one shared tensor, per-site and per-reflection
projection are identical; graphite's two sites differ in $u_c$ (0.016/0.017),
so a per-reflection scalar is exact only with the site mean.

## 4. Continuum subtraction

By `brem-lattice-coherent-fraction`, the amorphous (BremsLib) continuum
contains, per reciprocal vector, the full-sphere orientation-averaged CBS share
$\lvert U_{\mathbf g}\rvert^2h(\lvert\mathbf g\rvert)$. Scoring the oriented CBS
of a set $D$ explicitly therefore requires removing that share for exactly the
same set $D$ — every Friedel mate and every $C$-orbit member, each with its own
$B(\mathbf g)$ (the 4π average rotates the crystal, not $U$ relative to
$\mathbf g$, so the crystal-frame projection is correct inside the isotropic
average). The subtraction must be linear in $D$, so per-batch shares are added
first and any nonlinear safety clamp (share $\le1$, continuum $\ge0$) is applied
once to the total; clamping per batch would under-subtract. Removing the 4π
share while adding the fiber-oriented CBS balances only in the orientation mean
(a fiber is not isotropic), which the ledger states as a limitation. Non-basal
PXR has no counterpart in BremsLib and is not subtracted.

## 5. Limits expected

- $N\to\infty$: sector midpoint sum $\to$ continuous average.
- Domain $D=\{00l\}$ only: the non-basal sum is empty, $I=\sum_lI_{00l}$, and
  with $B_{\rm basal}=8\pi^2u_c$ the prior basal model is recovered up to the
  $B_{33}$ vs $8\pi^2u_c$ difference above.
- $u_c=u_b$: $B(\mathbf g)$ is constant; tensor rows equal scalar rows.
- $CD\ne D$: periodicity fails; must be refused.
- $N$ distinct nodes: the sector rule must not duplicate orientations.

## Source-to-code comparison

Implementation read after sections 1–5, at `05e82aaed`.

| Derived term | Code | Agreement |
|---|---|---|
| $I=\sum_{\parallel}I_{\mathbf g}+\frac1N\sum_i\sum_{\not\parallel}I_{\mathbf g}(\phi_i)$ | `FiberReflectionSet._average` (`fiber.py:219-248`): basal rows with `rolls=1` at `azimuth_rad`; other rows at `azimuth_rad + (pi/3)(i+0.5)/rolls`, each added as `value / rolls` | exact; basal test `h[:2] == (0, 0)` is $\mathbf g\parallel\hat{\mathbf c}$ for hexagonal indices |
| $\phi_0$ | `azimuth_rad` (caller) or the case's `azimuth_rad` (`fiber_case.py:82`) | exact |
| $R_\phi$ about $\hat{\mathbf c}$, before mosaic/miscut | `_orientation_R`: cut maps reciprocal `(0,0,1)` to $+\hat z$, then `Rz(azimuth)`, then miscut; mosaic tilts act on the oriented $\mathbf g$; `surface_hkl` defaults to `(0,0,1)` and any other axis is refused (`_validate_geometry`) | exact |
| $C:(h,k,l)\mapsto(-k,h+k,l)$ | closure test `fiber.py:155-158` | correct 60° rotation of $h\mathbf a^*+k\mathbf b^*$ ($\mathbf a^*\to\mathbf b^*$, $\mathbf b^*\to\mathbf b^*-\mathbf a^*$); checks membership and equal $B$, refuses otherwise |
| condition (b) of section 2 | not tested in code; guaranteed by the HOPG-only gate (`fiber.py:82`) | see finding 1 |
| $B(\mathbf g)=8\pi^2[u_b+(u_c-u_b)\cos^2]$ | `prepare` (`fiber.py:120-129`) → `UniaxialThermalSite.b_factor_along` | exact; same form in `common_uniaxial_b` (used by the check's independent arm, not by `prepare`) |
| $e^{-W}=\exp(-B\lvert g\rvert^2/16\pi^2)$ on $\chi_{\mathbf g}$ and $U_{\mathbf g}$ | `reflection_coupling_tables` via `per_reflection_b` (`crystal.py:681-684`): one `dwf[i]` multiplies both the $\chi$ table and `u_coef`; `emission_coupling_tables` looks up $B(-\mathbf h)$, so Friedel evenness is required and enforced (`fiber.py:153`) | exact; PXR ($\propto\chi$) and CBS ($\propto U$) share the factor |
| shell $\lvert U_{\mathbf g}\rvert^2$ with the same $B(\mathbf g)$, exact set $D$ | `shell_weights` → `_reflection_shell_weights(crystal, self.hkls, lookup)` | exact; same `hkls` tuple as the emission average, every Friedel mate and orbit member |
| sum all shells, then clamp once | `corrected_bremslib_tables` passes the whole $(g, w)$ list in one `_lattice_share_tables` call; `_finish_table` applies `np.maximum(ddcs, 0)` after the full subtraction | exact |
| raw continuum, no second subtraction | `_case_bremslib_tables` loads uncorrected tables; `_layer_brem_from_segments(tables=...)` applies no further share | exact |
| intensity addition across $\mathbf g$ | batches of `hkls` scored by the incoherent `mc_cxr_spectrum` (per-reflection squaring); coherent emission refused (`_validate_route`, `_single_radiator`) | exact |

## Numerical checks

Probe (FP64 CPU, not committed): 24 fixed straight segments, 25–30 keV,
velocity spread about $\hat{\mathbf c}$, observation 60°, bin-mean lines on
1–12 keV, $\lvert g\rvert\le4$ Å⁻¹ (34 rows).

- **Condition (b), independent structure factors.** Hand-entered ITA No. 194
  coordinates (graphite 2b + 2c; h-BN B at 2c, N at 2d) with Cromer–Mann $f_0$:
  the largest change of $\lvert\chi\rvert$, $\lvert U\rvert$ or $\chi U^*$
  under $C$ over $\lvert h\rvert,\lvert k\rvert\le4$, $\lvert l\rvert\le6$ is
  $2.1\times10^{-13}$ (graphite) and $4.6\times10^{-13}$ (h-BN, including the
  $f_B-f_N$ odd-$l$ terms). A trigonal counterexample (primitive hexagonal,
  atoms at $(0,0,0)$ and $(\tfrac13,\tfrac23,0.3)$) changes
  $\lvert\chi_{101}\rvert$ by 3.6 under $C$: a $\mathbf g$-set closure test
  alone would accept it.
- **Projected $B$.** $W$ from the code's $B$ against independent
  $\tfrac12(u_bg_\perp^2+u_cg_z^2)$ with
  $g_\perp^2=(2\pi/a)^2\tfrac43(h^2+hk+k^2)$: max relative difference
  $1.3\times10^{-14}$ over all 34 rows. $B_{(00l)}=1.3028$ Å².
- **Real-kernel periodicity.** Non-basal sum at $\phi=0.37$ vs
  $\phi+\pi/3$ and $\phi+2\pi/3$: L1 difference $3.5\times10^{-15}$ and
  $3.6\times10^{-15}$ relative; at $\phi+\pi/6$ it is $0.31$, so the roll
  dependence is not trivial.
- **Quadrature convergence** against an independent 360-node full-circle mean
  (no sector assumption) plus basal:

  | $N$ | integrated rel. error | L1 rel. error |
  |---:|---:|---:|
  | 2 | $6.1\times10^{-4}$ | $1.8\times10^{-2}$ |
  | 4 | $7.6\times10^{-4}$ | $8.9\times10^{-3}$ |
  | 8 | $5.7\times10^{-5}$ | $3.6\times10^{-3}$ |
  | 16 | $3.2\times10^{-5}$ | $5.6\times10^{-4}$ |
  | 32 | $4.2\times10^{-7}$ | $4.5\times10^{-6}$ |

- **Basal limit.** Basal rows with the tensor against the same rows with the
  scalar $B_{33}=1.30$ Å²: integrated difference $1.7\times10^{-4}$, the
  expected $8\pi^2u_c=1.3028$ vs $1.30$ offset.
- **Isotropic limit.** A constant callable $B$ equals the scalar-$B$ route
  bit for bit (max relative difference 0).
- **Focused tests.** `pixi run pyrite-dev test tests/montecarlo/spectrum/test_fiber_model_1480.py tests/materials/test_reflection_thermal.py`:
  30 passed.

## Findings

1. **Closure test is geometric plus $B$ only (concern, not a defect at
   present).** `FiberReflectionSet.__post_init__` (`fiber.py:155-158`) checks
   that $C\mathbf h$ is in the set with equal $B$, but not that
   $\lvert\chi\rvert$, $\lvert U\rvert$ and $\chi U^*$ agree across the
   orbit. Section 2 needs both. For HOPG (P6₃/mmc) the coupling condition
   holds by the $6_3$ screw (verified to $2\times10^{-13}$), and the model is
   gated to HOPG (`fiber.py:82`), so the scored result is correct. Lifting
   the gate to a crystal with a trigonal basis would pass the closure test
   and give a wrong sector average. When the gate is relaxed, compare the
   coupling rows across each $C$-orbit as well. h-BN satisfies the
   condition (verified), so its refusal rests only on the missing in-plane
   ADP, as the ledger states.
2. **Test coverage.** The unit test of the average
   (`test_intensity_average_batching_and_components`) uses a mocked scorer
   with an imposed $\cos6\phi$. No test checks real-kernel 60° periodicity or
   compares against an independent full-circle average. The probe above
   shows that a test of this kind takes seconds. It would pin item 2 against
   an independent reference.
3. **Ledger `Code` field.** It lists `materials/_reflection_thermal.py::common_uniaxial_b`,
   but `FiberCXRModel.prepare` calls `UniaxialThermalSite.b_factor_along`
   directly. `common_uniaxial_b` feeds only the check's independent arm in
   `checks/fiber_nonbasal_cxr_1480.py`. The two agree in form. Listing it is
   harmless, but the row could say which one production uses.
4. **Stated limitation, confirmed.** The continuum subtraction is the 4π
   orientation average of $D$, not the fiber-conditioned share. Lines and
   continuum therefore balance only in the orientation mean. The ledger
   `Notes` already say this.

No factor, sign, exponent, unit or convention differs between the derivation
and the implementation.

## Verdict

`rederived`. The suggested ledger edit (applied by the task owner, never as
`signed-off`) is Status `rederived`. Add to Notes: "2026-11-26 fresh-context
verification (write-up: `radiation-physics/cxr-fiber-intensity-average.md`):
matches term for term; real-kernel 60° periodicity to $4\times10^{-15}$;
$N=32$ sector rule against a 360-node full-circle mean, L1 $4.5\times10^{-6}$.
The closure test covers the $\mathbf g$ set and $B$ only, which is sufficient
under the HOPG gate." `anchored` is appropriate once a real-kernel
periodicity or full-circle test (finding 2) is in the `Anchor` field and green.
