# `xray-in-medium-resonance`

## Independent Snell derivation, 2026-10-05

This section supersedes the historical bulk-direction claim below for an
externally observed direction. The historical calculation remains valid when
the supplied direction is the internal mode direction, and at normal exit.
The verifier did not implement this change. The old write-up was read to
preserve its history; the changed implementation bodies were not read before
the derivation in this section was recorded.

The source inputs are tangential wavevector continuity at a planar interface,
the homogeneous real-index dispersion relation, and the conservation equation
$\omega=\mathbf v\cdot(\mathbf k+\mathbf g)$ cited as Zhai SI Eq. (10) in
the new derivation docstring (Feranchuk–Spence Eq. (10)/(13) in the historical
ledger). Primary-source inspection subsequently found that citation number
incorrect: Zhai SI Eq. (7) defines the mismatch and Eq. (9) gives the vacuum
root; Eq. (10) specifies Gaussian beam coordinates. Here
$\omega=E/(\hbar c)$ is in inverse angstroms, $\mathbf v$ is
the dimensionless electron velocity, $\mathbf g$ is in inverse angstroms,
$\hat{\mathbf n}$ is the external vacuum unit direction,
$\delta(E)=1-\operatorname{Re}n(E)$, and $L(\mathbf r)$ is the vacuum-ray
escape distance in angstroms. The supplied signature is
`_in_medium_kinematics(v_dot_n, v_dot_g, n_re_tab, E_tab, v_dot_grad=None,
grad2=None) -> (denom, n_re)`. This verification concerns one affine escape
piece with fixed exit normal and straight electron velocity. It excludes
critical/grazing exit, Fresnel transmission, multiple escape faces within
one segment, and absorption already counted by Beer–Lambert attenuation.

### Interface closure and cheap filters

Let $\hat{\mathbf e}$ be the outward exit normal and
$\mu=\hat{\mathbf e}\cdot\hat{\mathbf n}>0$. On this planar piece,
$L=(d-\hat{\mathbf e}\cdot\mathbf r)/\mu$, hence
$\mathbf a=\nabla L=-\hat{\mathbf e}/\mu$. Tangential continuity gives
the internal wavevector divided by $\omega$ as

$$
\mathbf q
=\hat{\mathbf n}-\mu\hat{\mathbf e}
+\sqrt{(1-\delta)^2-1+\mu^2}\,\hat{\mathbf e}
=\hat{\mathbf n}+\delta\mathbf a
+O(\delta^2/\mu^3).
$$

The outgoing square-root branch fixes the sign. Expanding requires
$2|\delta|/\mu^2\ll1$, equivalently
$2|\delta|\,|\mathbf a|^2\ll1$. A guard at unity excludes the
critical-angle regime; it does not assert a uniform small truncation error
for all accepted points arbitrarily near that boundary.

Define $A=\mathbf v\cdot\hat{\mathbf n}$,
$B=\mathbf v\cdot\mathbf a$, and $G=\mathbf v\cdot\mathbf g$.
Substitution into the conservation equation yields

$$
\boxed{D(E)=1-A-\delta(E)B,\qquad E_{\rm res}D(E_{\rm res})=\hbar c G.}
$$

All of $A,B,D,\delta,\mathbf a,\mathbf q$ are dimensionless, and both
sides of the root equation are in eV. At zero index contrast the vacuum
root is recovered. At normal exit $\mathbf a=-\hat{\mathbf n}$, giving
$D=1-(1-\delta)A$, the historical bulk result. The sign of the shift
depends on $B$, not on $A$ alone: to first order,
$\Delta E/E_{\rm vac}=\delta(E_{\rm vac})B/(1-A)$. Reversing the exit
normal and its denominator together leaves $\mathbf a$ unchanged.
Units, limits, and signs therefore pass for this first-order closure.

The three-pass fixed-point solve is conditional, not a consequence of
$|\delta|\ll1$ alone. For $T(E)=\hbar cG/D(E)$, the derivative at a root is

$$
T'(E_{\rm res})
=\frac{E_{\rm res}\delta'(E_{\rm res})B}{D(E_{\rm res})}.
$$

Convergence requires its modulus to be below unity; a small $D$, a large
escape gradient, or sharp index dispersion can defeat contraction. The
existing convergence rejection remains necessary.

### Vector terms entering PXR and CBS

The same vector must be used throughout the first-order kinematics:

$$
\begin{aligned}
\mathbf k\cdot\mathbf v&=\omega(1-D),\\
\mathbf k\cdot\mathbf g
&=\omega\left(\hat{\mathbf n}\cdot\mathbf g
+\delta\mathbf a\cdot\mathbf g\right),\\
\mathbf k\cdot\mathbf e_s
&=\omega\delta\mathbf a\cdot\mathbf e_s,
\quad \hat{\mathbf n}\cdot\mathbf e_s=0,\\
k^2&=\omega^2|\hat{\mathbf n}+\delta\mathbf a|^2
=\omega^2(1-2\delta)+O(\delta^2|\mathbf a|^2),\\
|\mathbf k+\mathbf g|^2-k^2
&=g^2+2\mathbf k\cdot\mathbf g.
\end{aligned}
$$

Here $\hat{\mathbf n}\cdot\mathbf a=-1$ follows from the planar escape
geometry. Squaring the first-order vector retains an incomplete
second-order term; it does not make the approximation exact to second
order. PXR's polarization numerator includes
$[\mathbf v\cdot(\mathbf k+\mathbf g)]
[(\mathbf k+\mathbf g)\cdot\mathbf e_s]-k^2(\mathbf v\cdot\mathbf e_s)$.
External transverse polarization generally gives a nonzero
$\mathbf k\cdot\mathbf e_s$ at oblique exit. Dropping that term while
correcting only the resonance would be inconsistent at first order.

### Full dispersion and the single-segment intensity

For a straight segment with affine $L$ and constant coupling, the coherent
phase, apart from a constant, is

$$
\Phi(E,t)
=\omega[t-\hat{\mathbf n}\cdot\mathbf r(t)-\delta(E)L(\mathbf r(t))]
-\mathbf g\cdot\mathbf r(t).
$$

For duration $T$ in angstroms, its squared time integral is

$$
|I(E)|^2
=T^2\operatorname{sinc}^2\!\left(
\frac{T}{2\hbar c}
[E D(E)-\hbar cG]\right),
\qquad\operatorname{sinc}x=\frac{\sin x}{x}.
$$

The local energy derivative at the root is

$$
\boxed{J
=\frac{d[ED(E)]}{dE}\bigg|_{E_{\rm res}}
=D(E_{\rm res})-E_{\rm res}\delta'(E_{\rm res})B.}
$$

A frozen-index incoherent sinc uses width
$D(E_{\rm res})T/(2\hbar c)$; the local full-dispersion coherent width is
$|J|T/(2\hbar c)$. With slowly varying prefactors and a narrow isolated
line, energy integration gives $2\pi\hbar cT/|J|$ for the coherent
time-integral factor, versus $2\pi\hbar cT/|D(E_{\rm res})|$ for the
frozen-index expression. The ratio is $|D(E_{\rm res})/J|$.
For a nonlinear dispersion relation the exact variable substitution also
has an energy-dependent Jacobian, so even this local replacement is not
an exact finite-width identity. General exact equality is therefore
false. Constant index gives exact equality of the phase kernels;
negligible $E\delta'B/D$ and negligible curvature over the line gives
approximate equality. Energy-dependent coupling or attenuation further
restricts an equality claim about total intensity.

As an independent numerical check of the missing factor, take
$A=0.3$, $B=-0.8$, $E_{\rm res}=1000$ eV and
$\delta(E)=10^{-3}(1000\,\mathrm{eV}/E)^2$. Then
$D=0.7008$, $E\delta'=-0.002$, and $J=0.6992$.
The narrow-line coherent/frozen-index integrated ratio is
$0.7008/0.6992=1.00228833$. This difference persists on a single segment
without scattering or multiple-face geometry.

### Source check for the polarization extension

The [Zhai supplementary information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-025-66063-6/MediaObjects/41467_2025_66063_MOESM1_ESM.pdf),
page 3, Eq. (3), displays the eigenfield numerator with
$\mathbf g\cdot\mathbf e_s$, already specialized to a transverse incident
mode. Its Eq. (6)/(7) defines the CBS braces. The Feranchuk primary PDF
requires APS authentication, so its exact equation numbering was not
confirmed here. The general-vector extension below follows directly from
the Maxwell inverse, not from a claimed literal quotation of Eq. (13).

For $\mathbf p=\mathbf k+\mathbf g$ and real medium wavenumber squared
$K^2$, the Maxwell operator is
$M=(p^2-K^2)I-\mathbf p\mathbf p^{\mathsf T}$, and

$$
M^{-1}=\frac{I-\mathbf p\mathbf p^{\mathsf T}/K^2}{p^2-K^2}.
$$

Thus its scattering numerator contains
$\mathbf p(\mathbf p\cdot\mathbf e_s)-K^2\mathbf e_s$ before applying
$\mathbf k\cdot\mathbf e_s=0$. With fixed external polarization, the
first PXR term therefore needs $\mathbf g\cdot\mathbf e_s+
\mathbf k_{\rm eff}\cdot\mathbf e_s$. This algebra alone does not certify
the transmitted internal mode: a physical interface treatment must also
enforce its internal transversality and boundary normalization. Fresnel
amplitudes remain outside this claim.

For CBS, perturbing the current gives a direct velocity term
$\delta\mathbf v\cdot\mathbf e_s$ and a trajectory-phase term proportional
to $(\mathbf k\cdot\delta\mathbf r)(\mathbf v\cdot\mathbf e_s)$.
The force response is proportional to
$\mathbf b=\mathbf g-\mathbf v(\mathbf v\cdot\mathbf g)$.
Consequently the two terms are $\mathbf b\cdot\mathbf e_s$ and
$(\mathbf v\cdot\mathbf e_s)(\mathbf k\cdot\mathbf b)/G$, respectively:
the existing CBS braces remain valid with the corrected $\mathbf k$.
No additional $\mathbf k\cdot\mathbf e_s$ term enters that current expansion.

### Exterior synthetic zero-escape limit

The maintained single-segment anchor places its synthetic flight outside
the slab, ending on the entrance plane, with material escape distance
identically zero. On this affine piece $L\equiv0$ implies
$\mathbf a=\nabla L=0$. Therefore

$$
\mathbf k_{\rm eff}=\omega\hat{\mathbf n},\qquad
D=1-\mathbf v\cdot\hat{\mathbf n},\qquad
k^2=\omega^2,\qquad \mathbf k\cdot\mathbf e_s=0,\qquad J=D.
$$

The general norm identity is
$|\hat{\mathbf n}+\delta\mathbf a|^2
=1+2\delta\hat{\mathbf n}\cdot\mathbf a+\delta^2|\mathbf a|^2$.
Its specialized interior form $1-2\delta+\delta^2|\mathbf a|^2$
uses $\hat{\mathbf n}\cdot\mathbf a=-1$ and cannot be extended to zero
gradient. Exterior slab rows therefore use zero gradient and vacuum
wavevector magnitude; actual in-crystal rows retain the face-gradient
formula. This is a synthetic radiation-model limit, not a claim of
physical crystal emission from a charge outside the material. More
generally, a constant material-path distance contributes only constant
phase and attenuation, with no refractive slope along the flight.

### Implementation comparison after independent derivation

`_in_medium_kinematics` updates
`denom = 1 - v_dot_n - (1 - n_re) * v_dot_grad`, with an algebraically
equivalent bulk expression when `v_dot_grad == -v_dot_n`. Its final-pass
relative convergence rejection and the
`2 * abs(1 - n_re) * grad2 < 1` guard preserve the conditional domain.
`segment_escape_gradient` returns the derived planar gradient, selects the
finite-prism exit face on already split pieces, and uses the normal-exit
gradient for the working-facet groove geometry. All production CPU and
streaming callers pass this gradient explicitly; omitted geometry retains
the historical bulk convention for direct callers.

An independent constant-index numerical point with $A=0.3$, $B=-0.8$,
$\delta=0.001$, $|\nabla L|^2=4$, and $\hbar cG=1000(0.7008)$ eV
gave $D=0.7008$ and $E_{\rm res}=1000$ eV from the production helper,
matching the direct conservation calculation to the displayed precision.
No implementation helper was used to construct the expected denominator.

The batched, per-reflection, and streaming prologue paths sample all
couplings at this root. Their $\mathbf k\cdot\mathbf v$,
$\mathbf k\cdot\mathbf g$, detuning, and squared magnitude agree with
the same first-order vector. The squared magnitude keeps the explicit
second-order square of that vector, with the truncation qualification above.
The coherent formation argument retains the full tabulated
$\delta(E)\omega(E)$ phase; the incoherent width remains frozen at
$D(E_{\rm res})$. This is consistent with the requested kinematics scope
and intentionally leaves the integrated-yield Jacobian residual.

For incoherent splitting, the implementation retains parent duration
$T_p$ in every piece's prefactor and width and multiplies its mean
transmission by fraction $f_i$. The resulting model is

$$
S_p(E)=\sum_i f_i C_i T_p^2\bar{\mathcal T}_i
\operatorname{sinc}^2\!\left[
\frac{D_iT_p}{2\hbar c}(E-E_i)\right].
$$

Each $C_i$ includes the root-sampled coupling and spectral prefactor.
Identical roots and couplings reduce to the parent sinc times its
piece-averaged transmission, since $\sum_i f_i=1$. This is the intended
segment-level incoherent approximation; using each tiny piece's duration
instead would change the spectral shape. Different escape-face roots
give a weighted mixture of parent-width lines, not an exact piecewise
coherent integral. Coherent/grouped routes use actual piece duration and
sum fields, as required by their distinct model.

The follow-up implementation includes
`k_dot_e = omega_res * delta * (grad L).e` in both eager amplitude paths,
the fused real amplitude helper, and both streaming polarizations. Each
PXR numerator now uses `g_dot_e + k_dot_e`; CBS retains its original
brace expression with the corrected scalar wavevector terms. This matches
the independent general-vector algebra. The fused helper defaults this
new term to zero for historical direct callers.

**Scoped verdict: rederived.** The first-order root, scalar vector
invariants, general-vector PXR/CBS algebra, guard domain, and stated
segment-level weighting match.
Exact integrated-yield equality with the dispersive coherent route is
excluded; its first local divergent factor is
$D\mapsto J=D-E\delta'B$. The initially omitted PXR term
$[\mathbf v\cdot(\mathbf k+\mathbf g)]
\omega\delta\,\mathbf a\cdot\mathbf e_s$ was corrected and verified.
This verdict excludes physical interface mode normalization, Fresnel
transmission, and polarization matching across the boundary; it certifies
the stated fixed-external-polarization algebra. The ledger must replace its historical bulk/external
direction claim with this first-order interface kinematics, qualify
contraction, and correct the Zhai equation citation. No human sign-off is
implied.

## Claim and source

- Claim: CXR line kinematics closed on the in-medium photon dispersion `k = Re n(ω) ω n̂`: resonance `ω_res = v·g / (1 − Re n (v·n̂))`, `k·v = ω(1 − denom)`, `k·g = Re n ω (n̂·g)`, PXR detuning `|k+g|² − k² = g² + 2 k·g`, and PXR numerator `k² = (Re n ω)²`.
- Anchor: `montecarlo/spectrum/lines.py::_in_medium_kinematics`, `::mc_spectrum` (`xray_dispersion="refractive"`); CUDA port `montecarlo/spectrum/coherent_stream_jit_kernel.py::_coherent_prologue_kernel`.
- Source: energy–momentum conservation `ω = v·(k+g)` closed with the Maxwell dispersion relation in a homogeneous dielectric, `k² = (1+χ₀)ω²` (`xray-refractive-index`); Feranchuk–Spence 2000 Eq. (10)/(13) with `k² → εω²` rather than an ad hoc `n` inserted into the vacuum result.
- Intended quantity/signature (from the derivation docstring, read before the implementation body): `_in_medium_kinematics(v_dot_n, v_dot_g, n_re_tab, E_tab) -> (denom, n_re)`, with `denom` and `n_re` broadcast to the shape of `v_dot_g`; the docstring states the governing implicit equation `omega_res = v.g / (1 - Re n(omega_res)(v.n_hat))`, that only `Re n` enters (`Im n` is folded into the existing Beer–Lambert `mu(E)` escape factor, so applying it again here would double-count absorption), that the equation is solved by 3-pass fixed-point iteration from the vacuum root because the map's derivative is `~delta~1e-5`, and that the caller reconstructs `k·v = omega(1-denom)` exactly while `k·g` and `k²` pick up one and two powers of `n_re` respectively. The companion `mc_spectrum` docstring adds: `xray_dispersion="vacuum"` (default) keeps `k=omega` bit-for-bit; `"refractive"` uses `n = sqrt(1+chi_0)` (`materials.crystal.refractive_index`, `xray-refractive-index`), only the real part is applied, and the model is bulk-only (no interface/Fresnel refraction, so grazing observation geometry is out of scope).

## Independent derivation

### 0. Vacuum baseline (already-established result)

`line-energy-dispersion` / `coherent-emission` fix the repository's sign convention: for a charge on a straight trajectory `r(t) = r_c + v(t-t_c)` radiating into an outgoing vacuum mode `k = ω n̂` while coupling to a lattice harmonic `g` (susceptibility reconstructing as `χ_g exp(-i g·r)` in real space, per the resolved `coherent-emission` sign pivot), the phase is

```
Φ_g(t) = ω(t - n̂·r(t)) - g·r(t)
```

and stationarity (`dΦ/dt = 0`) gives exactly the energy–momentum matching condition cited by this claim,

```
ω = v·(k+g) = k·v + g·v.                                            (A)
```

Substituting the vacuum dispersion relation `k = ω n̂` into (A) gives the already-rederived vacuum resonance `ω_res = v·g/(1 - v·n̂)`. This claim's job is to redo the same closure with the medium's dispersion relation instead of the vacuum one, and to track how the extra factors of `n` propagate into the downstream algebraic pieces used by the PXR amplitude.

### 1. A vector identity that needs no physics

For *any* vector `k` and fixed `g`,

```
|k+g|² - k² = k² + 2k·g + g² - k² = g² + 2k·g.                      (B)
```

This is pure algebra — true whatever dispersion relation fixes `k`. It is recorded here because the claim calls it "PXR detuning" and because, below, substituting the *in-medium* `k` into it is the only step needed to carry the detuning from vacuum to medium; no new derivation is required at this step, only bookkeeping of which power of `n` enters.

### 2. Closing energy–momentum conservation with the medium dispersion relation

`xray-refractive-index` establishes the Maxwell dispersion relation for a homogeneous dielectric, `k² = (1+χ₀(ω))ω² ≡ ε(ω)ω²`, with `n(ω) = √(1+χ₀(ω))` so `|k| = n(ω) ω`. This claim's scope note says only the real part of `n` propagates into the *kinematics* (the imaginary part already enters the amplitude through the existing Beer–Lambert transmission factor, so re-applying it in `k` would double-count absorption), and that the bulk medium has no interface/Fresnel term — i.e. the direction of the emitted in-medium photon is taken to remain the vacuum observation direction `n̂` (no ray bending at the exit surface is modelled; consistent with `δ ~ 10⁻⁵–10⁻³` making any such bending negligible pointwise, the same scope caveat carried by `xray-refractive-index` itself). Under those two restrictions the in-medium wavevector entering the kinematics is

```
k = Re n(ω) · ω · n̂.                                                 (C)
```

Substituting (C) into the *same* energy–momentum matching condition (A) used in the vacuum case (Feranchuk–Spence Eq. (10)/(13), closed with `k → εω²` rather than adjusting the vacuum answer after the fact) gives

```
ω = Re n(ω) ω (n̂·v) + g·v
ω [1 - Re n(ω) (n̂·v)] = g·v.
```

Because `Re n` is itself evaluated at the resonance frequency, this is an implicit equation for the resonance, exactly the claim's

```
ω_res = v·g / (1 - Re n(ω_res)(v·n̂)).                                (D)
```

Defining `denom ≡ 1 - Re n(ω_res)(v·n̂)` (matching the vacuum-limit definition `denom = 1 - v·n̂` when `Re n → 1`) makes (D) `ω_res = v·g/denom`, formally identical in shape to the vacuum resonance with `denom` promoted from a `g`-independent constant to a segment-and-`g`-dependent root of an implicit equation.

**Existence/uniqueness of the root and the stated fixed-point rate.** Write `F(ω) = v·g/(1-Re n(ω)(v·n̂))`. Off-edge, `Re n(ω) = 1-δ(ω)` with `δ ~ 10⁻⁵–10⁻³` and `dδ/dω` smooth and small over the relevant line width, so

```
F'(ω) = v·g · (v·n̂) · dRe n/dω / (1-Re n(v·n̂))²,
```

which is `O(δ)` relative to unity (the same order as the fractional index contrast) whenever `|v·n̂|<1` keeps the denominator bounded away from zero — this reproduces the docstring's stated contraction rate `(v·n̂)(dn/dE)(dE/ddenom) ~ delta ~ 1e-5`. A contraction with ratio `~10⁻⁵` started at the vacuum root `ω_res^(0) = v·g/(1-v·n̂)` therefore gains `~5` decimal digits per iterate; two iterates already reach float64 rounding and the third is margin, matching the docstring's stated "three passes taken."

```{warning}
The premise of this subsection — `Re n(ω) = 1-δ` with `δ ~ 10⁻⁵–10⁻³`, and
`|v·n̂|<1` keeping the denominator bounded away from zero — was falsified on
2026-08-20. It holds off-edge in the X-ray regime, and the argument above is
correct there, but it is not unconditional: see **Addendum 2026-08-20: the
contraction is conditional**, at the end of this write-up. The code no longer
relies on it.
```

### 3. Downstream kinematic identities in terms of the same `denom`

From (C), `k·v = Re n(ω) ω (n̂·v)`. Using the definition of `denom`, `Re n(ω_res)(v·n̂) = 1-denom`, so, evaluated at `ω=ω_res`,

```
k·v = ω_res (1 - denom).                                             (E)
```

This is an identity that holds by construction of `denom` (it is exactly how `denom` was defined from (D)) — it is *not* an independent physical assumption, only bookkeeping that makes `k·v` cheap to recompute from the resonance solve without re-forming the vector `k`. It reduces to the vacuum identity `k·v = ω(1-denom) = ω(v·n̂)` when `Re n → 1`.

From (C) again, taking the dot product with `g` directly,

```
k·g = Re n(ω_res) ω_res (n̂·g).                                       (F)
```

`k·g` therefore carries exactly **one** power of `Re n` relative to the vacuum value `ω(n̂·g)` — the same one power that entered the resonance denominator, applied consistently to the same vector `k`.

Substituting the in-medium `k` into the vector identity (B) gives the "PXR detuning,"

```
|k+g|² - k² = g² + 2k·g = g² + 2 Re n(ω_res) ω_res (n̂·g),            (G)
```

using (F). No new physics enters here beyond (B) and (F): the detuning is just the same algebraic identity evaluated with the medium's `k`.

Finally, the squared magnitude of the in-medium wavevector, which enters the PXR amplitude as the coefficient of the `v·e` term (Feranchuk–Spence Eq. (13)'s `k²` — the term that is `ω²` in vacuum), is, from (C),

```
k² = (Re n(ω_res) ω_res)².                                           (H)
```

`k²` therefore carries **two** powers of `Re n`, i.e. `O(δ)` corrections entering `k·g` and `O(δ²)` (times a compensating `2` from squaring — concretely `(1-δ)²≈1-2δ`) entering `k²`. This asymmetry — one power of `n` in the linear-in-`k` term `k·g`, two powers in the quadratic-in-`k` term `k²` — is exactly what distinguishes *closing momentum conservation with the actual dispersion relation* from an *ad hoc* substitution: an ad hoc scheme that simply multiplied every occurrence of `ω` in the vacuum formula by a single common factor of `n` would not reproduce this graded power structure, since `k·g` and `k²` are respectively linear and quadratic in the same vector `k`. Equations (D), (E), (F), (G), (H) are the complete kinematic content that Feranchuk–Spence's Eq. (10) (resonance/energy conservation) and Eq. (13) (amplitude, with its `k²`-weighted term) require once `k` is defined through `k² = εω²` rather than left at its vacuum value.

## Cheap filters (before comparison)

- **Units.** `v, n̂` dimensionless; `g, k, ω` in Å⁻¹ (`c=1`); `Re n` dimensionless. `denom` is dimensionless, `k·v`/`ω_res` dimensionless, `k·g`/`g²`/`k²` all Å⁻², `E = ℏc ω` in eV. All consistent.
- **Limiting case `Re n → 1`.** (D)→ vacuum `ω_res = v·g/(1-v·n̂)`; (E)→ `k·v = ω(v·n̂)`; (F)→ `k·g = ω(n̂·g)`; (G)→ vacuum detuning `g²+2ω(n̂·g)`; (H)→ `k² = ω²`. Every expression collapses to the already-`rederived`/pre-existing vacuum form bit-for-bit, matching the ledger's stated `xray_dispersion="vacuum"` no-op limiting case.
- **Sign.** With the detector upstream of the beam so `v·n̂ < 0` (the hopg/100 keV/θ_obs=119° regression geometry), `Re n < 1` makes `Re n(v·n̂) > v·n̂` (both negative, `Re n(v·n̂)` less negative), so `denom = 1-Re n(v·n̂) < 1-v·n̂ = denom_vac`; since `ω_res = v·g/denom` with `v·g` fixed by the emission harmonic, a smaller `denom` gives a **larger** `ω_res`, i.e. the line moves up in energy — matching the ledger's recorded sign check.
- **Order of the shift.** `denom - denom_vac = (1-Re n)(v·n̂) = δ(v·n̂)`, so to first order `Δω_res/ω_res ≈ -Δdenom/denom_vac = -δ(v·n̂)/(1-v·n̂)`, matching the closed form the ledger cites for the fractional line shift.
- **Convention parity with the vacuum amplitude.** In vacuum, Feranchuk–Spence Eq. (13)'s amplitude numerator uses `ω²` multiplying the `v·e` term; this claim's role is only to identify which physical quantity `ω²` stands for once the photon is on the medium's mass shell rather than the vacuum one — namely `k²`, not `n²ω²` inserted as an independent multiplicative correction after the fact. (H) shows these are the same thing given (C), so there is no separate convention choice here.

## Implementation comparison

`_in_medium_kinematics(v_dot_n, v_dot_g, n_re_tab, E_tab)` solves

```python
denom = 1.0 - v_dot_n
for _ in range(3):
    E_res = HBARC_EV_ANG * (v_dot_g / denom)
    n_re = interp(n_re_tab, E_res)
    denom = 1.0 - n_re * v_dot_n
return denom, n_re
```

This is exactly the fixed-point iteration for (D): initialize at the vacuum root (`Re n = 1`), then repeatedly evaluate `Re n` at the current resonance estimate and refold it into `denom`. The CUDA port (`_coherent_prologue_kernel`, lines around the `use_medium` branch) performs the identical 3-pass loop in float32 with the same initialization and update, sharing the same docstring-cited contraction argument.

`_line_kin_core` (CPU batched path) and its equivalent inline block in `_accumulate_reflection` (CPU per-hkl path) and `_coherent_prologue_kernel` (CUDA) each then compute, given `denom` and (for the refractive branch) `n_re`:

```python
k_mag = om if n_re_seg is None else om * n_re_seg  # (H)'s |k|
k_dot_v = om * (1.0 - denom)  # (E)
k_dot_g = k_mag * n_dot_g  # (F), n_dot_g already n̂·g
v_dot_kg = v_dot_g + k_dot_v  # v·(k+g) = ω, trivially
detuning = g2 + 2.0 * k_dot_g  # (G)
```

and the PXR amplitude (`lines.py` line ~1070, Feranchuk–Spence Eq. (13)):

```python
A_PXR = chi / detuning * (v_dot_kg * g_dot_e - k_mag**2 * v_dot_e)
```

uses `k_mag**2`, i.e. `(H)`, in exactly the slot the vacuum amplitude fills with `ω²`. Every implementation line maps one-to-one onto (D)–(H):

| independent result | implementation |
|---|---|
| (D) `ω_res = v·g/(1-Re n(ω_res)(v·n̂))` | `_in_medium_kinematics` fixed point |
| (E) `k·v = ω(1-denom)` | `k_dot_v = om * (1.0 - denom)` |
| (F) `k·g = Re n ω (n̂·g)` | `k_dot_g = k_mag * n_dot_g` with `k_mag = om*n_re` |
| (G) `|k+g|²-k² = g²+2k·g` | `detuning = g2 + 2.0 * k_dot_g` |
| (H) `k² = (Re n ω)²` | `k_mag**2` in `A_PXR`'s numerator |

No divergent sign, factor, or power of `n` was found: `k·g` carries exactly one power of `n_re` (through `k_mag`), `k²` carries exactly two (through `k_mag**2`), and `k·v` is left in the `denom`-only form (E) that requires no explicit `n_re` at all — precisely the graded structure predicted by closing momentum conservation on `k = n(ω)ω n̂` rather than scaling the vacuum formula by a single global `n` factor. The `Re`-only restriction (`Im n` excluded to avoid double-counting the Beer–Lambert `mu(E)` factor) and the bulk/no-interface scope note are carried verbatim from the docstring into both the CPU and CUDA implementations and match the assumptions used in Section 2 above. The vacuum branch (`n_re_seg is None` / `use_medium=False`) leaves `k_mag = om`, `denom` unchanged, and every quantity above reduces identically to the vacuum forms, matching the `xray_dispersion="vacuum"` bit-for-bit no-op the ledger records.

## Adjudication

**rederived.**

The resonance condition, its implicit closure and fixed-point contraction rate, the `k·v`/`k·g`/detuning/`k²` identities, and their placement inside the Feranchuk–Spence Eq. (13) amplitude (`k_mag**2` standing in for the vacuum `ω²`) all follow directly from closing energy–momentum conservation `ω = v·(k+g)` with the Maxwell dispersion relation `k² = (1+χ₀)ω²` and `k = Re n(ω) ω n̂`, matching the CPU (batched and per-hkl) and CUDA implementations term for term with no divergent sign, factor, or power of `n`. The graded one-power-in-`k·g` / two-powers-in-`k²` structure is specifically the signature that distinguishes this from an ad hoc insertion of a single index factor, and it is present in the implementation exactly as derived.

Scope carried over unchanged (not re-litigated here, and not undermining the `rederived` verdict): real part only (imaginary part is the existing Beer–Lambert absorption, so it is deliberately not duplicated in `k`); bulk response only, no interface/Fresnel term, so grazing exit geometry is out of scope; this claim depends on `xray-refractive-index` (`filtered`, not yet independently `rederived`) for the dispersion relation `k²=(1+χ₀)ω²` and `n=√(1+χ₀)` itself — a discrepancy discovered in that upstream claim would propagate here.

Suggested ledger action: advance `xray-in-medium-resonance` from `filtered` to `rederived`. Human applies the ledger edit.

## Addendum 2026-08-19: the vacuum-dispersion switch was removed

`xray_dispersion` no longer exists; the in-medium resonance is unconditional. The derivation and its agreement with the implementation are unaffected, and the `rederived` determination stands. Two evidence changes:

- The limiting case `xray_dispersion="vacuum"` -> `k = omega`, bit-for-bit, is no longer a production code path. It survives at kernel level, where the CUDA prologue still accepts `Re n = 1` and collapses onto the vacuum kinematics exactly (`test_prologue_unit_index_reproduces_the_vacuum_kinematics`).
- The host anchor no longer differences a refractive run against a vacuum one. It solves the implicit in-medium root in closed form and asserts the spectral peak lands on it to inside one grid step, then checks the displacement from the vacuum root against `-delta (v.n_hat)/(1 - v.n_hat)` (`test_line_sits_on_the_in_medium_resonance_not_the_vacuum_one`). That pins the root itself, which the differential form did not.

## Addendum 2026-08-20: the contraction is conditional

Section 2's fixed-point rate is derived off-edge, from `Re n(ω) = 1-δ` with `δ ~ 10⁻⁵–10⁻³`, and from `|v·n̂|<1` keeping `denom = 1 - Re n (v·n̂)` bounded away from zero. Both premises fail together outside the X-ray regime, and the resulting defect was found downstream on `feature/relativistic-bethe-stopping` while measuring thick-target spectra: a small fraction of seeds returned a characteristic-line total ~10 orders of magnitude too large, finite rather than NaN, so nothing flagged it.

How it happens, on the traced sample (graphite, 100 μm, 300 keV):

1. A segment scattered nearly perpendicular to `g` gives `v·g = 1.727e-3` against a median `|v·g|` of 1.27, so the vacuum root `E_res = ħc (v·g)/(1 - v·n̂)` is only 6.57 eV.
2. At 6.57 eV carbon's tabulated `Re n` reads **2.07** — correct physics, not a data defect. The tabulation runs down to 1 eV and carries `Re n > 1` over 6.24–285 eV, peaking at 4.766 at 6.40 eV. This is the optical/UV regime, not the regime the solve was derived for.
3. With `v·n̂ = 0.4815`, `Re n (v·n̂) = 0.99931`, so `denom = 6.9e-4`: a spurious Cherenkov-like near-zero, reachable whenever `Re n > 1/(v·n̂)`.
4. The map is then not a contraction but an **expansive 2-cycle**, oscillating between `E_res ~ 6.57 eV` (`Re n ~ 2.07`, `denom ~ 1e-3`) and a keV-scale root (`Re n ~ 1`, `denom ~ 0.5186`). Three hard-coded passes return whichever half of the cycle pass 3 lands on — here `denom = 5.94e-4`, `E_res = 4942.7 eV`, matching the observed peak bin to the bin.
5. That root clears the `E_res > 10` eV keep window, so the cut `cbs-amplitude` names load-bearing for keeping `v·g` away from zero does not stop it: the cut bounds `v·g = ω denom`, so as `denom → 0` it stops bounding `v·g` at all.
6. `A_CBS ~ 1/(γ (v·g)²)` then gives `|A|² = 81.5` against a median `1e-10`.

The perturbative expansion has genuinely failed on such samples, by `cbs-amplitude`'s own stated validity condition `|U_g| g²/(γ mc² (v·g)²) ≪ 1`: with `U_g/mc² = 1.842e-5`, `g² = 3.506 Å⁻²`, `γ = 1.392` and `v·g = 1.727e-3`, the left-hand side is **15.6**. So the correct handling is to reject them, not to solve the root more carefully.

**What the implementation now does.** `_in_medium_kinematics` checks convergence instead of assuming it: the last fixed-point pass must move `denom` by less than `_RESONANCE_ROOT_RTOL = 1e-3`; pairs that fail carry NaN out of `denom` and drop on the caller's existing finite mask, the same route out-of-range tabulation energies already take. A genuine contraction moves `denom` by ~`δ³` ~ 1e-15 in float64 and is floored by float32 rounding (~1e-7) on the device twin, so the tolerance has five orders of margin either side. The CUDA prologue kernel in `coherent_stream_jit_kernel.py` carries the identical guard with the tolerance passed in, the way `hbarc` already is.

Measured: the traced seed falls from 7602 to 8.03e-07, inside the healthy population (6.6e-07–1.05e-06), and healthy seeds are bit-identical. At the catalog's 1000 Å production thickness the guard rejects **zero** pairs (hopg 30 keV, hopg 300 keV, silicon 100 keV); at 1e6 Å it rejects 0.233%. No golden moved.

New anchors, all in `tests/montecarlo/test_xray_dispersion.py`: `test_the_two_cycle_root_is_rejected_rather_than_returned`, `test_the_rejected_root_would_otherwise_have_passed_the_energy_window` (which pins that the 10 eV cut cannot serve as the guard), `test_a_converged_root_is_untouched_by_the_guard`, and `test_the_guard_is_inert_across_the_xray_regime`.

**Still owed on this row.** Section 2 above still presents the contraction rate as unconditional; it wants rewriting to state the domain in which it holds, and the `rederived` verdict wants re-confirming in fresh context at the same time. That belongs to whoever owns this validation, not to the branch that found the defect. `cbs-amplitude` carries the matching correction for its `v·g → 0` row.
