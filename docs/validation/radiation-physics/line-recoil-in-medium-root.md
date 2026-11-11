# Exact-recoil line root on a first-order Snell escape piece

Validation: `line-recoil-in-medium-root`.
Ledger row: [`line-recoil-in-medium-root`](../ledger-core-coherent-physics.md#line-recoil-in-medium-root).

Issue #1099 (option A of #527) moves every production PXR/CBS line onto the
exact recoil root. This page derives the root used by the production routes and
its lineshape slope. It also records which terms are dropped and how the
coherent phase carries them. The vacuum root and the exact mismatch slope are
the anchored claims [`line-recoil-resonance`](line-recoil-resonance.md) and
[`line-recoil-formation`](line-recoil-formation.md). The first-order Snell
geometry is [`xray-in-medium-resonance`](xray-in-medium-resonance.md). Whether
the vacuum root applies inside a crystal is
[`line-recoil-in-crystal`](line-recoil-in-crystal.md).

## Root

Notation follows the resonance derivation:

- $M=mc^2$, $E=\gamma M$, $\mathbf P=\beta E\hat{\mathbf u}$.
- $\mathbf Q=\hbar c\,\mathbf g$, where the lattice takes $+\hbar\mathbf g$
  (the reciprocal label is negated relative to Shi et al. 2024).
- $\epsilon$ is the photon energy and $\hat{\mathbf n}$ the detector direction.
- First-order Snell escape gives the in-crystal photon momentum
  $c\mathbf k=\epsilon(\hat{\mathbf n}+\delta\mathbf h)$, with
  $\mathbf h=\nabla L_{\rm esc}=-\hat{\mathbf e}/(\hat{\mathbf e}\cdot\hat{\mathbf n})$,
  $\hat{\mathbf n}\cdot\mathbf h=-1$ and $\delta=1-\operatorname{Re}n$.

Energy and momentum conservation, $E-\epsilon=\sqrt{M^2+\lVert\mathbf a-\epsilon(\hat{\mathbf n}+\delta\mathbf h)\rVert^2}$
with $\mathbf a=\mathbf P-\mathbf Q$, square once to

$$
P^2-\mathbf a^2=2\epsilon\bigl(E-\mathbf a\cdot\hat{\mathbf n}-\delta\,\mathbf a\cdot\mathbf h\bigr)
 -\epsilon^2\bigl(2\delta-\delta^2h^2\bigr).
$$

Divide by $2E$ and use $P^2-\mathbf a^2=2\mathbf P\cdot\mathbf Q-Q^2$:

$$
\beta\hat{\mathbf u}\cdot\mathbf Q-\frac{Q^2}{2E}
=\epsilon\Bigl[\underbrace{1-\beta\hat{\mathbf u}\cdot\hat{\mathbf n}
+\frac{\mathbf Q\cdot\hat{\mathbf n}}{E}
-\delta\beta\hat{\mathbf u}\cdot\mathbf h}_{D_{\rm ex}}
+\underbrace{\frac{\delta(\mathbf Q\cdot\mathbf h-\epsilon)}{E}
+\frac{\delta^2h^2\epsilon}{2E}}_{\text{dropped}}\Bigr].
$$

Production keeps $D_{\rm ex}$. In the kernels' inverse-angstrom units, with
$\mathbf v=\beta\hat{\mathbf u}$, $\omega=\epsilon/\hbar c$:

$$
\boxed{\omega=\frac{\mathbf v\cdot\mathbf g-r_{\rm num}}{D_{\rm vac}+r_{\rm den}-\delta(\omega)\,\mathbf v\cdot\mathbf h},\qquad
r_{\rm num}=\frac{\hbar c\,g^2}{2E},\quad r_{\rm den}=\frac{\hbar c\,\hat{\mathbf n}\cdot\mathbf g}{E}.}
$$

$D_{\rm vac}=1-\mathbf v\cdot\hat{\mathbf n}$ is formed cancellation-free
(`line-kinematics-precision`). The existing three-pass fixed point, its
convergence guard and the critical-angle refusal $2|\delta|h^2\ge1$ are
unchanged. They act on the shifted numerator and denominator
(`_in_medium_kinematics`).

**Limits.**

- $\delta=0$ gives exactly the vacuum root of `line-recoil-resonance`
  (Shi et al. 2024, Eq. (6)).
- $\hbar\mathbf g\to0$ ($r_{\rm num}=r_{\rm den}=0$) gives exactly the classical
  first-order Snell root of `xray-in-medium-resonance`.
- Normal exit ($\mathbf h=-\hat{\mathbf n}$) gives the bulk root
  $1-\operatorname{Re}n\,\mathbf v\cdot\hat{\mathbf n}+r_{\rm den}$.

**Dropped terms.** The dropped terms are mixed $O(\delta\,\hbar)$:
$\delta(\mathbf Q\cdot\mathbf h-\epsilon)/E$, plus $O(\delta^2)$ terms already
outside first-order Snell. Their relative effect on the root is bounded by
$[\,|\delta|(|\mathbf Q\cdot\mathbf h|+\epsilon)+\delta^2h^2\epsilon\,]/(E\,D_{\rm ex})$.

- The critical-angle refusal gives $|\delta||\mathbf h|<\sqrt{|\delta|/2}$, so
  $|\delta\,\mathbf Q\cdot\mathbf h|/E<\sqrt{|\delta|/2}\,Q/E$.
- For Huang's 10–15 keV graphite (002) lines ($\delta\sim10^{-5}$,
  $Q/E\approx7\times10^{-3}$, $D\sim1$) the dropped terms are below
  $10^{-7}$ relative, about 1 meV.
- With $\delta=10^{-5}$ and 30 keV–50 MeV oblique, backward and reverse
  geometries, the production root differs from a 65-digit in-medium mass shell
  by 0.37–1.0 of this bound. That is $\le5\times10^{-7}$ relative.

## Slope, width and Jacobian

The exact mismatch slope at the root is
$(1-\boldsymbol\beta_f\cdot\hat{\mathbf n})/\hbar c$ (`line-recoil-formation`).
In vacuum, $E_f=E-\epsilon$ and $\mathbf P_f\cdot\hat{\mathbf n}=\mathbf a\cdot\hat{\mathbf n}-\epsilon$
give

$$
1-\boldsymbol\beta_f\cdot\hat{\mathbf n}=\frac{E-\mathbf a\cdot\hat{\mathbf n}}{E-\epsilon}
=\kappa D_{\rm ex},\qquad \kappa=\frac{1}{1-x},\quad x=\frac{\epsilon}{E}=\omega\frac{\hbar c}{E}.
$$

The `sinc²` half-width $a=\kappa D_{\rm ex}t_L/(2\hbar c)$ and the energy
Jacobian of every route therefore carry $\kappa$ (`_recoil_slope_factor`).
`line-grid-resonance-bandwidth` uses the same first-zero width, so its tail
bound uses $\kappa D_{\rm ex}$. The dispersive $\delta'(E)$ Jacobian term of
`xray-in-medium-resonance` is unchanged.

A physical line needs a positive-mass final electron with $0<\epsilon\le T$:
$D_{\rm ex}>0$ and $x\le1-1/\gamma$ (`_recoil_line_allowed`). The guard rejects
a forbidden root as a non-radiating row. Coverage diagnostics count a forbidden
vacuum seed as non-positive.

## Coherent and formation phase

A phased piece of duration $t_L$ (`c dt` in Å) at electron age $t$ has
exact mismatch, linearized about the root, $\kappa(D_{\rm ex}\epsilon/\hbar c-N)$
with $N=\mathbf v\cdot\mathbf g-r_{\rm num}$. The classical phase is
$D_{\rm vac}\epsilon/\hbar c-\mathbf v\cdot\mathbf g$. The difference
accumulated over the age is carried by two cancellation-free terms:

- $d\to d+\kappa(xD_{\rm vac}+r_{\rm den})t$;
- $\mathbf g\cdot\mathbf r\to\mathbf g\cdot\mathbf r+\kappa(x\,\mathbf v\cdot\mathbf g-r_{\rm num})t$.

They use $\kappa-1=\kappa x$ (`_recoil_formation_line`; CUDA twin
`coherent_stream_jit_kernel.py::_coherent_prologue_kernel`). The age is the
electron's own segment-midpoint age. The bunch arrival time enters separately
through the inter-electron phase.

The photon escape phase $\delta\,\omega L_{\rm esc}$ stays geometric, with no
$\kappa$, both inside a piece and between pieces. Putting $\kappa$ on it would
break invariance under segment splitting, because $\kappa$ depends on the root
and $L_{\rm esc}$ is additive along the flight.

A coherent self-term therefore peaks where
$\kappa D(\epsilon-\epsilon_c)=\delta\,\epsilon\,\mathbf v\cdot\mathbf h$, with
$\epsilon_c$ the vacuum exact centre. That differs from the incoherent
in-medium root by $O(x\delta)$ relative ($\lesssim10^{-6}$), below the
first-order Snell order of the root itself.

## Scope

- Zero-phonon, kinematic, straight-piece line positions only.
- The catalogue's pinned families are expanded into complete symmetry orbits,
  so reflections sharing $g_z$ score as separate rows that the exact root
  splits (Shi et al. 2024 transverse recoil). Intensities, couplings and
  polarization factors are evaluated as before, at the shifted root.
- Not claimed:
  - transverse-recoil-resolved or shaped-electron spectra (#527 B/C);
  - many-beam or Bloch-branch shifts (`line-recoil-in-crystal` budget);
  - the high-energy optical-data deferral (#249).
- `materials.emission.photon_energies` uses the same vacuum root by default.
  `recoil=False` keeps the classical comparison.
- `CXR_ESCAPE_MODEL = "segment-mean-v6-exact-recoil"` forks identities.

## Evidence

| Check | Anchor |
|---|---|
| Production root and $\kappa D$ slope against a 65-digit unsquared mass shell at $\delta=0$, 30 keV–50 MeV, rtol $10^{-14}$ / $10^{-13}$ | `tests/montecarlo/spectrum/test_line_recoil_exact.py::test_kernel_root_and_slope_match_the_65_digit_mass_shell_at_zero_decrement` |
| Snell root against the 65-digit in-medium mass shell within the mixed-order bound | `::test_snell_root_differs_from_the_in_medium_mass_shell_only_at_mixed_order` |
| $\hbar\to0$ hook returns the classical Snell root bitwise | `::test_zero_hbar_hook_is_the_classical_snell_root`, `::test_classical_hook_recovers_the_classical_roots` |
| Huang et al. (2023) SI Table S4: 13 graphite/hBN rows through `mc_cxr_spectrum` on straight segments, Snell refraction included. Simulated $\overline{2E_2-E_4}=19.46$ eV against measured $18.19\pm0.66$ eV (stat ⊕ calibration), $-1.92\sigma$. Classical production lines give $0.41$ eV (refraction only), excluded at $27\sigma$ | `::test_huang_2023_harmonic_defect_through_production_lines` |
| Shi et al. (2024) Fig. 2: 30 nm Si [001], θ = 100°, 30/300/5000 keV. Each $g_z=2$ shell reflection peaks on its own exact root within one grid step. The shell split is >1 % at 30 keV and below 15 % of that at 5 MeV | `::test_shi_fig2_geometry_lines_sit_on_each_reflections_exact_root`, `::test_shi_fig2_splits_collapse_toward_mev` |
| Shi SI §S10 forward case, 100 keV, 100 nm Si [001], θ = 0.5–5° | `::test_shi_si_s10_forward_lines_sit_on_the_exact_root` |
| Coherent single segment on the incoherent root within $2\times10^{-6}$; split invariance | `tests/montecarlo/spectrum/test_xray_dispersion.py::test_single_segment_coherent_line_sits_on_the_escape_path_root`; `tests/montecarlo/coherent/test_coherent_formation_absorption.py::test_coherent_spectrum_is_invariant_under_segment_splitting` |
| CUDA prologue against the CPU formation line | `tests/montecarlo/coherent/test_coherent_formation_absorption_cuda.py::test_prologue_exact_recoil_matches_the_cpu_formation_line` (GPU-gated) |
| Complete symmetry orbits in pinned families | `tests/materials/test_crystallography.py::test_pinned_families_are_complete_symmetry_orbits` |
