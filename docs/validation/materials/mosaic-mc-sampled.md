# mosaic-mc-sampled

## Claim

- **ID:** `mosaic-mc-sampled`
- **Symbols:** `montecarlo/spectrum/cxr/_mosaic_sampled.py` (`history_tilts`, `lattice_points`, `tilt_rotate`, `polarization_pairs`, `accumulate_sampled_mosaic`, `kept_orientation`, `dot_g`, `mosaic_orientations`); `montecarlo/spectrum/cxr/_per_hkl.py::_accumulate_reflection` (per-row $\mathbf g$); `montecarlo/spectrum/cxr/_grouped.py::_referenced_phases`; `montecarlo/spectrum/cxr/_setup.py::_refuse_mosaic_sampling`; `montecarlo/spectrum/reflection_rows.py::_seed_orientations`/`_orientation_terms`.
- **Source:** `docs/physics/materials/crystal-mosaicity.md`, section "Sampled crystallites"; the reference average is the `mosaic-mc` Gauss–Hermite rule ([write-up](mosaic-mc.md)).
- **Intended quantity:** an unbiased Monte-Carlo estimator of the incoherent mosaic average
  $\bar S(E)=\int S(E;\mathbf t)\,p_\eta(\mathbf t)\,d^2\mathbf t$, with $p_\eta$ the isotropic 2-D Gaussian tilt density of per-axis standard deviation $\sigma=\eta_{\rm FWHM}/(2\sqrt{2\ln 2})$.
- **Estimator as claimed:** history $e$ carries $K$ tilts $t_{e,k,a}=\sigma\,\Phi^{-1}\big((p_{k,a}+s_{e,a})\bmod 1\big)$, $a\in\{x,y\}$, where $p_k$ is a $K$-point rank-1 lattice and $s_e$ a Cranley–Patterson shift from counter uniforms of the `SeedSequence(seed).spawn(8)[7]` child keyed by `elec_id`. Every row and every reflection of history $e$ is scored at each $\mathbf t_{e,k}$ with weight $1/K$.

## Independent derivation

The derivation below was formulated from the physics page and the ledger row before the implementation bodies were compared.

### 1. Each draw is an exact tilt sample

Let $U$ be uniform on $[0,1)$ and $p\in[0,1)$ fixed. The map $u\mapsto(p+u)\bmod 1$ is a measure-preserving rotation of the circle $\mathbb R/\mathbb Z$, so $(p+U)\bmod 1$ is again uniform. If the two shift components $s_{e,x}$, $s_{e,y}$ are independent uniforms, the pair $\big((p_{k,x}+s_{e,x})\bmod 1,\,(p_{k,y}+s_{e,y})\bmod 1\big)$ is uniform on the unit square for *every* lattice point $p_k$, independent of the generator. The probability-integral transform then gives

$$
t_{e,k,a}=\sigma\,\Phi^{-1}(u_{e,k,a})\sim\mathcal N(0,\sigma^2),\qquad t_{e,k,x}\perp t_{e,k,y},
$$

so each $\mathbf t_{e,k}$ has density $p_\eta(\mathbf t)=\exp[-\lvert\mathbf t\rvert^2/(2\sigma^2)]/(2\pi\sigma^2)$, the `mosaic-mc` weight. Draws within one history are correlated (they share $s_e$); this does not enter the mean.

**Discretization and clipping.** If the uniforms are an exact midpoint grid $u=(m+\tfrac12)2^{-52}$, the shifted value can reach $0$ only through rounding of $p_k+s_e$ near $1$; an open-interval clip at $2^{-53}$ maps it to $\Phi^{-1}(2^{-53})\approx-8.2$. The tail mass beyond $\lvert z\rvert=8.2$ is $\approx 2\times10^{-16}$ and its contribution to $\langle z^2\rangle$ is $\approx 2\cdot 8.2\,\varphi(8.2)\approx 4\times10^{-14}$, so the clipping and the $2^{-52}$ discreteness bias any bounded integrand by $\lesssim10^{-13}$ relative — far below any statistical error. The lattice only stratifies the $K$ draws against one another; it cannot bias the mean.

### 2. Weighting and normalization

Let $S_e(E;\mathbf t)$ be the (already per-electron normalized) contribution of all rows of history $e$ when the crystallite is at tilt $\mathbf t$, for all reflections. Conditional on the trajectories $\mathcal T$, and with the tilts drawn from a stream independent of transport,

$$
\mathbb E\Big[\sum_e \tfrac1K\sum_{k=1}^K S_e(E;\mathbf t_{e,k})\,\Big|\,\mathcal T\Big]
=\sum_e \tfrac1K\sum_{k=1}^K \int S_e(E;\mathbf t)\,p_\eta(\mathbf t)\,d^2\mathbf t
=\sum_e \bar S_e(E).
$$

The GH route computes $\sum_e\sum_{ij}W_{ij}S_e(E;\mathbf t_{ij})$ with $\sum_{ij}W_{ij}=1$, which tends to the same $\sum_e\bar S_e(E)$ as the node count grows. The two estimators therefore share their mean provided (i) the per-electron normalization outside the orientation loop is identical, (ii) the sampled weights $1/K$ sum to one per history, and (iii) $S_e$ is linear in the orientation-resolved intensity — true for any incoherent tally (intensities, not amplitudes, are summed across crystallites).

### 3. Granularity

Assigning one crystallite to a whole history (primary, its substeps and any secondaries carrying the same `elec_id`) only changes which draws are shared. Linearity of expectation gives the same mean for per-row, per-flight or per-history assignment, because each row's tilt is marginally $p_\eta$ and independent of that row's trajectory. Only the covariance between rows changes, hence the error bars and seed-to-seed correlations. The requirement is that transport never consults the tilt — true here, because the tilt is generated after transport from a disjoint seed-sequence child.

### 4. What must be recomputed per row

For a crystallite rotation $R(\mathbf t)$, $\mathbf g\to R\mathbf g$. Every quantity built from the *direction* of $\mathbf g$ changes: $\mathbf v\cdot\mathbf g$ and $\hat{\mathbf n}\cdot\mathbf g$ (hence $E_{\rm res}=\hbar c\,(\mathbf v\cdot\mathbf g-r_{\rm num})/D$ and the recoil terms), the polarization pair $\hat e_s=\hat{\mathbf n}\times\mathbf g/\lvert\hat{\mathbf n}\times\mathbf g\rvert$, $\hat e_p=\hat e_s\times\hat{\mathbf n}$ (normalized), the projections $\mathbf g\cdot\hat e_{s,p}$ and $\mathbf k\cdot\mathbf g$, the PXR detuning, the CBS/PXR amplitudes, the sinc width (through $E_{\rm res}$), and $\mu(E_{\rm res})$ and $\chi_{\mathbf g}(E_{\rm res})$ evaluated at the shifted energy. The tabulated $\chi_{\mathbf g}(E)$, $U_{\mathbf g}(E)$ depend only on $hkl$ and energy, and $\lvert\mathbf g\rvert^2$ is rotation invariant, so both stay per reflection. The escape geometry along $\hat{\mathbf n}$ is $\mathbf g$-independent.

### 5. Rotation convention

The tilt must be the same rotation the GH route uses: Rodrigues rotation about the in-plane axis $\hat{\boldsymbol\kappa}=(t_x,t_y,0)/\lvert\mathbf t\rvert$ by angle $\lvert\mathbf t\rvert$,

$$
R(\mathbf t)=I+\sin\lvert\mathbf t\rvert\,[\hat{\boldsymbol\kappa}]_\times'+(1-\cos\lvert\mathbf t\rvert)\,[\hat{\boldsymbol\kappa}]_\times'^{\,2},
$$

applied in the lab frame after the mean crystal orientation $R_{\rm orient}$: $\mathbf g_{e,k}=R(\mathbf t_{e,k})R_{\rm orient}\mathbf g_{hkl}$. (The sign convention of the skew matrix $[\cdot]'_\times$ is whatever `mosaic-mc` uses; because $p_\eta$ is symmetric under $\mathbf t\to-\mathbf t$ any consistent convention gives the same average, but the bit-level comparison requires the same one.)

### 6. Flight-grouped reduction

In the grouped reduction each flight's rows add as fields with phase $\phi_j=d_j\omega-\mathbf g\cdot\mathbf r_j-\dots$, and the reduction subtracts the flight's first row, $\phi_j-\phi_{\rm first}$. That reference is a common phase only if every row of the flight carries the same $\mathbf g$; then $\mathbf g\cdot\mathbf r_j-\mathbf g\cdot\mathbf r_{\rm first}=\mathbf g\cdot(\mathbf r_j-\mathbf r_{\rm first})$ and the subtraction is exact. A flight belongs to a single particle, hence a single `elec_id`, hence a single crystallite per draw, so the condition holds under per-history sampling (it would fail under per-row sampling, which would also be physically wrong: substeps of one flight interfere and must see one lattice).

### 7. Coherent route

The coherent route sums fields across electrons (pair terms). Under per-history sampling two histories carry different crystallites; summing their amplitudes would interfere radiation from different lattices with an orientation-mismatched phase, which is not the mosaic model (incoherent ensemble). The per-history draw is therefore defined only for incoherent tallies; refusing the coherent route is the correct scope restriction, not an approximation.

### 8. Limits

- $\eta\to0$: $\sigma=0$, every tilt is exactly zero, and the estimator is the perfect crystal. With $\eta$ exactly zero the route should be the perfect-crystal path itself (bit-for-bit).
- $K=1$: plain Monte Carlo, still unbiased.
- Constant $S$: preserved exactly, because $\sum_k 1/K=1$.
- Histories $\to\infty$: the sample mean converges to the GH $N\to\infty$ limit in every linear functional (yield, first moment); ratio functionals (centroid, RMS width) are consistent with $O(1/N_e)$ ratio bias.

Units: tilts, $\sigma$ and $\eta$ in radians (dimensionless); weights dimensionless; the spectrum keeps its units.

## Implementation comparison

| Item | Derivation | Code | Result |
|---|---|---|---|
| $\sigma$ | $\eta/(2\sqrt{2\ln 2})$ | `_FWHM_TO_SIGMA = 1/2.3548200450309493` | identical to float64 (rel. diff $0$) |
| shift $s_e$ | two independent uniforms per history | `counter_uniforms(root, ..., 2, start=lo)[ids - lo]`, root `child_stream_root(seed, 8, 7)`; uniforms $(m+\tfrac12)2^{-52}$ | match; depends only on `(seed, elec_id)` |
| lattice $p_k$ | any rank-1 lattice | `(i/K, (i a mod K)/K)`, Korobov $a\approx K(\sqrt5-1)/2$ coprime to $K$ | match; generator cannot bias |
| $u$, clip, quantile | $(p+s)\bmod 1$, $\sigma\Phi^{-1}$ | `np.mod(...)`, `np.clip(u, 2**-53, 1-2**-53)`, `ndtri` | match; bias $\lesssim10^{-13}$ |
| rotation | $R(\mathbf t)R_{\rm orient}\mathbf g$ | `tilt_rotate(R_orient @ g, dx, dy)`; GH: `R_m @ (R_orient @ g)` | same composition; same skew matrix as `_small_tilt_R` |
| polarization | per-row $\hat{\mathbf n}\times\mathbf g$ pair, same fallback | `polarization_pairs` | equal to scalar `_polarization_pair` |
| weight | $1/K$ per draw | `wm = 1.0 / sampled.draws`, multiplied into every target (`weight`, PXR/CBS components, temporal boxes, grouped `wm`) | match |
| per-row recompute | §4 list | `_accumulate_reflection`: `dot_g(v_all, g)`, per-row `n_dot_g` into recoil terms and in-medium root, `kept_orientation` restricts $\mathbf g$, $\hat e_{s,p}$, projections to kept rows; $\chi,U,\mu$ interpolated at per-row $E_r$; `g2` per reflection | match |
| history map | row $\to$ its `elec_id`'s draws | `np.unique(..., return_inverse=True)`; tilts keyed by the unique ids | match |
| grouped phase | common $\mathbf g$ per flight | `_referenced_phases`: per-row `(dr * g[sel]).sum(axis=1)` after `minus_first`; flight rows share `elec_id` | match |
| coherent | refuse | `_refuse_mosaic_sampling` raises `NotImplementedError` when `coherent`; also requires `mosaic_seed` | match |
| $\eta=0$ | perfect crystal | `mosaic_orientations`: falsy `mosaic_fwhm_rad` goes to `_mosaic_quadrature`, which returns `None` | bit-for-bit (anchor test) |
| line-grid seeds | same tilts as the spectrum | `reflection_rows._seed_orientations` builds `sampled_mosaic(owner=elec_id, ...)` with weights $1/K$ | match |

Route selection: `_needs_per_hkl_route` returns true whenever `st.mosaic_sampled` is set, so the batched table route (which has no per-row $\mathbf g$) is never reached. The coverage screen (`_coverage._screening_orientations`) substitutes GH nodes of the same width under the sampled route; it is an opt-in, count-only diagnostic and does not touch yield.

### Numeric spot checks (scratch, CPU)

Script `scratch/mms_verify*.py` in the task worktree; independent of the anchor tests.

1. **Marginal law.** 200 000 histories, $K=5$, $\eta=3.5^\circ$: every draw and axis has standardized mean within $\pm0.006$, standard deviation $0.998$–$1.002$, Kolmogorov–Smirnov $p\in[0.07,0.60]$; $x$–$y$ correlation $-0.0013$. The $K=5$ lattice is $\{(0,0),(0.2,0.6),(0.4,0.2),(0.6,0.8),(0.8,0.4)\}$.
2. **Unbiasedness on a nonlinear integrand.** $f(\mathbf t)$ = off-centre Lorentzian of a tilt-shifted line plus a $\cos(t_x/\sigma)\,t_y/\sigma$ cross term; reference: 80-point product GH ($0.243311$). Sampled means over $10^5$ histories: $z=-0.03,+0.53,-0.63,-0.50$ for $K=1,2,4,8$; standard error falls from $2.6\times10^{-3}$ to $6.1\times10^{-4}$ (stratification gain beyond $1/\sqrt K$).
3. **Rotation and polarization.** 500 random tilts: `tilt_rotate` vs `_small_tilt_R @ g` max abs difference $1.1\times10^{-16}$; zero tilt returns $\mathbf g$ exactly; `polarization_pairs` vs `_polarization_pair` difference $0$.
4. **Per-row machinery vs explicit rotations.** All histories forced onto one crystallite (patched `history_tilts`), HOPG (002)+(004), 20 electrons, $30^\circ$ miscut, $K=3$: the sampled spectrum equals the GH-path spectrum with those three rotations at weight $1/3$ to $1.1\times10^{-14}$ L1-relative; after `subdivide_flights` (328 substep rows, flight-grouped reduction) to $8.8\times10^{-15}$.
5. **History mapping.** Even/odd histories assigned tilt sets $A$/$B$: $S(A,B)+S(B,A)=S(A,A)+S(B,B)$ to $1.6\times10^{-16}$, while $S(A,B)$ differs from $S(A,A)$ by $0.31$ — rows are routed to their own history's crystallite and the contributions are additive per history.
6. **Anchor tests.** `tests/montecarlo/spectrum/test_mosaic_sampled.py`: 13 passed (includes the 16-seed GH $N=33$ moment comparison within 5 standard errors and $\eta=0$ bit-for-bit).

An attempted per-history decomposition by remapping other histories' `elec_id` out of range did not reproduce the full perfect-crystal spectrum either (L1 $1.1\times10^{-3}$ with no mosaic at all), so that masking is not additive for reasons outside this claim; checks 4–5 avoid it.

### Findings outside the physics

- **Stale ledger anchor.** The row's **Code** field names `montecarlo/spectrum/cxr/_per_hkl.py::_accumulate_sampled_mosaic`, which does not exist; the function is `accumulate_sampled_mosaic` in `_mosaic_sampled.py` (already covered by the module entry). `tests/dev/test_validation_ledger.py::test_ledger_python_owners_have_matching_markers` fails on exactly this.
- **Toctree.** This write-up is not yet listed in `docs/validation/materials/index.md`.
- The row could also name `_grouped.py::_referenced_phases` and `_per_hkl.py::_accumulate_reflection`, which carry the per-row $\mathbf g$ logic; the former's docstring has the `mosaic-mc-sampled` marker, the latter mentions it only parenthetically.

## Verdict

- **Claim**: `mosaic-mc-sampled` — `montecarlo/spectrum/cxr/_mosaic_sampled.py::{history_tilts, accumulate_sampled_mosaic}` with per-row $\mathbf g$ in `_per_hkl.py::_accumulate_reflection` and `_grouped.py::_referenced_phases` — per-history Cranley–Patterson-shifted rank-1 lattice, $\sigma\Phi^{-1}$ per axis, weight $1/K$, as an unbiased estimator of the `mosaic-mc` Gaussian orientation average (`docs/physics/materials/crystal-mosaicity.md`, "Sampled crystallites")
- **Filters**: units `pass`; limits `pass` ($\eta=0$ bit-for-bit, $K=1$ plain MC, constant preserved, sampled mean $\to$ GH $N\to\infty$); signs/conventions `pass` (same Rodrigues skew and lab-frame composition $R(\mathbf t)R_{\rm orient}$ as GH; same polarization fallback)
- **Re-derivation**: `matches` — no divergent factor; each draw is an exact $\mathcal N(0,\sigma^2)^{\otimes2}$ sample (clip bias $\lesssim10^{-13}$), weights $1/K$ sum to one per history, every orientation-dependent term is per row while $\chi_{\mathbf g}$, $U_{\mathbf g}$, $\lvert\mathbf g\rvert^2$ stay per reflection, the grouped phase reference is exact because a flight shares its history's $\mathbf g$, and per-history granularity changes only the covariance
- **Verdict**: `rederived`
- **Write-up**: `docs/validation/materials/mosaic-mc-sampled.md`
- **Suggested ledger change**: status `filtered` → `rederived`; fix **Code** `montecarlo/spectrum/cxr/_per_hkl.py::_accumulate_sampled_mosaic` → `montecarlo/spectrum/cxr/_mosaic_sampled.py::accumulate_sampled_mosaic` (optionally add `montecarlo/spectrum/cxr/_grouped.py::_referenced_phases`); append to **Notes** "fresh-context re-derivation: draws exact Gaussian per axis, $1/K$ weighting reproduces the GH-normalized mean, per-row machinery equals explicit $R(\mathbf t)$ rotations to $10^{-14}$ (incoherent and grouped), history additivity to $10^{-16}$; `[validation write-up](materials/mosaic-mc-sampled.md)`"; add the page to the `docs/validation/materials/index.md` toctree. Never `signed-off`.
