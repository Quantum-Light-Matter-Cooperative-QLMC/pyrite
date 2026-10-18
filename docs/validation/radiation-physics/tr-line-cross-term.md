# `tr-line-cross-term`

## Claim and source

- Claim: the ensemble average of the interference between TR (the $\mathbf g=0$, homogeneous-$\varepsilon$ field) and a coherent PXR/CBS field of reciprocal vector $\mathbf g$ is multiplied by the characteristic function of the beam entry distribution at $\mathbf g_\parallel$, stated as $\exp(-(g_\parallel\sigma_\perp)^2)$ for a Gaussian spot. TR and the coherent lines are therefore additive for every $\mathbf g$ with $\mathbf g_\parallel\neq0$; "surface-parallel $\mathbf g$" and a point spot are excluded.
- Code: none (component-accounting decision). It is documented in `docs/physics/radiation-physics/transition-radiation.md` § Interference with PXR and CBS.
- Sources: lateral translation invariance of amorphous transport through a laterally infinite stack. Line phase $e^{i[\omega t-(\omega\hat{\mathbf n}+\mathbf g)\cdot\mathbf r]}$ of `coherent-emission`; TR crossing phase $e^{i(\omega t-\mathbf k_\perp\cdot\mathbf r_\perp)}$ of `tr-crossing-superposition`.
- Units: $\hbar=c=1$, $\omega=E/\hbar c$ [Å$^{-1}$], lengths in Å. $\sigma_\perp$ is the per-axis r.m.s. entry width, the convention of `coherent-emission.md` § The transverse partner.
- Frozen revision: commit `0984a60`.

## Independent derivation

### Phases of the two fields under a lateral shift

Write one history $h$ (all tracks) and shift it rigidly by $\boldsymbol\Delta=(\Delta_x,\Delta_y,0)$ in the surface plane, with the lattice held fixed. Suppose the shifted history is equally likely and its amplitudes depend on the shift only through phases. Then:

- **TR.** The stack is laterally homogeneous, so the free waves conserve $\mathbf k_\parallel=\omega\hat{\mathbf n}_\parallel$ (Snell) in every layer. The vacuum amplitude acquires $e^{-i\omega\hat{\mathbf n}_\parallel\cdot\boldsymbol\Delta}$.
- **Line ($\mathbf g$).** The segment phase is $\omega t_{{\rm abs},j}-(\omega\hat{\mathbf n}+\mathbf g)\cdot\mathbf r_j-\delta\,\omega L_{{\rm esc},j}$. The amplitude $A_j$, the finite-time factor $Q_j$ and the escape transmission depend on velocity, energy and depth only. $L_{{\rm esc},j}$ depends only on $z_j$ for planar faces. The field acquires $e^{-i(\omega\hat{\mathbf n}+\mathbf g)_\parallel\cdot\boldsymbol\Delta}$.

The time phase $e^{i\omega t}$ is common to both fields: same transport clock, same $t_0$, and the same face-arrival delay on a tilted face. The single-electron cross term $\mathbf S_{\rm TR}\cdot\mathbf S_{\mathbf g}^\ast$ thus acquires exactly $e^{+i\mathbf g_\parallel\cdot\boldsymbol\Delta}$. The $\omega$-dependent parts cancel because both fields use the **vacuum** $\omega\hat{\mathbf n}$ at one detector direction. The refractive part sits only in the depth-dependent $\delta\omega L_{\rm esc}(z)$, and on the TR side in $k_z$ inside the layers.

### Ensemble average, one electron

Write $\mathbf S(h)=\mathbf S(h_0)\times$phase, where $h_0$ is the history's shape relative to its entry point. If the shape's law is independent of the entry point $\boldsymbol\Delta\sim\rho$,

$$
\mathrm E\bigl[\mathbf S_{\rm TR}\cdot\mathbf S_{\mathbf g}^\ast\bigr]
=\mathrm E_{h_0}\bigl[\mathbf S_{\rm TR}^0\cdot\mathbf S_{\mathbf g}^{0\ast}\bigr]\;\tilde\rho(\mathbf g_\parallel),
\qquad \tilde\rho(\mathbf q)=\mathrm E\,e^{i\mathbf q\cdot\boldsymbol\Delta}.
$$

For a round Gaussian spot of per-axis r.m.s. $\sigma_\perp$,

$$
\boxed{\tilde\rho(\mathbf g_\parallel)=\exp\bigl[-\tfrac12(g_\parallel\sigma_\perp)^2\bigr]}
$$

On a tilted face $\rho$ is the face-point law, with covariance $J\Sigma J^{\mathsf T}$.

The cross term is **linear** in each field, so the characteristic function enters once, not squared. $\exp[-(q\sigma_\perp)^2]=\lvert\tilde\rho\rvert^2$ is the factor for the *squared* modulus of a mean field, as in the inter-electron line terms.

### Cross-electron terms of a bunch

For $e\neq f$, independence gives $\mathrm E[\mathbf S_{{\rm TR},e}\cdot\mathbf S_{\mathbf g,f}^\ast]=\mathrm E[\mathbf S_{\rm TR}]\cdot\mathrm E[\mathbf S_{\mathbf g}]^\ast$. With normal incidence and no front tilt, each mean carries its own factor: $\tilde\rho(-\mathbf K)$ with $\mathbf K=\omega\hat{\mathbf n}_\parallel$ for TR, and $\tilde\rho(-\mathbf K-\mathbf g_\parallel)$ for the line, times the common longitudinal $\lvert\mathrm E\,e^{i\omega t_0}\rvert^2$. For the Gaussian,

$$
\bigl\lvert\tilde\rho(\mathbf K)\,\tilde\rho(\mathbf K+\mathbf g_\parallel)^\ast\bigr\rvert
=e^{-\frac12\sigma_\perp^2\left(K^2+\lvert\mathbf K+\mathbf g_\parallel\rvert^2\right)}
\le e^{-\frac14(g_\parallel\sigma_\perp)^2},
$$

since $K^2+\lvert\mathbf K+\mathbf g\rvert^2\ge g^2/2$. With the general $\mathbf K=\omega(\hat{\mathbf n}-\mathbf b/\beta-J^{-\mathsf T}\mathbf a)_{xy}$ of `transverse-bunch-form-factor`, both factors shift by the same $\mathbf K$ and the bound still holds. Cross-electron TR–line terms are therefore suppressed by at least $e^{-(g_\parallel\sigma_\perp)^2/4}$ per pair. In the bunch estimator that bound is weighted by $N-1$. They are **not** multiplied by $\tilde\rho(\mathbf g_\parallel)$ itself.

### When the argument fails

| Case | Why |
|---|---|
| $\mathbf g_\parallel=0$: $\mathbf g$ along the surface normal, lattice planes **parallel** to the surface (symmetric Bragg) | The shift phase is trivial. The term depends on the termination and the absolute phase of $A_{\rm PXR}+A_{\rm CBS}$. |
| $g_\parallel\sigma_\perp\lesssim$ a few (small miscut, mosaic nodes near zero tilt of a $(00l)$ reflection) | $\tilde\rho$ is continuous in $\mathbf g_\parallel$. The required condition is $g_\parallel\sigma_\perp\gg1$ (cross-electron: $(N-1)e^{-(g_\parallel\sigma_\perp)^2/4}\ll1$), not $\mathbf g_\parallel\neq0$. |
| Point spot | $\tilde\rho\equiv1$. |
| Channeling transport | The trajectory law depends on $\boldsymbol\Delta$ modulo the lattice. The average gives the lattice Fourier coefficient of that dependence at $\mathbf g_\parallel$, which is nonzero in general. |
| Finite footprint, side exits, grooves | Lateral invariance fails near edges: $L_{\rm esc}$, attenuation and hit/miss depend on $\boldsymbol\Delta$. The TR side assumes laterally infinite interfaces. Grooves are rejected by `boundary_crossings`; finite footprints are not. |

Symmetric **Laue** geometry has the planes perpendicular to the surface, so $\mathbf g$ lies in the surface plane and $g_\parallel=\lvert\mathbf g\rvert$. That is the most strongly suppressed case, not an exclusion.

## Comparison with the implementation conventions

- Line phase (`cxr/_setup.py:519`, `:566`): $d_j=t_{\rm mid}+t_0-\hat{\mathbf n}\cdot\mathbf r_j$ with the vacuum $\hat{\mathbf n}$. Kernel phase `ps * E0 - gp - Lj * dw0` (`coherent_jit_kernel.py:171`), with `ps` $=d_j/\hbar c$ (`cxr/_batched.py:902`), `gp` $=\mathbf r_j\cdot\mathbf g$ (`cxr/_per_hkl.py:648`) and `dw` $=(1-\mathrm{Re}\,n)\,\omega$ (`cxr/_setup.py:531-537`). The spatial term is therefore vacuum $\omega\hat{\mathbf n}$, not $\mathrm{Re}\,n\,\omega\hat{\mathbf n}$. The medium enters only through $\delta\omega L_{\rm esc}$, whose gradient is along the face normal (`segment_escape_gradient`, `cxr/_setup.py:498-502`). No $\omega$-dependent in-plane term survives in the relative phase.
- Frame: $\mathbf g$ is rotated into the sample (slab) frame by `R_orient` and then by each mosaic node `R_m` (`cxr/_per_hkl.py:1183-1190`, likewise `cxr/_batched.py:263`). `seg_r` is `r_mid` in the same frame (`cxr/_setup.py:406`), including the sampled entry point. Under `physical_electrons` the infinite-slab path pairs complete fields (`cxr/_setup.py:686-690`), exactly as `coherence="bunch"` does for TR.
- TR phase: `interface_transition_radiation_amplitude` uses `k_perp = omega * n[:, :2]` (`transition_radiation.py:408`). Numerically, shifting a history by $(\delta t,\boldsymbol\Delta)$ multiplies its summed amplitude by $e^{i(\omega\delta t-\omega\hat{\mathbf n}_\parallel\cdot\boldsymbol\Delta)}$ to $9\times10^{-13}$, forward and backward. This was checked with the scratch script `ctr_check.py` (see [`tr-coherent-bunch`](tr-coherent-bunch.md)).
- Factor check (`cross_term.py`, scratch): for $\sigma=1.3$ Å and $g_\parallel=0.985$ Å$^{-1}$, a $2\times10^6$-sample average of $e^{i\mathbf g_\parallel\cdot\boldsymbol\Delta}$ gives $0.4408$. Compare $e^{-(g\sigma)^2/2}=0.4406$ and the stated $e^{-(g\sigma)^2}=0.194$. The cross-electron product at $\mathbf K\approx-\mathbf g_\parallel$ gives $0.4878$, against the predicted $0.4876$ and the bound $0.664$.

## Adjudication

The translation argument and the additivity decision are sound for $g_\parallel\sigma_\perp\gg1$. Both fields use the same vacuum $\omega\hat{\mathbf n}_\parallel$, so only $\mathbf g_\parallel$ survives. $\mathbf g$ and $\mathbf r_j$ share the slab frame. Escape refraction is depth-only. The claim as written still diverges in these places:

1. **First divergent factor: the exponent.** The single-electron factor is $\tilde\rho(\mathbf g_\parallel)=\exp[-\frac12(g_\parallel\sigma_\perp)^2]$ for per-axis r.m.s. $\sigma_\perp$, not $\exp[-(g_\parallel\sigma_\perp)^2]$. The stated form is $\lvert\tilde\rho\rvert^2$, the factor of a squared mean. This is numerically immaterial for $g_\parallel\sim1$ Å$^{-1}$, but it is an error in the ledgered expression.
2. **Cross-electron extension.** The page says the same factor "holds for the cross-electron TR–line terms". The correct factor is $\tilde\rho(\mathbf K)\tilde\rho(\mathbf K+\mathbf g_\parallel)^\ast$, bounded by $e^{-(g_\parallel\sigma_\perp)^2/4}$ and weighted by $N-1$. The page also calls it "the same factor that removes inter-electron line terms" (`coherent-emission.md` § The transverse partner). That one is $\exp[-(q_\perp\sigma_\perp)^2]$ with $\mathbf q_\perp=(\omega\hat{\mathbf n}+\mathbf g)_\perp$: a different argument and power.
3. **Exclusion wording.** The ledger's "surface-parallel $\mathbf g$" and the Notes' "Surface-parallel reflections (symmetric Bragg/Laue)" are wrong, and so are the page's "(symmetric Bragg or Laue geometry)" and the decision table's "surface-parallel reflections". The excluded set is $\mathbf g$ along the surface normal (planes parallel to the surface, symmetric Bragg). Symmetric Laue has $g_\parallel=\lvert\mathbf g\rvert$. The condition should be $g_\parallel\sigma_\perp\gg1$, which matters for small miscuts and near-zero-tilt mosaic nodes, not $\mathbf g_\parallel\neq0$.
4. **Unstated exclusions.** Channeling transport, finite footprints with spot or transport reach near an edge, and grooved faces. The claim's premise covers these, but the "Not covered" list should name them. The flagship default geometry, a $(00l)$ basal reflection of HOPG/hBN with $[001]$ along the slab normal, has $\mathbf g_\parallel=0$ exactly, as does the central mosaic node. The excluded case is therefore the main science configuration, and the page should say so.

### DTR statement

The page says the $\mathbf g_\parallel=0$ geometry "also produces" DTR and that a cross term without DTR is incomplete. The second half is right. The first half is misleading. DTR is the boundary term carrying $\chi_{\mathbf g}$ and exists for every $\mathbf g\neq0$, in Bragg and Laue geometry. Under the same lateral shift it acquires $e^{-i(\omega\hat{\mathbf n}+\mathbf g)_\parallel\cdot\boldsymbol\Delta}$, exactly like PXR. The DTR–PXR interference is therefore **not** suppressed by $\tilde\rho$ for any $\mathbf g$, while the TR–DTR term is suppressed like TR–PXR. The page's own § Diffracted TR and PXR estimate ($\lvert A_{\rm DTR}/A_{\rm PXR}\rvert\sim\lvert\chi_0\rvert/(\gamma^{-2}+\theta^2)$) applies to all reflections. "TR adds to PXR/CBS for $\mathbf g_\parallel\neq0$" is correct for the $\mathbf g=0$ TR, but the PXR line itself is incomplete without DTR whenever $\gamma^{-2}\sim\lvert\chi_0\rvert$, regardless of $\mathbf g_\parallel$. The decision table should not imply that DTR is needed only for surface-normal $\mathbf g$.

Original verdict: **discrepancy** (exponent factor 2 and exclusion wording); superseded by the re-check below. These are documentation-only fixes; the additivity decision survives. After the edits below, the derivation here supports `rederived`.

### Suggested page and ledger edits (owner)

- Replace $e^{-(g_\parallel\sigma_\perp)^2}$ by $e^{-(g_\parallel\sigma_\perp)^2/2}$ (per-axis r.m.s.), and state that it is the characteristic function, entering linearly.
- Cross-electron: "suppressed by $\tilde\rho(\mathbf K)\tilde\rho(\mathbf K+\mathbf g_\parallel)^\ast$, at most $e^{-(g_\parallel\sigma_\perp)^2/4}$ per pair". Drop "the same factor that removes inter-electron line terms".
- "Surface-parallel $\mathbf g$" → "$\mathbf g$ along the surface normal (lattice planes parallel to the surface, symmetric Bragg)". Remove "Laue". State the condition as $g_\parallel\sigma_\perp\gg1$.
- Add channeling, finite footprints near an edge and grooves to "Not covered". Note that default-orientation $(00l)$ reflections and the central mosaic node fall in the excluded case.
- DTR: it exists for every $\mathbf g$, and its PXR interference is unsuppressed. The surface-normal case needs it for the TR–line term specifically.

## Re-check of the corrected text (2026-10-18)

Re-read against the derivation above: `docs/physics/radiation-physics/transition-radiation.md` § Interference with PXR and CBS, the decision-table CTR row, and the ledger row (working tree after `0984a60`).

| Item | Corrected text | Matches |
|---|---|---|
| Single-electron factor | $\tilde\rho(\mathbf g_\parallel)=e^{-(g_\parallel\sigma_\perp)^2/2}$, per-axis r.m.s., linear in each field | yes |
| Cross-electron factor | $\tilde\rho(\mathbf K)\tilde\rho^\ast(\mathbf K+\mathbf g_\parallel)\le e^{-(g_\parallel\sigma_\perp)^2/4}$ per pair, distinguished from $e^{-(q_\perp\sigma_\perp)^2}$ | yes |
| Condition | $g_\parallel\sigma_\perp\gg1$, with the small-miscut and near-zero-tilt mosaic caveat | yes |
| Exclusions | surface-normal $\mathbf g$ (symmetric Bragg, default $(00l)$ rows, central mosaic node); point spot; channeling; finite footprints near an edge; grooves. Laue is most suppressed | yes |
| Phase conventions | vacuum $\omega\hat{\mathbf n}$ in both fields, slab-frame $\mathbf g$, depth-only escape phase | yes |
| DTR | exists for every $\mathbf g\neq0$; its PXR interference is unsuppressed; not implemented | yes |

Two residual nits, neither a physics divergence:

- The page writes $\mathbf K=\omega\hat{\mathbf n}_\parallel$. That is the normal-incidence, untilted-front value. On a tilted face or with a bunch-front gradient, $\mathbf K=\omega(\hat{\mathbf n}-\mathbf b/\beta-J^{-\mathsf T}\mathbf a)_{xy}$. Both factors shift by the same $\mathbf K$, so the $e^{-(g_\parallel\sigma_\perp)^2/4}$ bound is unchanged (with $\sigma_\perp$ read from the face covariance).
- The ledger Notes keep "Escape refraction of the line field is a smooth function of `w` and does not change the `g_parallel` argument". The operative reason is that $L_{\rm esc}$ depends only on depth for planar faces, as the page now states. Smoothness in $\omega$ is not the reason. Suggested rewording: "Escape refraction of the line field depends only on depth for planar faces and adds no in-plane phase."

Updated verdict: **rederived** for the corrected claim text. The task owner applies the status. Human sign-off pending.
