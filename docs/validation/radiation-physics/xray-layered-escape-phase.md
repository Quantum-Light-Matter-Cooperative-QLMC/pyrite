(validation-xray-layered-escape-phase)=
# xray-layered-escape-phase

## Claim (from the ledger row, before reading implementation bodies)

- **ID:** `xray-layered-escape-phase` (issue #1180, commit `76feb0a90`)
- **Quantity:** through a $K$-layer absorber stack, the escape-leg phase of a flight-grouped CXR row is $-\omega(E)\sum_l\delta_l(E)L_l$, with $\delta_l=1-\operatorname{Re}\sqrt{1+\chi_{0,l}}$, $\chi_{0,l}=-(r_e\lambda^2/\pi)\sum_i n_i(f_{1,i}+if_{2,i})$ from layer $l$'s composition, and $L_l$ the straight escape path along $\hat{\mathbf n}$ inside layer $l$. On each linear escape piece every $L_l$ is affine, so the formation argument becomes $v=a_{\rm vac}(E-E_{\rm vac})-\sum_l\delta_l\omega\,\Delta L_l/2$ and the attenuation $\tau=\sum_l\mu_lL_l$.
- **Builds on:** `xray-in-medium-propagation-phase`, `xray-chi-zero`, `self-absorption`, `coherent-formation-absorption`.
- **Units:** lengths in Å, photon energy $E$ in eV, $\omega=E/\hbar c$ in Å$^{-1}$, $n_i$ in atoms/Å$^3$, $r_e$ and $\lambda$ in Å; $\delta_l$, $v$, $\tau$ dimensionless.
- **Stated assumptions:** kinematic emission; straight, refraction-free escape along $\hat{\mathbf n}$; sharp planar interfaces; no interface reflection or transmission loss; the in-medium resonance root keeps the emitting crystal's $\delta$.

## Independent derivation

### Per-layer index

For a medium of atomic number densities $n_i$ (Å$^{-3}$), the forward-scattering susceptibility is the $\mathbf g=0$ term of `xray-chi-zero` with the unit-cell sum $\sum_{\rm cell}f_i/V_{\rm cell}$ replaced by $\sum_i n_if_i$:

$$
\chi_{0,l}=-\frac{4\pi r_e}{k^2}\sum_in_if_i(E)
=-\frac{r_e\lambda^2}{\pi}\sum_in_i\bigl(f_{1,i}+if_{2,i}\bigr),
\qquad k=\frac{2\pi}{\lambda}=\omega .
$$

Then $n_l=\sqrt{1+\chi_{0,l}}$ and $\delta_l=1-\operatorname{Re}n_l\approx(r_e\lambda^2/2\pi)\sum_in_if_{1,i}>0$. With $\lambda=2\pi\hbar c/E$, $\lambda^2\propto E^{-2}$; $\delta_l$ is dimensionless (Å $\cdot$ Å$^2\cdot$ Å$^{-3}$). If the layer has the crystal's basis composition at the crystal's density, $\sum_in_if_i=\sum_{\rm cell}f_i/V_{\rm cell}$ (forward, so no Debye–Waller or structure phase), and $\delta_l$ equals the crystal's $\delta$ exactly.

### Escape-leg phase through $K$ layers

Use $c=1$, so $t$ and lengths share Å and $\omega$ is in Å$^{-1}$. A row (point emitter) at $(t_j,\mathbf r_j)$ radiates toward a far detector at $R\hat{\mathbf n}$. Along the straight escape ray, the photon crosses layer $l$ over a geometric length $L_l(\mathbf r_j)\ge0$ (zero for layers the ray does not cross) and then the vacuum leg $L_{\rm vac}$. The Fourier phase of the observation-time kernel $e^{i\omega t_{\rm obs}}$ is

$$
\Phi_j=\omega\Bigl(t_j+\sum_l\operatorname{Re}n_l\,L_l+L_{\rm vac}\Bigr)-\mathbf g\cdot\mathbf r_j .
$$

The straight path is $\sum_lL_l+L_{\rm vac}=\lvert R\hat{\mathbf n}-\mathbf r_j\rvert=R-\hat{\mathbf n}\cdot\mathbf r_j+O(1/R)$. Substituting $\operatorname{Re}n_l=1-\delta_l$,

$$
\Phi_j=\omega R+\omega\bigl(t_j-\hat{\mathbf n}\cdot\mathbf r_j\bigr)-\mathbf g\cdot\mathbf r_j
\;\boxed{-\;\omega\sum_l\delta_lL_l(\mathbf r_j)} .
$$

$\omega R$ is common to all rows of a reflection and drops from $\lvert\sum\rvert^2$. The imaginary part of each $n_l$ multiplies the amplitude by $e^{-\omega\operatorname{Im}(n_l)L_l}$ (sign set by the absorbing branch), i.e. $e^{-\mu_lL_l/2}$ with $\mu_l=2\omega\lvert\operatorname{Im}n_l\rvert$. Over the stack the amplitude is $e^{-\tau/2}$ with $\tau=\sum_l\mu_lL_l$, which is the Beer–Lambert stack attenuation of `self-absorption`. Phase and attenuation act on the same per-layer paths. Taking the real part of $n_l$ only in the phase avoids double counting the absorption already in $\tau$.

### Affine pieces and the formation argument

Let a row be a straight piece $\mathbf r(t')=\mathbf r_c+\mathbf v\,t'$, $t'\in[-T/2,T/2]$. For a planar stack, each $L_l(\mathbf r)$ is piecewise affine in $\mathbf r$. Its kinks are where the emitter crosses an interface, or where the exit face changes for a finite lateral extent. Between kinks, every $L_l$ is affine in $t'$: $L_l=L_{l,c}+(\Delta L_l/T)t'$, with $\Delta L_l=L_l(\text{end})-L_l(\text{start})$ ordered along travel. As in `coherent-formation-absorption`, with $s=2t'/T$,

$$
\frac{d\Phi}{ds}=\frac T2\bigl[\omega(1-\mathbf v\cdot\hat{\mathbf n})-\mathbf v\cdot\mathbf g\bigr]-\omega\sum_l\delta_l\frac{\Delta L_l}{2}
=a_{\rm vac}(E-E_{\rm vac})-\sum_l\delta_l(E)\,\omega(E)\,\frac{\Delta L_l}{2}\equiv v,
$$

$$
\frac{\tau}{2}=\frac{\tau_c}{2}+qs,\qquad q=\frac{\tau_{\rm end}-\tau_{\rm start}}{4}=\frac14\sum_l\mu_l\Delta L_l .
$$

The formation factor $F=e^{-\tau_c/2}\sinh w/w$ with $w=iv-q$ is otherwise unchanged. The inter-piece phase uses the midpoint value $-\omega\sum_l\delta_lL_{l,c}$, where $L_{l,c}=(L_l^{\rm start}+L_l^{\rm end})/2$ is exact because each $L_l$ is affine. Pieces must be cut at every emitter interface crossing. Otherwise the slope of the emitter's own layer (and of layers it leaves) changes inside the piece, and neither $v$ nor $q$ is constant.

### Planar front-escape geometry (closed form for the checks)

Take layers $0\le z<z_1$ (layer 1), $z_1\le z<z_2$ (layer 2), …, and escape through the entrance face $z=0$ with $n_z<0$. For an emitter at depth $z$,

$$
L_l(z)=\frac{\max\bigl(0,\min(z,z_l)-z_{l-1}\bigr)}{\lvert n_z\rvert}.
$$

For $K=2$, this gives $L_1=\min(z,z_1)/\lvert n_z\rvert$ and $L_2=\max(z-z_1,0)/\lvert n_z\rvert$. For escape through the back face of a finite stack, the same expression applies with depths measured from that face. Two rows at depths $z_a,z_b$ in a C film ($0<z<d$) on a Si substrate, at a common emission time-and-transverse geometry, differ in escape phase by

$$
\Delta\Phi_{ab}=-\frac{\omega}{\lvert n_z\rvert}\Bigl[\delta_{\rm C}\bigl(\min(z_a,d)-\min(z_b,d)\bigr)+\delta_{\rm Si}\bigl(\max(z_a-d,0)-\max(z_b-d,0)\bigr)\Bigr].
$$

### Limiting cases (derived)

1. **$K=1$:** $-\omega\delta L_{\rm esc}$, $v=a_{\rm vac}(E-E_{\rm vac})-\delta\omega\Delta L/2$, $\tau=\mu L$. This is exactly `xray-in-medium-propagation-phase` plus `coherent-formation-absorption`.
2. **Identical adjacent layers ($\delta_1=\delta_2$, $\mu_1=\mu_2$):** $\sum_l\delta_lL_l=\delta(L_1+L_2)$. The internal interface only adds a piece cut, and since the grouped sum is split invariant the pieces reassemble the uncut result.
3. **Transparent, non-refracting overlayer ($\delta_l,\mu_l\to0$):** the layer contributes nothing to $v$, $\tau$ or the phase, so it acts as vacuum.
4. **Normal exit ($\hat{\mathbf n}=-\hat{\mathbf z}$):** $\Delta L_l=-\hat{\mathbf n}\cdot\Delta\mathbf r$ inside the emitter's layer, so the refractive slope is $\delta_{\rm emit}\omega\,\mathbf v\cdot\hat{\mathbf z}$. Only the emitter's own layer contributes to the slope, and the layers above add a constant phase per piece.
5. **Sign:** $\delta_l>0$ lowers the optical path's phase relative to vacuum. A deeper row (larger $L$) lags by $-\delta\omega\,\Delta L$, the same sign as the single-slab claim.

## Cheap filters

| Filter | Result |
|---|---|
| Units | pass: $r_e\lambda^2n_i$ is dimensionless (Å$\cdot$Å$^2\cdot$Å$^{-3}$); $\delta_l\omega L_l$ is Å$^{-1}\cdot$Å; $a_{\rm vac}(E-E_{\rm vac})$ is eV$^{-1}\cdot$eV; $\mu_lL_l$ is Å$^{-1}\cdot$Å |
| $K=1$ | pass: reduces to `xray-in-medium-propagation-phase` and `coherent-formation-absorption` |
| Identical layers | pass: $\delta(L_1+L_2)=\delta L$ |
| Sign | pass: the refractive term has the sign of the vacuum retardation, because both come from one optical path; $\tau$ and the phase run over the same $L_l$ |
| Notation | pass: ledger row, `composition_refractive_decrement` docstring, `_layer_delta_omega` docstring and the grouped-kernel docstrings state the same $-\omega\sum_l\delta_lL_l$, $\tau=\sum_l\mu_lL_l$ |

## Implementation comparison (read after the derivation)

Commit `76feb0a90`.

**Per-layer decrement.** `materials/crystal.py::composition_refractive_decrement` computes

```text
S = sum_i n_i * ((Z + f') + 1j*f'')        # use_henke=True
lam = HC_EV_ANG / E
chi = -R_E_ANG * lam**2 / pi * S
delta = 1 - sqrt(1 + chi).real
```

This is the derived $\chi_{0,l}$ and $\delta_l$ term for term, with $\lambda$ in Å from `HC_EV_ANG` $=12398.4198$ eV Å and $r_e=2.8179403\times10^{-5}$ Å. It uses the same Henke convention ($f_1=Z+f'$) and the same exact square root as `chi_0`/`refractive_index`, with the per-cell sum $\sum_{\rm cell}f/V_{\rm cell}$ replaced by $\sum_in_if_i$. With `use_henke=False` it keeps only $Z$, matching `chi_0(use_henke=False)`.

**Tabulation.** `cxr/_setup.py::_layer_delta_omega` stacks $\delta_l(E)$ over the layers (`for *_, comp in layers`, the `(z_top, z_bot, comp)` format of `_stack_tau`). It forms them in float64, casts to `REAL`, and multiplies by `omega_grid` $=E/\hbar c$ (Å$^{-1}$), giving a $(K,n_E)$ grid. It uses the same `use_henke` as the crystal's `delta_omega_grid`.

**Per-layer paths.** `expand_escape_pieces` now returns `segment_escape_paths`' $(n,K)$ piece-end paths when `layers` is given (scalar $(n,)$ otherwise). `segment_escape._escape_paths_at` builds them from `materials.attenuation::_layer_dz` (slab) or `_layer_path_length` (finite footprint), each times $1/\lvert n_z\rvert$ or capped at the prism exit distance. These are the same two helpers `_stack_tau` sums for $\tau$, so the phase and the attenuation run over identical $L_l$. `_layer_dz` for $n_z<0$ is $\max(\min(z,z_{\rm bot})-z_{\rm top},0)$, which is the derived $L_l(z)\lvert n_z\rvert$. For $n_z>0$ it is the mirror (back-face) form. `_segment_escape_paths` cuts every row at each internal interface $z$ (`boundaries = layers[1:]`). For a finite footprint it also cuts where an exit-ray point crosses a layer boundary and at face switches. Ends are extrapolated from the 1/4 and 3/4 samples, so each $L_l$ is affine on every piece, as the derivation requires.

**Formation lines.** `cxr/_formation.py::_formation_cxr`, layered branch:

```text
mu = stack(mu_layers, axis=1)                  # (n, K), mu_l at E_r (_per_hkl)
tau_start = (mu * L_start).sum(axis=1); tau_end = (mu * L_end).sum(axis=1)
half_dL = 0.5 * (L_end - L_start)              # (n, K)
L_esc_mid = 0.5 * (L_start + L_end)            # (n, K)
```

`_per_hkl` passes `mu_layers = [_mu_total_inv_ang(comp, E_r) for _, _, comp in layers]`, the same per-layer $\mu$ as the incoherent route and `_stack_tau`. This gives $\tau=\sum_l\mu_lL_l$, $q=\sum_l\mu_l\Delta L_l/4$, the per-layer $\Delta L_l/2$, and the exact midpoint $L_{l,c}$.

**Host reduction.** `cxr/_cpu_reduce.py::_grouped_formation_reduce` ($K>1$):

```text
h_j = sum_k half_dL[j,k] * delta_omega_grid[k,e]
l_j = sum_k L_esc[j,k]   * delta_omega_grid[k,e]
v   = a_vac[j]*(E - E_vac[j]) - h_j * 1.0
arg = d[j]*om - g_phase[j] - l_j * 1.0
```

This is $v=a_{\rm vac}(E-E_{\rm vac})-\sum_l\delta_l\omega\,\Delta L_l/2$ and $\phi=d\omega-\mathbf g\cdot\mathbf r-\omega\sum_l\delta_lL_l$, the derived expressions with the derived signs. For $K=1$ it takes `half_dL[j,0] * delta_omega_grid[0,e]`, the identical floating-point products of the slab code, so slab output is bit-identical. `accumulate_grouped_formation` reshapes the inputs to $(n,K)$ and $(K,n_E)$ and refuses mismatched $K$. `cxr/_grouped.py::_referenced_phases` subtracts each flight's first-row $L_l$ per layer. That term is $-\omega\sum_l\delta_lL_{l,\rm first}$, common to the flight, so it cancels in $\lvert\cdot\rvert^2$ exactly.

**CUDA (algebraic review only; no device run here).** `_device_grouped` transposes `L_rel` and `half_dL` to $(K,n)$ before `ascontiguousarray`, so element $(l,j)$ sits at `l*n_seg + j`. The $(K,n_E)$ C-order grid has element $(l,e)$ at `l*n_E + e`. `coherent_grouped_jit_kernel.py::_escape_weight` reads exactly `values[layer*n_segments + segment] * delta_omega[layer*n_energies + energy]`, and `n_segments = n_seg = E_vac.size` (`n_g = 1`, so `line_offset = 0`). For $K>1$, `_medium_shift` returns $1$, so the kernel's `escape_distance * medium_shift` and `half_path_change * medium_shift` equal the host's `l_j`, `h_j`. For $K=1$ (including a one-layer `layers` list, whose $(n,1)$ arrays flatten to $(n,)$), both helpers return the old `values[segment]` and `delta_omega[e]`. The unused second-energy read is clamped into the grid. Layer-major indexing is correct.

## Independent numerical check

Script: a scratch file outside the repository (not committed); the recipe below reproduces it. Two rows of one flight, identical except depth ($z_1=900$ Å, $z_2=2600$ Å), $L=10^{-3}$ Å so $F$ is real and constant, in a stack with interface $d=1500$ Å (hopg 002, 25 keV, $119^\circ$, `use_henke=False`, $n_z=-0.4848$). The test runs end to end through `mc_cxr_spectrum`. The interference term $(S_{\rm pair}-S_1-S_2)/(2\sqrt{S_1S_2})=\cos\theta(E)$ was compared with $\cos(\omega\,\Delta d+C+\Delta\Phi_{\rm esc})$. Here $\Delta d=-\hat{\mathbf n}\cdot(\mathbf r_1-\mathbf r_2)$, $\Delta\Phi_{\rm esc}$ is the closed form above, and $\delta$ is computed independently ($f_1=Z$, exact root). The single constant $C$ ($-\mathbf g\cdot\Delta\mathbf r$ plus coefficient phases) was fitted once on an all-carbon stack and held fixed:

| Stack (film / substrate) | max $\lvert\cos_{\rm pred}-\cos_{\rm code}\rvert$ | if film used the crystal $\delta$ | if layered term had flipped sign |
|---|---|---|---|
| C / C (fit) | $3.8\times10^{-10}$ | — | — |
| Si / C | $3.8\times10^{-10}$ | $8.5\times10^{-3}$ | $1.7\times10^{-2}$ |
| C / Si | $3.8\times10^{-10}$ | $1.6\times10^{-2}$ | $3.1\times10^{-2}$ |
| Si / Si | $3.8\times10^{-10}$ | $2.4\times10^{-2}$ | $4.8\times10^{-2}$ |

The code reproduces the per-layer closed form, including which layer's $\delta$ multiplies which path segment, with seven orders of margin over the nearest wrong alternatives.

Project anchors (`pyrite-dev test`, host, 2026-11-14): `tests/materials/test_crystallography.py -k composition_decrement` 6 passed; `tests/montecarlo/spectrum/test_grouped_formation_reduce.py` 18 passed; `tests/montecarlo/spectrum/test_substep_invariance.py -k stack` 2 passed.

## Adjudication

**Matches.** No factor, sign, unit, index or convention divergence was found between the independent derivation and the host or CUDA code. $K=1$ keeps the slab arithmetic, and the per-layer paths are those of `_stack_tau`.

Findings (none change the physics):

1. **Stated approximation:** the stationary root that freezes the amplitudes ($\chi_g$, $\mu_l$ at $E_r$) uses the crystal's $\delta$ and $\nabla L_{\rm esc}$ (`segment_escape_gradient`, layer-blind). This is not the emitter layer's own $\delta$. For a planar stack, $\nabla L_{\rm total}$ equals the emitter layer's gradient, so the only difference is $\delta_{\rm crystal}$ vs $\delta_{\rm emit}$: an $O(\delta)$ shift of the amplitude-evaluation energy. The formation factor's own centre ($v=0$) does use the correct $\delta_{\rm emit}$. The ledger states this assumption.
2. $\delta_l$ comes from Henke $f_1$ while $\mu_l$ comes from EPDL (`_mu_total_inv_ang`). This split is inherited from the single-slab claim, not new.
3. Traceability: the ledger `Code` field omits `cxr/_cpu_reduce.py::_grouped_formation_reduce`, `cxr/_grouped.py::{accumulate_flight_grouped,_device_grouped}` and `coherent_grouped_jit_kernel.py::{_escape_weight,_medium_shift}`. The two device helpers carry the marker in comments, not docstrings. The `cxr/_grouped.py` module docstring still states the single-$\delta$ phase and only `coherent-formation-absorption`.
4. CUDA: this verification covers the code algebraically only. Device parity rests on the CUDA-gated `test_grouped_reduction_cuda.py::test_device_grouped_matches_host_kernel`, which was not run here.
5. The ledger row does not yet link this write-up (left to the task owner). The `radiation-physics/index.md` toctree entry is present in the working tree, where it was added outside this verification.

## Assumptions and scope

Kinematic emission; straight, refraction-free escape along $\hat{\mathbf n}$; sharp planar interfaces with no Fresnel reflection or transmission loss; amplitudes frozen at the crystal-$\delta$ root. Flight-grouped incoherent route only; `coherent=True` through a stack remains refused (`_refuse_incompatible_request`).
