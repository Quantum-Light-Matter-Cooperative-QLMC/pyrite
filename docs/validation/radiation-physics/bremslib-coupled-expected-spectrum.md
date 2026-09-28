# `bremslib-coupled-expected-spectrum`

## Scope and source

This fresh-context verification started from the ledger row, the "Production continuum estimator" section of [hard bremsstrahlung events](../../physics/radiation-physics/hard-bremsstrahlung-events.md), and the compensator (Campbell) identity for a marked point process with predictable intensity. The derivation below was written before reading the transport and scorer bodies. The claim is that on coupled tracks (`radiative_model="bremslib-soft-hard"`), BremsLib track-length scoring over the whole photon range, `brem_events.py::mc_coupled_brem_spectrum`, estimates the same continuum as the analog soft-plus-event estimator without bias. The claim allows for step quadrature. The analog estimator is `mc_soft_brem_spectrum` below $k_c$ plus `mc_hard_brem_event_spectrum` above it. The production entry point is `runner/emission.py::_coupled_brem_from_segments`. The output unit is photons/(eV sr incident electron).

## Independent derivation

### Analog estimator as a marked point process

Parameterize one electron history by path length $s$. Call its state just before $s$ $X(s^-)$: position $\mathbf r$, direction $\hat{\mathbf v}$, kinetic energy $T$ and layer $l$. Hard photons form a point process in $s$ with marks $(k,Z)$, $k\ge k_c$. Suppose transport samples them with the marked intensity

$$
\lambda(s;k,Z)\,ds\,dk
=n_{Z,l}\,\frac{d\sigma_Z}{dk}\bigl(T(s^-),k\bigr)\,\mathbf 1[k_c\le k\le T(s^-)]\,ds\,dk .
$$

Here $n_{Z,l}$ is the number density of element $Z$ in the current layer. The intensity is predictable if it depends only on $X(s^-)$. Each event contributes to photon bin $j$, of width $\Delta k_j$, the weight

$$
g(s;k,Z)=\frac{\mathbf 1[k\in\text{bin }j]}{N_e\,\Delta k_j}\,
\frac{d^2\sigma_Z/(dk\,d\Omega)\,(T(s^-),k,\hat{\mathbf v}\cdot\hat{\mathbf n})}{d\sigma_Z/dk\,(T(s^-),k)}\,
\exp\Bigl[-\sum_{l'}\mu_{l'}(k)\,L_{l'}(\mathbf r(s))\Bigr].
$$

This weight is also predictable. The next state jumps by $T\to T-k$; the event does not change the weight it has just scored.

For a predictable nonnegative $g$ and a point process with predictable intensity, $M_t=\sum_{s_i\le t}g_i-\int_0^t\!\!\int\lambda g$ is a martingale. By optional stopping at the electron's death, the compensator identity follows:

$$
E\Bigl[\sum_i g(s_i;k_i,Z_i)\Bigr]
=E\Bigl[\int_0^{s_{\rm end}}\!ds\sum_Z\int_{k_c}^{T(s^-)}\!dk\;\lambda\,g\Bigr].
$$

On the right, the SDCS in $\lambda$ cancels the denominator of $g$, which leaves

$$
\lambda g=\frac{\mathbf 1[k\in j]}{N_e\,\Delta k_j}\;n_{Z,l}\,
\frac{d^2\sigma_Z}{dk\,d\Omega}\,e^{-\tau(k,\mathbf r)} .
$$

This is the integrand of the BremsLib direction-resolved track-length estimator for bin $j$, evaluated on the same coupled tracks. The two expectations are taken over the same coupled track law, so the event debits $T\to T-k$ that shape the tracks enter both sides alike. Below $k_c$ no photon is sampled, so the track-length term already is the analog soft estimator. Adding the soft and hard parts shows that full-range track-length scoring has the analog mean, with each photon counted exactly once.

The cancellation needs three things:

1. The SDCS in the transport hazard and mark law equals the SDCS in the event weight, with the same tables, $Z^2$ scaling and interpolation.
2. The angular integral $\int d\Omega\, d^2\sigma/(dk\,d\Omega)$ equals that same SDCS, so that $g$ is a normalized conditional density.
3. The number densities are the same element number densities in both, layer by layer.

### Step quadrature

A condensed-history transport does not follow $T(s^-)$ exactly. Suppose a row from $s_0$ to $s_0+\ell$ uses a constant hazard $\mu_h(T_a)=\sum_Z n_Z\sigma_{h,Z}(T_a)$ at some energy $T_a$. Suppose further that it draws the mark $(k,Z)$ from $n_Z\,d\sigma_Z/dk\,(T_b)/\mu_h(T_b)$ at a possibly different energy $T_b$. Then the realized intensity is

$$
\lambda_{\rm eff}(s;k,Z)=\mu_h(T_a)\,\frac{n_Z\,d\sigma_Z/dk\,(T_b)}{\mu_h(T_b)} .
$$

The track-length scorer instead evaluates $n_Z\,d\sigma_Z/dk$ at a representative energy $T_r$ over the whole row. Expanding about the row midpoint energy, the relative per-row difference in the hard yield is

$$
\frac{\lambda_{\rm eff}-\lambda_{\rm TL}}{\lambda_{\rm TL}}
\simeq\frac{d\ln\mu_h}{dT}\,(T_a-T_r)
+\frac{\partial\ln(d\sigma/dk\,/\mu_h)}{\partial T}\,(T_b-T_r).
$$

With $T_a$ the row-start energy and $T_r$ the midpoint, the first term is $-\tfrac12\,(d\ln\mu_h/d\ln T)\,\Delta T_{\rm row}/T$, which is first order in the row's continuous loss. The second term averages to $O(\ell^2)$ when $T_b=T(s^-)$ is the continuous-loss energy at the event point and $T_r$ is the row midpoint. Both terms vanish as the steps shrink. This is the "step quadrature" allowance, and it does not introduce a bias that survives refinement.

### Variance

The write-up says the estimator "replaces each track's photon count by its conditional expectation given the track". That is not correct as a Rao–Blackwell statement. The coupled track records every energy debit, so the photon count is a function of the track, and conditioning on the track returns the analog sum unchanged. The correct decomposition uses $H=\sum_i g_i$, $A=\int\!\!\int\lambda g$ and $M=H-A$:

$$
\operatorname{Var}H=\operatorname{Var}A+E\!\int\!\!\int\lambda g^2+2\operatorname{Cov}(A,M),
\qquad E[M^2]=E\!\int\!\!\int\lambda g^2 .
$$

A photon debit changes the future intensity, so $\operatorname{Cov}(A,M)$ need not vanish. For example, a large hard photon shortens the remaining range and lowers the future $A$ just as $M$ jumps up. Cauchy–Schwarz gives the sufficient condition

$$
\operatorname{Var}H\ge\operatorname{Var}A
\quad\Longleftarrow\quad
\operatorname{Var}A\le\tfrac14\,E\!\int\!\!\int\lambda g^2 .
$$

Per electron and per bin, with a bounded weight, this condition amounts to $\operatorname{Var}(\nu_e)\le\tfrac14E[\nu_e]$. Here $\nu_e$ is the expected number of hard photons the electron emits into the bin. The condition holds whenever an electron emits far less than one hard photon per bin, which is every keV–MeV thin-target regime of interest. The "no higher variance" conclusion therefore holds in the production regime but needs this condition. It is not a general Rao–Blackwell identity.

### Limiting cases

- Below $k_c$ the estimator is the soft track-length scorer by construction. A cutoff inside bin $j$ changes only how the analog pair splits that bin; the expected-value estimator has no split.
- At zero BremsLib cross section the coupled tracks reduce to the uncoupled tracks, and the estimator is `mc_brem_spectrum(cross_section_model="bremslib")` on them.
- As the number of sampled events grows, $H/N_e\to A/N_e$ in probability, because both are means of i.i.d. electron histories with a common expectation.

## Source-to-code comparison

### Transport hazard and mark law

The hazard is in `transport/cores.py::make_cpu_transport_core`, shared by the CPU lockstep and per-electron cores. Its twin on CUDA is `transport/_jit_kernel.py`.

- The row's hard rate `mu_rad` comes from `radiative_layer_moments_scalar(..., L, E_j*1e3, rad_cutoff_eV, ...)` at the row-start energy `E_j` in the current layer `L` (`cores.py:357-377`). The CUDA twin is `_jit_kernel.py:380-395`. The optical depth `tau_rad` is a unit-exponential draw, spent as `step_j*mu_rad` and resampled after every non-substep row end, so the hazard is piecewise constant at each row's start energy.
- At a radiative event, the element and photon energy are drawn at `E_end_j`, the continuous-loss energy at the event point before the photon debit. This uses `radiative_layer_moments_scalar` at `E_end_j` for the element split and `sample_hard_photon_energy_scalar(..., E_end_j*1e3, ...)` for $k$, following the comment "The optical depth used the row-start hazard ... The photon and its element are conditional on the electron's actual pre-event energy at this row end". So $T_a=T_{\rm start}$ and $T_b=T(s^-)$.
- A photon that leaves the electron at or below its cutoff turns the row into a terminal `EVENT_CUTOFF` row that keeps its photon. That row has its full length to the event point in `L_ang`.

### Scorer

`brem_events.py::mc_coupled_brem_spectrum` checks the transport cutoff and table identity (`_check_transport_partition`). It rejects `cross_section_model` and any `E_cut_keV` reclip, strips the `radiative` marker, and calls `mc_brem_spectrum(view, cross_section_model="bremslib", bremslib_tables=...)`. That scorer:

- selects rows with `elec_id < Ne` and divides by `Ne` (`brem.py:969`, `:1171`);
- evaluates `4π d²σ/(dk dΩ)` at `E_repr_keV` $=(T_{\rm start}+T_{\rm end})/2$ (`brem.py:980`, `transport/api.py:1253`), with $T_{\rm end}$ the pre-debit energy on event rows;
- takes the angle from the row's `v_hat`, the same direction the event scorer uses;
- averages escape along the segment.

It keeps every row, including terminal `CUTOFF` rows carrying a photon. `_clip_segments_to_cutoff` returns immediately for `E_cut_keV=None`.

### Numerical checks (CPU, released BremsLib tables for C, Si and W)

| Quantity | Result |
| --- | --- |
| Transport SDCS `_scaled_sdcs_at`$\times10^{-27}Z^2/k$ vs scorer `evaluate_bremslib`, 200 random $T\in[2,300]$ keV $\times$ 5 $k$ each | $\max\lvert\Delta\rvert/\sigma\le6.7\times10^{-16}$ |
| $2\pi\int\sin\theta\,d\theta$ of the scorer DDCS vs scorer SDCS | $\le7.8\times10^{-9}$ (quadrature error of the check) |
| Transport $\sigma_h$ (`radiative_moments_scalar`) vs $4\times10^5$-point quadrature of the scorer SDCS on $[k_c,T]$, $T=2$–300 keV | $\le4.6\times10^{-11}$ |

Requirements 1 and 2 hold to rounding. Transport packs `density*1e24` atoms/cm³ times $10^{-8}$ Å/cm from the same per-layer compositions `[layer[2] for layer in layers]` (`api.py:714`). The runner scores layer $l$ with `abs_layers[l][2]`, and a single slab with `case["composition"]`, which are the same objects, so requirement 3 holds.

The step-quadrature systematic of the hazard at the row start was measured on real coupled tracks with production numerics (`max_dE_frac=0`, no straggling, 1 keV electron cutoff, $k_c=1$ keV, seed 7). The quantity is $\sum\ell\,\mu_h(T_{\rm start})/\sum\ell\,\mu_h(T_{\rm repr})-1$. A Simpson-rule version of the analog compensator along each row gave the same value at every photon energy tested, 2 keV to $0.9E_0$, to $10^{-6}$:

| Case | Rows | Hard events | $\ell\mu_h$-weighted row $\Delta T/T$ | Analog expectation / track length $-1$ |
| --- | --- | --- | --- | --- |
| Si, 100 keV, 1 µm, 200 electrons | 2909 | 0 | $1.06\times10^{-3}$ | $-2.7\times10^{-4}$ |
| C (HOPG density), 30 keV, 1000 Å, 300 electrons | 1064 | 0 | $3.3\times10^{-3}$ | $-9.0\times10^{-4}$ |

The sign is negative because $\sigma_h$ falls with $T$ in this range through the $1/\beta^2$ factor, so the row-start hazard slightly undersamples. The effect is proportional to the row loss and vanishes with step size. The track-length side, a midpoint rule, is the more accurate quadrature of the continuous-hazard process. Neither run sampled a hard photon, which is the sparsity the change addresses.

For variance, the Si run gives $\operatorname{Var}(\nu_e)/E[\nu_e]$ between $2\times10^{-10}$ and $4\times10^{-8}$ for 1 keV bins at 2 keV to 90 keV. That is far below the sufficient bound of $1/4$.

### Double counting

`_coupled_brem_from_segments` returns only the per-layer sum of `mc_coupled_brem_spectrum` (`emission.py:96-141`). `mc_hard_brem_event_spectrum` and `mc_soft_brem_spectrum` have no production caller left in `src/`, `checks/` or `apps/`. The live runner (`runner/__init__.py:1177`), the brem repair `_brem_for_case` (`emission.py:266`), and `energy_grid/convergence_case.py:178` all route through `_brem_wide_from_segments`, which dispatches coupled cases only to `_coupled_brem_from_segments`. The `event_segments` argument was removed at all three call sites. `mc_brem_spectrum` still rejects rows that carry the coupled marker. No path adds the sampled photons as well.

### Anchors

`PYRITE_MC_BACKEND=cpu ... pyrite-dev test tests/montecarlo/test_hard_radiative_transport.py tests/montecarlo/test_coupled_radiative_runner.py -q` passed (32 tests, 2026-09-27). The expected-versus-analog anchor allows $4/\sqrt{N_{\rm events}}\approx21\,\%$. It therefore guards against gross factor errors, such as $4\pi$, $Z^2$, $N_e$ or double counting. It cannot resolve the $10^{-3}$-level quadrature term above.

## Findings

1. **Hazard energy (wording, not bias).** The physics page states the intensity as $\lambda=\sum_Z n_Z\,d\sigma_Z/dk\,(T(s^-))$ and says the event scorer "uses the pre-event energy". Transport actually uses $\mu_h(T_{\rm start})\,n_Z\,d\sigma_Z/dk\,(T(s^-))/\mu_h(T(s^-))$, with the rate at row start (`cores.py:357-377` and `:733-778`; `_jit_kernel.py:380-395`) and the mark at the pre-event energy. The difference is first order in row $\Delta T$, measured at $-2.7\times10^{-4}$ and $-9.0\times10^{-4}$ above, and it vanishes with step size. It falls within the claim's step-quadrature allowance, but the page should state it.
2. **Variance justification.** The Rao–Blackwell sentence ("conditional expectation given the track") is not valid for coupled tracks. The no-higher-variance conclusion holds under $\operatorname{Var}A\le\tfrac14E\int\!\!\int\lambda g^2$, which is satisfied here by eight or more orders of magnitude. It is not unconditional.

## Verdict

- **Claim:** `bremslib-coupled-expected-spectrum`: `montecarlo/spectrum/brem_events.py::mc_coupled_brem_spectrum`; `montecarlo/runner/emission.py::_coupled_brem_from_segments`. Source: the compensator identity for a predictable marked point process, applied to the BremsLib DDCS/SDCS.
- **Filters:** units pass (photons/(eV sr electron) on both sides); limits pass; signs and conventions pass.
- **Re-derivation:** matches. Unbiased up to a first-order step-quadrature term from the hazard at the row start, measured at $\lvert\Delta\rvert\le9\times10^{-4}$. The variance claim holds under a stated sufficient condition rather than by Rao–Blackwell.
- **Verdict:** `rederived`.
- **Write-up:** `docs/validation/radiation-physics/bremslib-coupled-expected-spectrum.md`.
- **Suggested ledger change:** status `rederived`. Notes should record the hazard-at-row-start quadrature term and its measured size, and the corrected variance condition. The physics page wording for both should be fixed. Human sign-off remains pending.

## Resolution

2026-09-27: both findings were applied as wording fixes. The physics page now states the realized intensity $\mu_h(T_{\rm start})\,p(k\mid T(s^-))$ with the measured quadrature term, and replaces the Rao–Blackwell sentence with the variance decomposition and its sufficient condition. The ledger row records status `rederived` from this verdict. No code changed. Human sign-off remains pending.
