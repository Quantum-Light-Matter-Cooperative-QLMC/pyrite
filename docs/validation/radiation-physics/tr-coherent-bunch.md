# `tr-coherent-bunch`

## Claim and source

- Claim: coherent TR of a bunch of $N$ independent electrons, per electron, is $E\,n_z^2\,[G+\frac{N-1}{M-1}F(P-G)]/(16\pi^4M)$ with $G=\sum_e\lVert\mathbf S_e\rVert^2$, $P=\lVert\sum_e\mathbf S_e\rVert^2$ over $M$ sampled histories, $\mathbf S_e$ the history's crossing-superposition vector amplitude phased with its own arrival offset and transverse entry; $F=1$ for sampled offsets, an analytic form factor only for offset-free histories.
- Code: `montecarlo/spectrum/transition_radiation_crossings.py::{crossing_transition_radiation, _bunch_power}`; estimator weight `montecarlo/spectrum/coherent_population.py::pair_scale`.
- Sources: distinct-pair estimator of `coherent-physical-bunch-population`, applied to the `tr-crossing-superposition` amplitudes; bunch form factor as in `coherent-inter-electron-decoherence`.
- Units: Gaussian, $\hbar=c=1$, photon energy $E$ in eV, $\omega=E/\hbar c$ in Å$^{-1}$, lengths and times in Å. Output in photons eV$^{-1}$ sr$^{-1}$ per electron.
- Frozen revision: commit `0984a60`.

The derivation below was written from the ledger row, the docstrings and the physics page ("CTR from transported crossings"). The implementation bodies were read afterwards.

## Independent derivation

### Ensemble power per electron

Let electron $e$ of a bunch radiate the vacuum amplitude $\mathbf S_e(\omega,\hat{\mathbf n})$, the coherent sum of its crossings (all tracks of the history). The bunch field is $\sum_{e=1}^N\mathbf S_e$ and the yield is $E\,n_z^2\lVert\sum_e\mathbf S_e\rVert^2/(16\pi^4)$ (normalization of `tr-crossing-superposition`). For $\mathbf S_e$ independent and identically distributed,

$$
\mathrm E\Bigl\lVert\sum_{e=1}^N\mathbf S_e\Bigr\rVert^2
=\sum_e\mathrm E\lVert\mathbf S_e\rVert^2+\sum_{e\neq f}\mathrm E[\mathbf S_e]\cdot\mathrm E[\mathbf S_f]^\ast
=N\,\mathrm E\lVert\mathbf S\rVert^2+N(N-1)\,\lVert\mathrm E\mathbf S\rVert^2 .
$$

Per physical electron: $\mathrm E\lVert\mathbf S\rVert^2+(N-1)\lVert\mathrm E\mathbf S\rVert^2$. This is the claim's moment pair with $F=1$, provided $\mathbf S_e$ is the *complete* field (offsets included). No independence between offsets and transport is needed for this form: only i.i.d. histories.

### Unbiased estimators from $M$ samples

With $M$ i.i.d. samples, $\mathrm E[G/M]=\mathrm E\lVert\mathbf S\rVert^2$. For $e\neq f$, $\mathrm E[\mathbf S_e\cdot\mathbf S_f^\ast]=\lVert\mathrm E\mathbf S\rVert^2$, and $P-G=\sum_{e\neq f}\mathbf S_e\cdot\mathbf S_f^\ast$ has $M(M-1)$ terms, so $\mathrm E[(P-G)/(M(M-1))]=\lVert\mathrm E\mathbf S\rVert^2$. The unbiased estimate per electron is

$$
\boxed{\;\frac1M\Bigl[G+\frac{N-1}{M-1}(P-G)\Bigr]\;}
$$

times $E\,n_z^2/(16\pi^4)$. It needs $M\ge2$ when $N>1$. With $N=M$ it is $P/M$, the plain coherent sum of the sample per electron: one realization of an $M$-electron bunch, not the ensemble. $N=1$ leaves $G/M$, the per-history incoherent yield.

### Analytic form factor

Suppose the transport is independent of the offsets and laterally translation invariant, and the histories were transported without them, giving $\mathbf S_e^0$. The crossing phase $e^{i(\omega t_c-\mathbf k_\perp\cdot\mathbf r_{\perp,c})}$, $\mathbf k_\perp=\omega\hat{\mathbf n}_\perp$, then gives $\mathbf S_e=\mathbf S_e^0\,e^{i\phi_e}$ with $\phi_e=\omega t_{0,e}-\mathbf k_\perp\cdot\Delta\mathbf r_{\perp,e}$. Here $\lVert\mathbf S_e\rVert^2=\lVert\mathbf S_e^0\rVert^2$ and $\mathrm E\mathbf S=\mathrm E[\mathbf S^0]\,\chi$ with $\chi=\mathrm E\,e^{i\phi}$, so the excess carries $F=\lvert\chi\rvert^2$ and the self term none. For independent Gaussian $t_0$ (r.m.s. $\sigma_t$ in Å, i.e. $c\sigma_t$) and $\Delta x$ (r.m.s. $\sigma_x$), with $\hat{\mathbf n}=(\sin\theta,0,\cos\theta)$,

$$
\chi=e^{-\frac12\omega^2\sigma_t^2}\,e^{-\frac12(\omega\sin\theta\,\sigma_x)^2},
\qquad
F=\exp\bigl[-\omega^2\sigma_t^2-(\omega\sin\theta\,\sigma_x)^2\bigr],
\qquad \omega=E/\hbar c .
$$

A round spot of per-axis r.m.s. $\sigma_\perp$ gives $e^{-(\omega\sin\theta\,\sigma_\perp)^2}$ for any azimuth. If the offsets are already in $\mathbf S_e$, the sampled average already supplies $\lvert\chi\rvert^2$ through $P-G$. A further analytic $F$ then multiplies the excess by $F^2$: the documented double-counting caveat is correct. The analytic factor composes per offset: one may sample the transverse offsets and pass only the longitudinal factor, or the reverse.

### Sampling error of the sampled-offset test

For the Gaussian-phase case $z_e=e^{i\phi_e}$ the pair estimate $\hat F=(\lvert\sum z\rvert^2-M)/(M(M-1))$ is a U-statistic with kernel $\mathrm{Re}(z_ez_f^\ast)$. Its leading variance is $4\zeta_1/M$ with $\zeta_1=\mathrm{Var}\,\mathrm{Re}(z\chi^\ast)=\chi^2\mathrm{Var}\cos\phi=F(1-F)^2/2$, so

$$
\frac{\sigma_{\hat F}}{F}=(1-F)\sqrt{\frac{2}{MF}} .
$$

## Cheap filters

| Check | Result |
|---|---|
| Units | $G,P$ carry $\lvert\mathbf a\rvert^2$ of `tr-crossing-superposition`; $\frac{N-1}{M-1}$ and $F$ are dimensionless; $\omega\sigma_t$, $\omega\sigma_x$ dimensionless with $\omega=E/\hbar c$ [Å$^{-1}$]. Pass. |
| Limits | $N=1\Rightarrow G/M$; $N=M\Rightarrow P/M$; identical in-phase histories $\Rightarrow(1+(N-1)F)\lVert\mathbf S\rVert^2$, i.e. $N^2$ per point bunch; $F\to0$ (long bunch) $\Rightarrow$ incoherent sum. Pass. |
| Conventions | The crossing phase sign is irrelevant to $F=\lvert\chi\rvert^2$ for symmetric offset laws. Pass. |

## Comparison with the implementation

- `crossing_transition_radiation` (lines 359–395) groups crossings by `electron_id` (stable sort), sums each history's crossing amplitudes by `cumsum` differencing within whole-history chunks, accumulates $G$ in `total` and $\sum_e\mathbf S_e$ in `field` across chunks (line 389), and divides by $16\pi^4M$ with $M=$ `n_electrons`. This matches.
- `_bunch_power` (lines 398–428) forms $G+\text{scale}\cdot F\,(P-G)$ with `scale = pair_scale(N, M)` $=(N-1)/(M-1)$, $N=M$ when `physical_electrons is None`. This matches. `pair_scale` returns 0 for $N\le1$, so $M=1$ with $N=$ `None` is safe, and it raises `CoherentSamplingError` for $N>1$, $M<2$.
- The analytic `form_factor` multiplies the excess only (line 418), as derived. The docstring's example $\exp(-(E\sigma_t/\hbar c)^2)$ takes $\sigma_t$ in Å ($c=1$), as stated in the module's units.
- `boundary_crossings` sets `time_ang = t0 + age` (line 245) and the crossing position carries the sampled entry point, so `coherence="bunch"` pairs complete fields with $F=1$, as derived.

### Independent numerical checks

The scratch script `ctr_check.py` (session scratchpad, not committed) used:

1. **Brute-force pair sum.** 37 synthetic histories through a three-layer absorbing stack with random $t_0$, entry points, speeds and tilts. About half carry a second, opposite-charge track. The records were scrambled out of history order. Per-history fields came from `crossing_amplitudes` summed by an explicit loop. The per-electron estimate was formed as $G/M+(N-1)\sum_{e\neq f}\mathrm{Re}(\mathbf S_e\cdot\mathbf S_f^\ast)/(M(M-1))$ with an explicit double loop. It agrees with `crossing_transition_radiation(..., coherence="bunch")` to $\le2\times10^{-15}$ (relative to the peak) for $N\in\{$`None`$,1,2,10^5\}$ and `max_elements` $\in\{1,7,50,2^{22}\}$, forward directions, with energy and direction broadcast $(3,1)\times(2,)$. With an analytic $F$ the agreement is $2\times10^{-15}$.
2. **Translation phase.** Shifting a history by $\delta t=1.7$ Å and $\Delta\mathbf r=(13,-8,0)$ Å multiplies its summed amplitude by $e^{i(\omega\delta t-\omega\hat{\mathbf n}_\parallel\cdot\Delta\mathbf r)}$ to $9\times10^{-13}$, both forward and backward ($n_z<0$, entrance-face rotation). This confirms the phase that the sampled form factor relies on.
3. **Sampling distribution of the sampled-offset anchor.** The anchor setup ($M=4000$, $\sigma_t=2$ Å, $\sigma_x=30$ Å, $N=10^6$) was repeated over 200 seeds:

   | $E$ [eV] | $F$ | empirical s.d. of ratio $-1$ | U-statistic $(1-F)\sqrt{2/(MF)}$ | test's $\sqrt{2(1-F)/(MF)}$ | seed-7 deviation |
   |---|---|---|---|---|---|
   | 300 | 0.798 | 0.0047 | 0.0050 | 0.0112 | 0.0065 |
   | 500 | 0.535 | 0.0134 | 0.0142 | 0.0209 | 0.0220 |
   | 800 | 0.201 | 0.0377 | 0.0398 | 0.0445 | 0.0832 |

   The mean deviation over 200 seeds is within $1.2$ s.e. of zero at every energy, so the estimator is unbiased. The test's $\sigma$ is larger than the true one by $1/\sqrt{1-F}$ (2.2, 1.5 and 1.1 times). Its "three standard deviations" band is therefore 6.7, 4.4 and 3.4 true standard deviations. The seeded realization sits at 1.3, 1.5 and 2.1 true standard deviations.

Existing suites: `tests/montecarlo/test_transition_radiation_ctr.py` and `tests/montecarlo/test_transition_radiation_crossings.py` pass (19 tests). The bunch chunk-invariance check on transported bilayer histories is `test_transition_radiation_crossings.py::test_transported_yield_is_substep_and_chunk_invariant` (rtol $10^{-12}$).

## Adjudication

No divergent factor, sign, exponent or unit. The estimator, its $N=$ `None` semantics, the cross-chunk field accumulation, the excess-only analytic factor, the Gaussian form factor and the $\hbar c$ conversion all match the independent derivation. Verdict: **rederived**.

### Findings (owner to apply; none changes physics)

1. **Low, test honesty.** `test_sampled_offsets_give_the_gaussian_form_factors` (lines 99–101) uses $\sqrt{2(1-F)/(MF)}$, which overstates the pair-estimate standard deviation by $1/\sqrt{1-F}$. The ledger anchor ("within three standard deviations") and the physics page then overstate the gate's strictness at the high-$F$ point (effectively 6.7$\sigma$). Suggested fix: use $(1-F)\sqrt{2/(MF)}$. The seeded realization still passes at $3\sigma$ (largest 2.1$\sigma$).
2. **Low, wording.** The page's assumption "transport does not depend on the offsets" is needed only for the analytic `form_factor`. The default $F=1$ estimator is unbiased for any offset–transport correlation (tilted faces, bunch-front gradients), as `coherent_population` notes for the line path. Say so, and note that the analytic factor composes per offset (sample one, pass the other).
3. **Low, comment.** `test_a_long_bunch_returns_the_incoherent_sum` says the speckle decorrelates over "a few eV". With $\hbar c/\sigma_t=1973/5000\approx0.4$ eV it decorrelates within one 1 eV step. The 4$\sigma$ tolerance is honest for independent bins.
4. **Traceability.** The ledger `Anchor` names only the CTR file. The chunk-invariance limit the `Checks` list requires lives in `test_transition_radiation_crossings.py::test_transported_yield_is_substep_and_chunk_invariant`. Add it.

### Caveats

- The model inherits `tr-crossing-superposition`'s laterally infinite, ungrooved stack. Finite footprints are not rejected by `boundary_crossings` (side exits are only counted). Near an edge, the amplitude model, not the estimator, is the limitation.
- The negative-bin warning is the only sampling diagnostic. There is no variance estimate. With $(N-1)/(M-1)\gg1$ the relative noise of the excess is $\sim(1-F)\sqrt{2/(MF)}$, multiplied by $(N-1)F$ against the self term.
- Not exercised: the CuPy path; `physical_electrons=0`, which returns the self term $G/M$ (inherited from `pair_scale`); complex `form_factor` input, which is cast to float.
