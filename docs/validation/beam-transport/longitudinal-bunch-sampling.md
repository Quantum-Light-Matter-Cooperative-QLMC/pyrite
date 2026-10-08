# Longitudinal bunch sampling

**Validation id:** `longitudinal-bunch-sampling` **Code:** `montecarlo/transport.py::_sample_bunch_offsets` **Status:** rederived in fresh context on 2026-07-28

## Claim and convention

Each incident electron receives arrival offset `Δt_e`, separate from transport age. `bunch_length_fs = σ_t` uses an RMS, not FWHM, convention:

- Gaussian: `Δt ~ Normal(0, σ_t)`.
- Uniform flat-top: `Δt ~ Uniform(-√3 σ_t, +√3 σ_t)`, since a uniform distribution on `[-a,a]` has variance `a²/3`.
- Explicit offsets override both analytic distributions.

Every finite sample is shifted by its sample mean. This makes `t=0` the sampled bunch centroid without changing variance or pairwise time differences. Transport uses `c=1` clock lengths, so

```text
Δt_ang = Δt_fs × c
c = 2997.924580 Å/fs.
```

## Independent checks

- Units: fs multiplied by Å/fs gives Å.
- Limits: `None` returns exact zeros without constructing a bunch RNG; `σ_t=0` returns exact zeros; both recover the point bunch.
- Uniform normalization: half-width `√3 σ_t` gives RMS `σ_t`.
- RNG isolation: bunch draws use `SeedSequence(seed).spawn(4)[3]`, separate from transport, transverse-position, and groove-phase streams. Enabling a bunch left transport positions, directions, energies, lengths, ages, and electron ids bit-identical in a seeded comparison.
- Segment mapping: per-segment `t0_ang` equals `initial_t0_ang[elec_id]`; relative `t_ang` remains unchanged.
- Independent `N=200000`, `σ_t=17 fs` probes recovered Gaussian and uniform RMS values within 0.04%, with sample means numerically zero.

Explicit arrays are required to be one-dimensional, finite, and one value per incident electron. Analytic RMS values must be finite and nonnegative.

## Structured-policy extension (2026-07-29)

Fresh-context validation also covered the resolved `compressed` and `microtrain` policies added to `_sample_bunch_offsets`.

For a full-depth microtrain, the sampler draws an integer center index

```text
n = round(X),  X ~ Normal(0, sigma_center / T)
sigma_center^2 = sigma_envelope^2 - sigma_micro^2 - sigma_jitter^2
Delta t = n T + epsilon_micro + epsilon_jitter
```

where the two `epsilon` terms are independent zero-mean Gaussians with the named RMS widths. Direct indexed center sampling is distributionally equivalent to sampling from an explicitly materialized train with Gaussian center weights, without constructing tens of thousands of center locations. Subtracting the final sample mean changes only the common temporal phase and therefore leaves RMS pair separations and bunching-factor magnitude unchanged.

Rounding makes the requested envelope RMS asymptotic rather than algebraically exact. When `T << sigma_center` and fractional center coordinates are effectively uniform, quantization contributes approximately `T^2 / 12`, so

```text
Var(Delta t) ~= sigma_envelope^2 + T^2 / 12.
```

The correction is negligible for the approved attosecond-scale periods inside a 200 fs RMS envelope, but exact-RMS claims must retain this qualification.

At the target angular frequency `Omega`, `Omega T = 2 pi`, hence every integer center has identical phase. The Gaussian factors give

```text
|F(Omega)|^2
  = exp[-Omega^2 (sigma_micro^2 + sigma_jitter^2)]
  = eta * exp[-Omega^2 sigma_jitter^2]
```

when `sigma_micro = sqrt(-ln eta) / Omega`. Zero jitter therefore recovers the requested target bunching `eta`. The `compressed` policy draws one Gaussian with RMS `sigma_micro`, so it has the same zero-jitter target factor `eta` and is the single-microbunch limit.

For `0 < modulation_depth = D < 1`, implementation uses `D` as the Bernoulli fraction of electrons drawn from the train and `1-D` from the unmodulated Gaussian envelope. Its field form factor is therefore

```text
F_mix = D F_train + (1-D) F_envelope,
```

not a linear interpolation of intensities. When the 200 fs envelope is decoherent at the target, target intensity is approximately `D^2 eta exp[-Omega^2 sigma_jitter^2]`, not `D eta`. This is a convention caveat, not a discrepancy for the approved initial `D=1`.

All structured-policy draws use the same dedicated `SeedSequence(seed).spawn(4)[3]` bunch child stream as legacy analytic sampling. Additional train, jitter, and mixture draws consume only that child; transport, transverse-position, and groove-phase streams remain independent.

## Result

Independent derivation and numeric probes match implementation. Verdict: `rederived`. Current radiation sums electrons incoherently, so arrival offsets are diagnostic inputs for future coherent form factors and do not change current spectral yield.

## Counter-addressed draws (#361 re-verification, 2026-10-07)

Fresh-context check of commit `df221caf`. The uniform and normal construction and their independence are derived in [beam-phase-space-injection §6](beam-phase-space-injection.md). Here $u_{e,c}$ and $Z_{e,c}=\Phi^{-1}(u_{e,c})$ are draws of electron $e$ in the `spawn(4)[3]` child namespace.

### Draw roles and sampling laws

Using the documented role table ($c=0$ Gaussian or centre, $1$ microbunch, $2$ jitter, $3$ envelope, $4$ mixture uniform), the raw offsets before centring are the following.

- Legacy Gaussian: $\Delta t_e=\sigma\,Z_{e,0}$.
- Legacy uniform: $\Delta t_e=\sqrt3\,\sigma\,(2u_{e,0}-1)$, uniform on $(-\sqrt3\sigma,\sqrt3\sigma)$ with RMS $\sigma$.
- `gaussian` and `compressed`: $\Delta t_e=\sigma_{\rm rms}\,Z_{e,0}$.
- `microtrain`:

$$
\Delta t_e^{\rm train}=\operatorname{round}\!\Bigl(\tfrac{\sigma_{\rm c}}{T}Z_{e,0}\Bigr)T+\sigma_{\rm micro}Z_{e,1}+\sigma_{\rm jitter}Z_{e,2},
\qquad
\Delta t_e^{\rm env}=\sigma_{\rm env}Z_{e,3},
$$

  with $\sigma_{\rm c}^2=\sigma_{\rm env}^2-\sigma_{\rm micro}^2-\sigma_{\rm jitter}^2$, and $\Delta t_e=\Delta t_e^{\rm train}$ if $u_{e,4}<D$, else $\Delta t_e^{\rm env}$.

Because $Z_{e,0..3}$ and $u_{e,4}$ are independent with the stated laws, every distribution of the earlier sections is unchanged. That covers the RMS normalizations, the $T^2/12$ rounding term, the target bunching $\eta\,e^{-\Omega^2\sigma_{\rm jitter}^2}$, and the $D^2\eta$ mixture. The final offset is $\Delta t_e-\overline{\Delta t}$. Only that centroid depends on $N$, so the raw draws are prefix-stable, and the block driver redoes the centring once over the joined population.

### Implementation diff

`_sample_bunch_offsets` uses `child_stream_root(seed, 4, 3)` with `counter_normals(root, Ne, 1)[:, 0]` for the Gaussian, compressed and legacy-Gaussian draws, and `half_width * (2 * counter_uniforms(root, Ne, 1)[:, 0] - 1)` for the legacy uniform. The microtrain uses `z = counter_normals(root, Ne, 4)` in roles 0–3 and `counter_uniforms(root, Ne, 5)[:, 4] < depth` for the mixture. Centring is `dt - dt.mean()`. The no-bunch branch returns zeros without touching a stream. `block_transport.py::finalize_population_fields` calls the same sampler over `result["Ne"]` and gathers it onto rows by `electron_id` and `vacuum_elec_id`. This matches the role table and laws term for term.

### Numeric evidence (independent, CPU)

- Role reconstruction from an independent pure-Python SplitMix64 reference ($e<3000$, centred) matches the sampler output within $1.1\times10^{-13}$ fs for microtrain $D=1$, $0$ and $0.37$. It matches within $10^{-9}$ Å for legacy Gaussian and uniform ($e<200$).
- Laws at $N=2\times10^6$, seed $314159$: legacy Gaussian RMS$/\sigma=1.00039$; legacy uniform RMS$/\sigma=1.00030$ with $\max\lvert t\rvert/(\sqrt3\sigma)=1.00025$, where the excess is the centroid shift; compressed RMS$/\sigma=1.00039$; centred means $\sim10^{-16}$.
- Microtrain with $\sigma_{\rm env}=200$, $\sigma_{\rm micro}=0.05$, $T=0.6$ and $\sigma_{\rm jitter}=0.02$ fs, at $\Omega=2\pi/T$:

  | $D$ | RMS$/\sigma_{\rm env}$ | measured $\lvert F\rvert^2$ | predicted $D^2\eta\,e^{-\Omega^2\sigma_{\rm jitter}^2}$ |
  | --- | --- | --- | --- |
  | 1 | 1.00039 | 0.7277 | 0.7276 |
  | 0.37 | 0.99993 | 0.0997 | 0.0996 |
  | 0 | 0.99975 | $\sim0$ | 0 |

- No-bunch and $\sigma=0$ give exact zeros. $\rho(\delta, t)=5.6\times10^{-4}$ at $N=10^6$, consistent with zero, so energy spread and arrival time remain uncorrelated. Bunch counters for $[900,1000)$ equal the slice of $[0,1000)$.
- The ledger anchors pass, including `test_block_transport_equals_one_fixed_n_call` with a microtrain bunch.

### Result (#361)

No discrepancy. Verdict for the stream change: `rederived`; the anchored status is supported.
