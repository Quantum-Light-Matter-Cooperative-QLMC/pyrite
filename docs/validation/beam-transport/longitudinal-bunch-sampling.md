# Longitudinal bunch sampling

**Validation id:** `longitudinal-bunch-sampling`  
**Code:** `montecarlo/transport.py::_sample_bunch_offsets`  
**Status:** rederived in fresh context on 2026-07-28

## Claim and convention

Each incident electron receives arrival offset `Δt_e`, separate from transport
age. `bunch_length_fs = σ_t` uses an RMS, not FWHM, convention:

- Gaussian: `Δt ~ Normal(0, σ_t)`.
- Uniform flat-top: `Δt ~ Uniform(-√3 σ_t, +√3 σ_t)`, since a uniform
  distribution on `[-a,a]` has variance `a²/3`.
- Explicit offsets override both analytic distributions.

Every finite sample is shifted by its sample mean. This makes `t=0` the sampled
bunch centroid without changing variance or pairwise time differences.
Transport uses `c=1` clock lengths, so

```text
Δt_ang = Δt_fs × c
c = 2997.924580 Å/fs.
```

## Independent checks

- Units: fs multiplied by Å/fs gives Å.
- Limits: `None` returns exact zeros without constructing a bunch RNG;
  `σ_t=0` returns exact zeros; both recover the point bunch.
- Uniform normalization: half-width `√3 σ_t` gives RMS `σ_t`.
- RNG isolation: bunch draws use `SeedSequence(seed).spawn(4)[3]`, separate
  from transport, transverse-position, and groove-phase streams. Enabling a
  bunch left transport positions, directions, energies, lengths, ages, and
  electron ids bit-identical in a seeded comparison.
- Segment mapping: per-segment `t0_ang` equals
  `initial_t0_ang[elec_id]`; relative `t_ang` remains unchanged.
- Independent `N=200000`, `σ_t=17 fs` probes recovered Gaussian and uniform
  RMS values within 0.04%, with sample means numerically zero.

Explicit arrays are required to be one-dimensional, finite, and one value per
incident electron. Analytic RMS values must be finite and nonnegative.

## Structured-policy extension (2026-07-29)

Fresh-context validation also covered the resolved `compressed` and
`microtrain` policies added to `_sample_bunch_offsets`.

For a full-depth microtrain, the sampler draws an integer center index

```text
n = round(X),  X ~ Normal(0, sigma_center / T)
sigma_center^2 = sigma_envelope^2 - sigma_micro^2 - sigma_jitter^2
Delta t = n T + epsilon_micro + epsilon_jitter
```

where the two `epsilon` terms are independent zero-mean Gaussians with the
named RMS widths. Direct indexed center sampling is distributionally
equivalent to sampling from an explicitly materialized train with Gaussian
center weights, without constructing tens of thousands of center locations.
Subtracting the final sample mean changes only the common temporal phase and
therefore leaves RMS pair separations and bunching-factor magnitude unchanged.

Rounding makes the requested envelope RMS asymptotic rather than algebraically
exact. When `T << sigma_center` and fractional center coordinates are
effectively uniform, quantization contributes approximately `T^2 / 12`, so

```text
Var(Delta t) ~= sigma_envelope^2 + T^2 / 12.
```

The correction is negligible for the approved attosecond-scale periods inside
a 200 fs RMS envelope, but exact-RMS claims must retain this qualification.

At the target angular frequency `Omega`, `Omega T = 2 pi`, hence every integer
center has identical phase. The Gaussian factors give

```text
|F(Omega)|^2
  = exp[-Omega^2 (sigma_micro^2 + sigma_jitter^2)]
  = eta * exp[-Omega^2 sigma_jitter^2]
```

when `sigma_micro = sqrt(-ln eta) / Omega`. Zero jitter therefore recovers the
requested target bunching `eta`. The `compressed` policy draws one Gaussian
with RMS `sigma_micro`, so it has the same zero-jitter target factor `eta` and
is the single-microbunch limit.

For `0 < modulation_depth = D < 1`, implementation uses `D` as the Bernoulli
fraction of electrons drawn from the train and `1-D` from the unmodulated
Gaussian envelope. Its field form factor is therefore

```text
F_mix = D F_train + (1-D) F_envelope,
```

not a linear interpolation of intensities. When the 200 fs envelope is
decoherent at the target, target intensity is approximately
`D^2 eta exp[-Omega^2 sigma_jitter^2]`, not `D eta`. This is a convention
caveat, not a discrepancy for the approved initial `D=1`.

All structured-policy draws use the same dedicated
`SeedSequence(seed).spawn(4)[3]` bunch child stream as legacy analytic
sampling. Additional train, jitter, and mixture draws consume only that child;
transport, transverse-position, and groove-phase streams remain independent.

## Result

Independent derivation and numeric probes match implementation. Verdict:
`rederived`. Current radiation sums electrons incoherently, so arrival offsets
are diagnostic inputs for future coherent form factors and do not change
current spectral yield.
