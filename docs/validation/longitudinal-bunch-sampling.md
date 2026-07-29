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

## Result

Independent derivation and numeric probes match implementation. Verdict:
`rederived`. Current radiation sums electrons incoherently, so arrival offsets
are diagnostic inputs for future coherent form factors and do not change
current spectral yield.
