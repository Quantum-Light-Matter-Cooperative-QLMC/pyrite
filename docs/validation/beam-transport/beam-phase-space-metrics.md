# Sampled beam phase-space metrics

**Validation id:** `beam-phase-space-metrics` **Code:** `campaign/beam_metrics.py::sampled_beam_metrics`; `results/store.py::beam_current_na` **Status:** rederived in fresh context on 2026-07-28

## Covariance definitions

All moments are centered population moments of sampled macro-particles. For one transverse plane `(x,x')`,

```text
ε = sqrt(<x²><x'²> - <xx'>²)
α = -<xx'>/ε
β = <x²>/ε
γ = <x'²>/ε
ε_n = β_rel γ_rel ε.
```

Here `x` is in mm, `x'=v_x/v_z` is in radians, geometric and normalized emittance are in mm rad, `β` is in mm/rad, and `γ` is in rad/mm. For nonzero emittance the definitions give `βγ-α²=1`. Zero emittance is reported as zero; its Twiss parameters are undefined and therefore `NaN`.

For relative energy deviation `δ=(E-<E>)/<E>`,

```text
ε_z = sqrt(<t²><δ²> - <tδ>²)
σ_z = c σ_t.
```

`ε_z` has fs units and `σ_z` is reported in mm. Normalized emittance uses `γ_rel=1+E/(m_e c²)` and `β_rel γ_rel=sqrt(γ_rel²-1)`, with constants from SciPy.

## Charge and current

Average current follows direct SI conversion:

```text
I_avg [nA] = Q [pC] × f [Hz] / 1000.
```

Thus `1 pC × 5 kHz = 5 nA`. New result records stamp this derived current outside the legacy-shaped case payload; old checkpoints without the stamp or pulse fields retain `Settings.beam_current_na` as compatibility fallback.

`Q/(sqrt(2π) σ_t)` is exposed only as a **Gaussian-equivalent** peak current. It is not labeled as an actual peak for uniform or arbitrary explicit samples, whose density shape must be known or estimated.

## Independent checks

- Units, covariance signs, Twiss convention, relativistic normalization, and longitudinal determinant match independent derivation.
- Analytic arrays reproduce expected RMS, emittance, Twiss, and current values.
- Zero spread gives zero emittance; charged point bunch gives infinite Gaussian-equivalent peak current; zero charge gives zero current.
- Record-level cache keys include derived current, preventing equal case names at different pulse currents from sharing scaled metrics.

## Result

Independent derivation matches implementation. Verdict: `rederived`.
