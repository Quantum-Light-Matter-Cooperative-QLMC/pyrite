# Phase retention decisions

`mc_spectrum(..., phase_retention=policy)` optionally gates an already certified
reduction on its observable and measured cost. The owner is
`montecarlo/spectrum/phase_retention.py`. The initial supported rule is
[certified flat-term omission](../../validation/beam-transport/coherent-flat-term-omission.md).
The gate changes no physics inequality, field, transport, RNG stream, or bunch
weight. It can only retain more work than the existing certificate requires.

## Observable and accuracy contract

The reference is the full coherent reducer on **identical saved trajectories,
reflection/orientation rows, observation directions and evaluated energy nodes**.
For analytic bunch offsets, that reducer already evaluates the model's
conditional offset ensemble average. It is not a fresh transport ensemble
average. See [coherent emission](../../physics/radiation-physics/coherent-emission.md)
for the separation and its eligibility restrictions.

The implemented certificate controls removal of the inter-electron correction
against each row's grouped per-electron floor at evaluated nodes. It does not
certify relative error against a cancelling full spectrum, continuous energy
integrals, centroid, FWHM, time profiles, or a detector response. Positive
discrete quadrature weights preserve the existing absolute node bound; a
continuous observable still needs its own quadrature certificate.

`coherent_flat_omission_limit` supplies the existing accuracy share. The new
policy supplies no additional allowance and cannot raise that share. Combining
#350 grid/normalization charges with #362 omission charges still requires a
common reference and observable contract; the gate does not establish one.

## Interface

| Input/output | Meaning |
| --- | --- |
| `PhaseRetentionScope.sector` | `inter-electron` for the supported rule; intra-electron requests retain phase |
| `observable`, `reference`, `limit` | Evaluated spectrum; grouped per-electron floor; existing dimensionless share |
| `rule` | `certified-flat-omission`; unknown rules retain phase |
| `route`, `backend`, `precision` | Actual reducer route and array arithmetic; evidence is specific to these |
| `energies`, `retained`, `rows` | Total node count, certified retained count, rows sharing this decision |
| `segments`, `electrons` | Prepared spectrum-state sizes; not a substitute for a workload fingerprint |
| `row_vectors_inv_ang`, `physical_electrons` | Actual reciprocal vectors (1/Å) identifying the reflection/orientation rows, and physical bunch population |
| `PhaseRetentionCost` | Matching scope, full-time lower estimate, reduced-time upper estimate, overhead upper estimate in seconds, evidence identifier |
| `PhaseRetentionDecision` | Scope, accepted/refused status, reason, cost, retained/skipped node counts |

The `certified-flat-omission` mask is the union of two directed certificates:
the longitudinal $F_z$ bound and, for finite-footprint rows with a recorded spot,
the [row transverse bound](../../validation/beam-transport/coherent-transverse-flat-omission.md)
on $F_zF_\perp$. Both use the same allowance; the gate sees only the final mask.

`PhaseRetentionPolicy.estimate(scope)` returns matching cost evidence or `None`.
The provider owns workload applicability: saved-trajectory and axis fingerprints,
physical population, geometry, form-factor law, chunking, device, driver, and
other workload metadata must match its measured regime. Equal counts alone
cannot establish that match. No automatic cost calibration or default cost model
is installed. An evidence string is provenance, not independent verification.

The gate requires the full-time lower estimate to exceed the sum of reduced-time
and overhead upper estimates. Overhead includes computing the accuracy mask,
host/device synchronization and transfers, lookup, reporting and this gate.
Negative/nonfinite timings, mismatched scope, missing evidence and no positive
saving retain full phase. Exceptions from user callbacks propagate: failed
instrumentation must not silently produce accepted evidence.

`report(decision)` observes the actual gate decision without changing inputs.
Supported route names are `per-hkl-eager`, `per-hkl-jit`, `batched-eager`,
`batched-jit`, and `batched-stream`. Per-row routes decide separately; streaming
uses the existing union of retained coordinates across rows. Counts describe
node work, not a claim about the number of kernel operations saved.

## Controls and diagnostics

```python
from pyrite.montecarlo.spectrum.phase_retention import PhaseRetentionPolicy

decisions = []
policy = PhaseRetentionPolicy(report=decisions.append)
# No cost provider: full phase is retained even if the accuracy mask permits omission.
spec = mc_spectrum(
    segments,
    energy,
    crystal="hopg",
    hkl_list=[(0, 0, 2)],
    coherent=True,
    coherent_flat_omission_limit=1e-4,
    phase_retention=policy,
)
```

`enabled=False` disables simplification before certificate construction.
An absent cost provider also bypasses omission-mask certification and reports
`missing-cost-evidence`; no candidate mask is needed to retain the full reducer.
`coherent_flat_omission_limit=0` also retains the full reducer. `phase_retention=None`
preserves the existing explicit flat-omission behavior and runner policy;
this opt-in interface does not install an adaptive default in `Case` or campaigns.

The policy refuses temporal output and coefficient capture scopes even if the
requested observable is the evaluated spectrum. Their full-field arrays cannot
represent the skipped correction. The existing yield/centroid audit refusal for
active omission remains conservative; callers must set the omission limit to
zero for that audit, including when a cost gate would decline simplification.

## Incoherent limit

An accepted decision can skip every cross-electron energy node in a long-bunch
regime. It still accumulates the complete complex field of each electron: the
reported sector is `inter-electron`, and zero retained nodes refers only to
the all-electron reduction. It does not mean that all phase work disappeared.
One physical flight per electron can reach the incoherent result when the
escape quadratures agree. Multiple flights need a separate certificate for
their mutual interference before switching to the flight-grouped incoherent
reducer. Numerical substeps within a flight remain coherent in either case.
The [physical-limit contract](../../research/physics/phase-retention.md#target-in-physically-incoherent-limits)
records the intended endpoint and the currently unsupported sectors.

## Evidence and remaining scope

The fixed-trajectory regression module `test_coherent_flat_omission.py` checks
accepted masks against the existing reducer, and checks bit-identical full
output on disable, absent cost, no gain, excessive overhead, stale scope,
nonfinite/negative costs and unsupported observables. Its existing analytic
anchors cover constructive/destructive electron interference and physical
population weighting. The transverse certificate has its own ledger claim
and anchors (`test_coherent_transverse_omission.py`).

[Historical paired GPU evidence](../../validation/beam-transport/coherent-flat-term-omission.md#historical-task-owner-paired-gpu-measurements-sampled-population)
shows a 1.2506 median eager-CuPy speed ratio for one thick HOPG fixture, with
streaming and JIT ratios of 0.9600 and 0.8585. Those measurements precede the
current physical-population model and this gate. They justify backend-specific
calibration, not a built-in profitable-regime rule. Current paired measurements
and scoped calibration are recorded in the [remote benchmark report](phase-retention-benchmarks.md).
Automatic campaign calibration, additional supported mechanisms and complete
observable budgets remain open under
[issue #365](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/365).

See the [mechanism survey](../../research/physics/phase-retention.md) for
unsupported reductions and the information needed to assess them.
