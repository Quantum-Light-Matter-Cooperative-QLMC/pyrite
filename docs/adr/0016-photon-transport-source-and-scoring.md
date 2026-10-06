# 0016 — Photon transport source and scoring

- **Status:** Accepted
- **Date:** 2026-10-05

## Context

Straight-ray photon escape removes scattered and absorbed photons without
following redirection, fluorescence, or charged descendants. Issue #341
introduces optional photon Monte Carlo transport. PyRITE already estimates
continuous radiation from transported electron segments; replacing that
estimator with a histogram of rare hard emissions would change its coverage
and statistical efficiency.

## Decision

Use detailed photon interaction histories and a shared photon/charged
descendant queue. Implement ordinary detector-hit scoring as the reference,
following the architecture of PENELOPE's photon tracker and secondary stack
{cite:p}`salvat2019penelope`.

Accept both physical emission events and a weighted continuous-source adapter.
The adapter samples unattenuated emission along electron segments, retaining
the joint position, energy, and direction distribution. Sample the full
emission solid angle so initially off-cone photons can scatter into the
detector. Weighted scoring histories do not debit the original electron's
energy or add a second physical cascade to its energy ledger. Their own
charged descendants must be followed when required by the response model.

Keep PXR/CBS on its existing formation and escape path initially. Coherent
segment amplitudes are not independent photon sources. Photon mode remains
opt-in and adds identity/provenance keys only when enabled.

Add a conditional detector estimator only after deriving it and verifying it
against ordinary detector-hit scoring. Do not sum both estimates of the same
contribution. Choose source sampling and variance reduction using measured
uncertainty per runtime, rather than assuming either estimator is faster.

## Consequences

The reference tracker separates interaction correctness from scoring
optimization. Continuous-source sampling can reuse the electron transport
stage for mean spectra; it does not preserve every physical shower's
correlations. Event response requires retained shower histories. Thick MeV
targets require photon-born charged descendants and their radiation, not
only a photon attenuation/scattering pass.

New differential photon data and sampled relaxation are required before
the complete mode can be exposed. This ADR accepts the architecture, not
implementation support or physics validation. Interface contracts and data
gaps are in the [photon transport design](../research/physics/photon-transport.md).
