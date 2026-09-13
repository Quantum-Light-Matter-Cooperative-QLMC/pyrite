# Superradiant PXR: requirements beyond coherent tracking

**Status:** Research proposal, 12 September 2026; implementation assessment at
`a017a50f`. No absolute superradiant yield or paper reproduction is validated.

## Scope

The target is the externally microbunched, relativistic PXR source studied by
{cite:t}`feranchuk2022`. A bunch prepared by an XFEL enters a crystal, and its
density modulation resonates with PXR emission. The paper studies extremely
asymmetric diffraction with electrons traveling near the crystal surface and
photons exiting at a large angle to the beam. It does not establish a general
relativistic CBS model or demonstrate self-bunching inside PyRITE's target.

The published 2022 PDF was inspected from
`/home/alexa/dev/PhysRevAccelBeams.25.120702.pdf`; the Zotero parent record is
`IJNJA3GU`. The portable reference is the DOI in the shared bibliography. The
local PDF is not a repository dependency. Source equation references in the
[companion PXR/CBS assessment](relativistic-pxr-cbs.md) refer to the inspected
published text; the arXiv version is not assumed numerically identical.

## Existing foundations

PyRITE already supplies:

- complex per-polarization segment amplitudes and optional per-electron
  trajectory coherence;
- longitudinal Gaussian, compressed and microtrain policies, with modulation
  depth and timing jitter;
- transverse phase-space sampling, normalized emittance and energy spread;
- bunch form-factor treatments, with restrictions for finite geometry;
- arbitrary observation directions and finite-footprint transport.

See [longitudinal structure](../../physics/beam-transport/longitudinal-structure.md),
[beam phase space](../../physics/beam-transport/beam-phase-space.md), and
[coherent tracking](../../physics/radiation-physics/coherent-emission.md).
These facilities avoid rebuilding the beam sampler, but do not establish the
absolute normalization or diffraction response required by the paper.

## Physical charge must be distinct from sample count

At the inspected revision,
[`_finalize_spectrum`](../../../src/pyrite/montecarlo/spectrum/lines/_spectrum.py)
divides accumulated radiation by the simulated electron count. Meanwhile,
[`results.store`](../../../src/pyrite/results/store.py) uses bunch charge and
repetition rate as an average-current normalization. Fully constructive
inter-electron enhancement can therefore grow with the numerical population.
Increasing Monte Carlo resolution must not change a fixed physical bunch's
predicted yield.

For independent, identically distributed electron field amplitudes, the
candidate physical normalization separates self and pair terms:

```{math}
:label: eq-research-spxr-physical-population

\left\langle\left|\sum_{a=1}^{N} \mathcal E_a\right|^2\right\rangle
= N\left\langle|\mathcal E|^2\right\rangle
+N(N-1)\left|\left\langle\mathcal E\right\rangle\right|^2,
\qquad N=Q/|e|.
```

Here $N$ is the physical electron population, $Q$ is the magnitude of bunch
charge, $|e|$ is the elementary charge, and $\mathcal E_a$ is one electron's
complex radiation amplitude in a specified energy, direction and polarization
channel. Its squared modulus has that channel's spectral-yield units; $N$ is
dimensionless. The average is over the specified beam/trajectory ensemble at
fixed physical population. This is the independent-particle expansion behind
the spontaneous and coherent contributions discussed by
{cite:t}`feranchuk2022`, not an assertion that all real XFEL bunches are i.i.d.

Expanding the squared sum produces $N$ diagonal terms and $N(N-1)$ ordered
off-diagonal terms. Independence factors the latter into the squared mean
amplitude, yielding {eq}`eq-research-spxr-physical-population`. A zero mean field
leaves the linear self term; identical deterministic in-phase fields give the
quadratic population limit; a single electron has no pair term.

`Validation: spxr-physical-population` — research-only, unverified; see the
[core physics ledger](../../validation/ledger-core-coherent-physics.md).
The derivation here is implementation-context reasoning, not independent
verification or an implemented estimator.

An estimator must infer these moments from the numerical tracks while using
physical charge for population weights. Squaring a finite-sample mean introduces
a self-term bias into the estimated pair contribution; use an explicitly
derived pair estimator or equivalent correction. Check convergence as sample
count changes at fixed charge. Correlated electrons, collective fields and
shot-to-shot shared jitter require a joint or conditional ensemble treatment
rather than the independence factorization above.

For pulse output, apply repetition rate after computing the physical per-bunch
yield. Multiplying the present coherent spectrum by average current alone
does not supply the missing physical pair count.

## A fixed beam and a wavelength-matched design are different experiments

The existing `microtrain` policy derives spacing from the target line for each
material, beam energy and geometry. This is useful for asking what bunch would
match each candidate source. It is not a scan of one fixed XFEL bunch across
different crystal orientations: automatic retuning would remove the detuning
being investigated.

Add a fixed physical modulation period or accept a fully specified external
longitudinal distribution independently of target-line resolution. Keep the
existing matched policy as a separate design mode. Record modulation period,
microbunch width, envelope duration, modulation depth, jitter model and their
provenance in the physical case identity.

Model the relevant joint distribution if energy chirp, transverse–longitudinal
correlations or common microbunch timing fluctuations are needed. Current
independent arrival-time and energy sampling cannot represent all such beams.
Inspect each coherent backend's form-factor eligibility: a Gaussian analytic
average must not silently erase a microtrain's resonant structure.

## Diffraction, geometry and transport

The paper's coupled forward/diffracted fields and crystal–vacuum boundary
conditions require the dynamical extension described in the
[companion assessment](relativistic-pxr-cbs.md). Existing groove escape geometry
or a short Beer–Lambert exit path is not equivalent to that field solution.

Its transverse coherence and angular-divergence conditions must be evaluated
in the correct crystal and observation frames. A short longitudinal bunch alone
does not guarantee constructive radiation from a finite transverse beam.
Likewise, the present low-energy geometry's transverse suppression cannot be
transferred to an asymmetric relativistic reflection without recomputing its
phase matching.

Use the published 6.7 GeV, Si (400), 6.45 keV photon example as a candidate
reproduction target, including its full geometry, charge, susceptibilities and
beam assumptions {cite:p}`feranchuk2022`. Do not substitute the paper's ideal-beam
simplifications for the actual sampled emittance without checking their bounds.

Start with prescribed tracks to isolate diffraction and normalization. Later,
couple to transport with high-energy scattering and energy loss. Validate
near-surface path lengths, finite lateral dimensions, escape boundaries and
phase accumulation. GeV calculations need stable relativistic denominators and
adequate precision, beyond the default device float32 arithmetic.

## Suggested research stages and acceptance

| Stage | Scope | Required evidence |
| --- | --- | --- |
| Physical normalization | Separate numerical samples, physical charge and pulse rate | Fixed-charge sample-count convergence; self and pair limits; unbiased pair estimate |
| Beam resonance | Fixed-period microbunches and joint-distribution policy | Detuning scan; harmonic positions; jitter and transverse-decoherence limits |
| Radiation reference | Perfect crystal, prescribed tracks, polarization-resolved two-wave field | Boundary conditions; kinematic limit; angular and energy convergence |
| Published benchmark | Reproduce one fully specified 2022 calculation | Absolute yield, angular distribution and polarization against the same source assumptions |
| Transport integration | Scattering, energy evolution and finite geometry | Formation-length/subdivision convergence and comparison with reference tracks |
| Accelerated execution | Batched and device implementations | Numerical parity and precision convergence without changing physical population |

The first two stages reuse existing beam and coherent machinery but still need
new scientific validation. The diffraction reference and full paper
reproduction are research-scale additions. Runtime estimates should follow a
reference calculation; a simple detector-direction count is not a useful cost
model for all coherence and diffraction regimes.

New source equations and approximations must receive derivation records and
ledger entries before production adoption. Fresh-context verification is
required; only a human may mark a claim signed off. This note neither creates
an implementation task nor commits the project to a delivery schedule.
