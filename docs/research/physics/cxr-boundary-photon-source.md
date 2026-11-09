# PXR/CBS boundary photon source: first evaluation slice (#411)

**Status:** Research contract and synthetic normalization probe, not a CXR
sampler, photon-transport handoff, or physics validation. Prepared against
`main` at `e7a9c84f` (2026-11-09). No production defaults, spectral outputs,
validation statuses, or photon-tracker admissions change. This is the
reference-surface branch of [#411](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/411),
not a restart of [#341](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/341).

## Decision for the first slice

The **escaped** source can be transported from a boundary in an *intensity-only*
model for linear downstream **mean-count** observables when the boundary field
admits a nonnegative coarse-grained radiance. It is not yet possible to
construct that source from the existing PXR/CBS angular spectrum without an
additional, validated spatial/coherence model. In particular, **do not**
interpret a numerical electron segment, a radiation formation substep, or the
midpoint of a physical flight as an independently sampled local photon birth.

`mc_cxr_spectrum` currently returns an already attenuated escaped marginal
$d^2N/(dE\,d\Omega\,dN_e)$ in photons/(eV sr incident electron), for one
sample-frame observation direction. `mc_cxr_spectrum_solid_angle` integrates
that marginal over angular tiles; it does not recover a position or polarization
distribution. A new intensity-only surface handoff would require a measure

$$
 d\nu_S(\mathbf x,E,\hat{\mathbf n})
 = B_S(\mathbf x,E,\hat{\mathbf n})\,dA\,dE\,d\Omega,
 \qquad \hat{\mathbf n}\cdot\hat{\mathbf n}_S(\mathbf x)>0,
$$

where $B_S$ is the **outward crossing** density, in
photons/(incident electron Å² eV sr). Here $dA$ is the physical surface area,
not projected area; the outward normal and crossing direction are both in the
lab frame. If expressed instead as conventional radiance $L_S$ per projected
area, $B_S=L_S(\hat{\mathbf n}\cdot\hat{\mathbf n}_S)$: the cosine factor is
present **once**, not applied again by the downstream tracker.

For any angular/energy cell $C$ in the far-field regime, the new spatial
marginal must satisfy

$$
 \int_S B_S(\mathbf x,E,\hat{\mathbf n})\,dA
 \stackrel{\mathrm{tested\ limit}}{=}
 \left.\frac{d^2N}{dE\,d\Omega\,dN_e}\right|_{\rm escaped,\,combined}.
$$

This equality is a **required closure test**, not yet an established
capability; finite detector distance, wavefront curvature and diffraction may
invalidate a naive transfer of an asymptotic angular spectrum to a nearby
surface. Sum over *all* relevant outward faces of a closed reference surface.
A plane intercepting only one exit hemisphere is not a full-source closure.

### Couplings, polarization and coherence

Within one physical interference group and one transverse polarization,
$A_s=A_{\mathrm{PXR},s}+A_{\mathrm{CBS},s}$ and the intensity is
$\sum_s|A_s|^2$. The separately useful PXR/CBS diagonals **cannot** be
added as photon populations: that loses
$2\operatorname{Re}(A_{\mathrm{PXR},s} A^*_{\mathrm{CBS},s})$.
Likewise `coherent` and flight-incoherent are **alternative** reductions, not
independent components. The existing model sums distinct reflections as
intensities (with their documented omitted cross-reflection terms); a proposed
sampler must reproduce the **same** selected reduction, not silently add
cross-reflection interference.

For polarization-dependent downstream response, the boundary must carry a
Hermitian positive-semidefinite $2\times2$ transverse coherency matrix
$J_S$ with $\operatorname{tr}J_S=B_S$ in a declared common polarization basis.
The existing scalar spectrum has already traced over the polarization basis;
no polarization state can be inferred from that scalar alone. The proposed
analysis scaffold allows either explicit cell-integrated $J$ or explicit
**trace-only** samples. It does not label absent $J$ as unpolarized.

An even more demanding observable (interferometry, diffraction or
phase-sensitive imaging) requires mutual coherence
$W_S(\mathbf x_1,\mathbf x_2,E,\ldots)$ or complex outgoing modes, not just
local $J_S$. At finite formation/diffraction scales even a flight-incoherent
ensemble need not possess a unique positive local photon-ray distribution.
A verified coarse-graining/decoherence prescription is therefore a **gate**.
No negative Wigner-like quasidensity may be silently clipped into a ray source.

### Reference-surface and tracker boundary

1. Choose a closed convex surface **outside the emitting crystal** and all
   modeled in-radiator formation/attenuation, intersecting each escaping ray
   once. Start with a finite rectangular slab footprint plus a small vacuum
   clearance. Store crossing position and outward normal in lab Å, photon
   direction as lab unit vector, energy in eV, physical source/case identity,
   and statistical weight in photons per incident electron.
2. Preserve the joint $(\mathbf x,E,\hat{\mathbf n})$ measure (and $J$ if a
   polarization-sensitive result is required), not independently sampled
   position/energy/angle marginals. Do not construct surface positions from
   `mc_cxr_spectrum` by applying an unverified uniform-face prior. In a
   computationally convenient discretization, each quadrature cell carries
   $w_j=\int_{C_j}d\nu_S\ge0$. Sampling cell $j$ with $p_j=w_j/W$ and $M$
   draws gives constant history weight $W/M$ **per incident electron**, with
   $W=\sum_jw_j$. Samples at nodes are not continuous within-cell draws.
3. The source is already escaped. Photon tracking begins in **vacuum** outside
   the reference surface: the radiator's narrow-beam escape factor is not
   applied a second time, and the radiator is not re-entered in this first
   downstream-only level. EPDL independent-atom scattering in filters is a
   modeled approximation, not Bragg/dynamical diffraction in the crystal.
4. Do not debit electron energy when creating **weighted mean-response**
   histories; physical, correlated photon events and electron recoil belong
   to a separate feedback decision (coordinate with #527). Surface histories
   must not also be scored through the original direct CXR estimator. The
   tracker and a separately derived direct estimator can coexist only with a
   disjoint or conditionally corrected partition.
5. A tally in photons/(incident electron eV) integrates the **detector
   aperture** exactly once. The current directional `mc_cxr_spectrum` has an
   additional sr$^{-1}$ and is not directly interchangeable. Any existing
   photon-tracker denominator based on `n_primaries` requires a deliberate
   conversion from already-per-primary surface weights; do not divide twice.

These requirements extend [ADR-0008](../../adr/0008-no-arbitrary-target-geometry.md)
and respect [ADR-0020](../../adr/0020-photon-transport-source-and-scoring.md).
The current [#341 photon tracker](photon-scene-handoff.md) supports opt-in
physical-aperture scoring for **other** source types; this page does not
claim that it can consume CXR events or preserve their polarization.

## Fixed benchmark for the next derived source

Use a deliberately bounded case rather than a generic source immediately:

| Parameter | First benchmark choice | Reason |
| --- | --- | --- |
| Radiator | h-BN slab, thickness 921 nm | Existing [Zhai h-BN comparison](../../validation/radiation-physics/zhai-hbn-921-detected.md) |
| Electron | 20 keV, fixed seed and population | Existing reference conditions; avoid MeV extrapolation |
| Crystal pose | tilt 17°, azimuth 130° | Existing reference orientation |
| Radiation | combined PXR/CBS, initially flight-incoherent | Baseline model; no artificial PXR/CBS populations |
| Source spectrum | 0.25–1.5 keV, grid refined near the existing resonances | Includes possible lower-energy scattered/fluorescence contributions |
| Surface | six faces of a finite slab, e.g. 1 mm × 1 mm lateral footprint; record beam footprint separately | Single-crossing closed surface; size is a **proposed** simulation choice, not a reported Zhai dimension |
| Detector | ideal absorbing disk, normal to the central 119° viewing ray, 20 mm from origin, radius 2.922 mm | Approximately 0.066 sr **for a point source at the origin**; finite-position acceptance must be ray-tested |
| External object | 0.5 µm carbon plate, perpendicular to the central detector ray, between surface and detector | Simple EPDL scattering/absorption/fluorescence test; not assumed optically thin |
| Control | vacuum; then plate with all collisions disabled; then fully tracked plate | Separates source normalization, geometry and collision contributions |

The detector distance, area, foil and footprint above are **new test fixtures**,
not extracted dimensions from the Zhai paper. Before interpreting any scatter
yield, compute the EPDL total optical depth $\tau(E)=\sum_i n_i\sigma_{i,\rm
 tot}(E)\,t/|\hat n\cdot\hat n_{\rm foil}|$ and check energy coverage and
entrance geometry. The first-interaction scattering probability along a
homogeneous path is

$$
 P_{\rm scatter,first}(E)=
 \frac{\mu_{\rm Rayleigh}(E)+\mu_{\rm Compton}(E)}{\mu_{\rm total}(E)}
 [1-e^{-\tau(E)}],
$$

**not** a predicted fraction of detector photons. The detected redistributed
count needs the differential angular kernels, fluorescence daughters and the
actual transport geometry. Off-cone and weak-direct benchmarks must not
restrict the source proposal to the detector cone. For a clean shadowed case,
add an explicit reviewed absorbing occluder in a *later* slice, rather than
pretending that a passive attenuation coefficient accounts for re-direction.

## Executable bookkeeping probe

`checks/cxr_boundary_contract_411.py` supplies an **analysis-only** discrete
surface representation and Philox cell sampler. Its checks in
`tests/checks/test_cxr_boundary_contract_411.py` cover outward crossings,
nonnegative measure, per-primary conservation, correlated cell sampling,
PXR/CBS interference as an algebraic counterexample, trace-preserving optional
polarization, and fail-closed inputs. It deliberately **does not** generate
CXR boundary cells, execute #341 photon transport, approximate internal
formation, or assign a validation-ledger status. The simple synthetic source
cannot establish that a physical positive boundary radiance exists.

## Gates and next slices

| Gate | Required evidence / decision | Current state |
| --- | --- | --- |
| `cxr-boundary-photon-source` | Derive actual outgoing surface intensity/coherency for a physical flight; prove positivity in the admitted coarse-grained regime; reproduce all-face angular marginal in vacuum and single-flight limits | **Proposed, unverified**; synthetic bookkeeping only |
| PXR/CBS polarization | Retain the combined per-polarization amplitude before reduction; basis-rotation and trace-to-scalar checks | **Not implemented** |
| Downstream-scatter increment | Fixed foil optical depths, ordinary-hit direct/redistributed/fluorescence tallies, off-cone injection and aperture-normalization comparison | **Not measured** |
| `cxr-photon-detector-estimator` | Derive conditional estimator with proven no-overlap zero-collision partition; compare matched confidence intervals and wall time against ordinary hits | **Proposed, unverified** |
| `cxr-internal-source-coarse-graining` | Compare formation length, attenuation and interaction lengths; demonstrate local/nonlocal wave-to-ray boundary, or explicitly decline | **Deferred** |
| Recoil/energy feedback | Integrate full-angle $\int E\,dN$ and compare with trajectory-sensitive observables, without duplicate weighted energy debits | **Deferred**, coordinate with #527 |

No new physics validation record is added until a physical model and
independent derivation exist; future claim promotion follows the
[validation methodology](../../validation/methodology.md). Use measured
CPU/device runtime, electron/photons seeds and populations, warmup, elapsed
times, memory and uncertainty when claiming performance. The current
angular line evaluator repeats angular scoring, **not** electron transport,
so compare at matched uncertainty rather than counting M directions as M
full electron simulations.
