# Full-fiber CXR: opt-in emission model (issue #1480, phase II slice 1)

The preliminary frozen-transport investigation motivates a full reciprocal
lattice treatment of ideal fiber-textured HOPG. A random grain roll does not
remove nonbasal intensity. This is an implementation option, not a claim of
physical validation or a replacement for the default basal-only model.

## Using it

On the issue branch, with the existing verified BremsLib and lattice-share
assets, use the same transported scene as the paired diagnostic:

```python
from pyrite.config import CxrReflections, FiberAverage, Numerics
from pyrite.montecarlo.spectrum.cxr.fiber import FiberCXRModel
from pyrite.montecarlo.spectrum.cxr.fiber_case import score_fiber_case

# Only the fiber field is read; mosaic settings come from the scored case.
numerics = Numerics(
    cxr_reflections=CxrReflections(
        fiber=FiberAverage(cutoff_invang=8.0, azimuth_count=32, reflection_batch_size=256),
    )
)
# Tensor at its measured temperature; pass temperature_K= to override.
model = FiberCXRModel.from_numerics(numerics)
# case and tp are the runner case and full transported input record;
# grid is a finite, increasing photon-energy axis in eV.
result = score_fiber_case(model, case, tp, grid)
cxr, brem, total = result["cxr"], result["brem"], result["total"]
```

Since #1483, `simulate()` and campaign runs score `CxrReflections.fiber`
directly; the case then carries the model record as `cxr_fiber`. Since
#1489 an explicit `EnergyBins.cxr` grid is optional: the default automatic
line grid seeds from every domain row at every roll node at weight `1/N`,
and its truncation audit bounds the roll-averaged spectrum. Its ceiling
is the kinematic bound over the domain's |g|, and stored basal rows are skipped
([ADR-0024](../../adr/0024-texture-versus-reflection-domain.md)). Routes
outside the qualified envelope refuse instead of falling back to the
basal-only domain. `cutoff_invang` has no default. The case's
`mosaic_mc_fwhm_rad` and `mosaic_mc_nodes` set the mosaic quadrature.
Campaign runs lower them from `mosaic_route` and `mosaic_nodes`; the direct
scorer above does not read those two beside `fiber=`.
These cutoff and azimuth values are starting points, **not converged defaults**.
`model.prepare()` also exposes a low-level `spectrum()` for independent line
studies, and `corrected_bremslib_tables()` for use with the same prepared domain.
Use the paired entry point for CXR plus continuum; adding the full amorphous
continuum would double-count the CBS share. Its `total` omits characteristic,
transition and diffracted radiation. The result carries the model marker, exact
reflection and thermal rows, temperature/source, quadrature and subtraction
record (including clamped nodes). Do not cache it under a basal-only case key.

## Model and limits

For each reciprocal vector in the finite domain, project the #535 carbon
covariance as `B(g) = 8 pi^2 (u_b + (u_c - u_b) cos^2(g,c))`. Existing coupling
rows apply `exp(-B(g) |g|^2 / (16 pi^2))` to the amplitudes for both PXR and
CBS; the same projected B enters the shell `|U_g|^2`. All Friedel mates and
symmetry members are retained. Basal rows are scored once. Other rows are
summed as intensities over `N` equally weighted midpoint rolls of one
60-degree sector, applied before mosaic tilt and reciprocal miscut. The
domain is closed under 60-degree rolls (verified), so this is the uniform
full-circle average with `N` distinct orientations; equally spaced
full-circle nodes would alias whenever `gcd(N, 6) > 1`. The native kernels preserve recoil,
escape and PXR–CBS interference within each reflection. Batch size bounds
coupling table allocations; it is not a physics truncation. The continuum
shell share is accumulated over all reflections before the nonlinear clamp.
It uses the existing isotropic shell kernel: not a texture-conditioned diffuse
model. Transport retains its uncorrected radiative tables.

The paired route is currently ordinary incoherent, single-slab HOPG, BremsLib,
complete unsplit trajectories with deterministic mosaic quadrature. It refuses
hBN (no qualified element-specific tensor), channeling, two-beam scoring,
coherent emission, cone splitting and unsupported energy/reciprocal table
coverage. No finite crystallite or inter-domain phase model is implied. The
line density is bin-mean, while the continuum is node-sampled. For comparisons,
use identical grids and explicitly integrate with the corresponding weights.

## Required numerical qualification before normal campaign integration

Compare against arm C in `checks/fiber_paired_1480.py` using the same artifact,
thermal temperature, cutoff, roll origin and photon axis. Require spectral and
integrated parity at fixed numerical controls (FP64 CPU reference; qualify
FP32 device tolerance separately). Repeat with reflection batches 64/256/512
and azimuth counts 8/32/128, then cutoffs 8/16/24 inverse angstroms where the
verified kernel permits them. Compare total, CXR, continuum, L1 spectral change,
line windows, peak shape, clamp counts, and elapsed time/VRAM. If the kernel
cannot cover a needed domain, regenerate and validate it separately rather
than silently truncating subtraction. The candidate metric retains the
existing 1e-9 relative family floor; it is not an omitted-yield certificate.

### First qualification (2026-11-26, local FP64 CPU, CXR lines only)

HOPG 30 keV, 500 nm, tilt 15°, lab 45°, 150 electrons, one frozen transport,
20–6000 eV on a 2 eV axis. Raw: `~/dev/pyrite-benches/1480-fiber-nonbasal/parity/`.

| Comparison | Result |
|---|---|
| Model (cutoff 6, 32 rolls) vs independent per-family arm of `checks/fiber_nonbasal_cxr_1480.py` | L1 6.6e-5, max 1.2e-4 of peak. Fully explained by basal B: the model projects 8π²U∥ = 1.3027 Å², the check keeps catalog 1.30 Å². The check's 30 extra members are extinct (h−k≡0 mod 3, odd l) and score zero |
| Reflection batch 64 / 512 vs 256 | 1.3e-12 / 0 |
| 8 / 128 rolls vs 32 | yield −3e-4 / +6e-5; spectral L1 4.1 % / 1.5 % |
| Cutoff 8 / 10 Å⁻¹ vs 6 | yield **+22 % / +43 %** (242 / 478 reflections) |
| Wall time, cutoff 6, 32 rolls | 44 s model, 137 s check (per-family loop) |

Integrated yield converges by 8 rolls; spectral shape still moves at 32 (arm F below).
The cutoff was not converged at 10 Å⁻¹; see the ladder below.

### Cutoff ladder with the paired continuum (2026-11-26, qlmc RTX 5080)

`checks/fiber_paired_1480.py --ladder`: 600 electrons, 32 rolls, cutoffs
6–24 Å⁻¹, one transport per scene; the smallest rung reproduces
`score_fiber_case` to 3e-16. Device parity on one 32-electron transport
(cutoffs 6/10) against FP64 CPU: device FP64 yields within 2e-6, spectral L1
≤ 4e-5, continuum exact to 1e-15; device FP32 within 3e-6, L1 ≤ 1.3e-4,
continuum 1e-7. Device FP64 ran as fast as FP32 on this card, so the ladder
used `PYRITE_FP64=1`. FP32 is adequate at this level for the line route but
is not otherwise qualified.

| | 30 keV, tilt 15°, lab 45° | 300 keV, tilt 60°, lab 45° |
|---|---|---|
| CXR line yield 24 vs 6 Å⁻¹ | 1.97× (last step 20→24: +2.0 %) | 1.85× (+2.0 %) |
| Continuum removed, 6 → 24 Å⁻¹ | 5.2 % → 13.0 % (no clamps) | 3.5 % → 8.6 % (no clamps) |
| CXR + continuum, 24 vs 6 Å⁻¹ | +3.8 % (last step +0.1 %) | +3.0 % (+0.1 %) |
| Basal share of CXR, 6 → 24 Å⁻¹ | 0.31 → 0.16 | 0.34 → 0.19 |
| Wall, host RSS | 15.6 min, 1.4 GB | 11.5 min, 1.3 GB |

Treat 24 Å⁻¹ as the converged cutoff for 30–300 keV HOPG at these
geometries (2 % residual in lines, 0.1 % in total); this is a measured value
for these scenes, not a default. Arm F (128 rolls at 6/10/16 Å⁻¹, same 30 keV transport): integrated
CXR within 0.06 % of 32 rolls, spectral L1 2.4–3.3 %; use 32 rolls for
yields and at least 128 when narrow line shapes matter.

Raw: `~/dev/pyrite-benches/1480-fiber-nonbasal/{ladder,gpu-parity}/`.

### Campaign route qualification (#1483, 2026-11-26, qlmc-queen FP64 CPU)

The lowered `CxrReflections(fiber=...)` Scene, scored by the campaign line
seam and spectrum phase, against the Q arm of
`checks/fiber_nonbasal_cxr_1480.py` (projected basal B) on the same campaign
transport (`campaign` subcommand). HOPG 30 keV, 500 nm, tilt 15°, 45°
observation, 150 electrons, 2 eV bins to 6000 eV. Code `b09fa1c6a`, jobs
16148–16155; raw: `~/dev/pyrite-benches/1483-fiber-campaign/`.

| Comparison | Result |
|---|---|
| Campaign vs Q arm, cutoff 6 / 8 Å⁻¹ (108 / 242 reflections) | peak 2.8e-11 / 2.8e-11, L1 7.6e-11 / 6.3e-11 |
| Campaign vs Q arm, 300 keV, cutoff 6 | peak 1.1e-11, L1 1.2e-10 |
| Batch 64 / 512 vs 256 | L1 7.4e-12 / 0 |
| 8 / 128 rolls vs 32 | yield +3.4e-4 / +1.9e-4; spectral L1 13 % / 5.1 %, peak 42 % / 11 % |
| Cutoff 8 / 10 vs 6 | yield +20.5 % / +43.4 % |
| Line seam wall time, cutoff 6 / 8 / 10 (32 rolls) | 10 / 15 / 29 s; Q-arm loop 36 / 75 s; peak RSS ≈ 1.4 GB |

The residual campaign-vs-Q difference comes from grouping reflections
differently per kernel call (#1492), not from the fiber route. Integrated
yield converges by 8 rolls, but at 2 eV bins the line shape is still
moving between 32 and 128 rolls. Cutoff convergence remains open (#1484).

Remaining phase II work: joint roll/reciprocal adaptive convergence with a
reported unresolved status, symmetry-reduced quadrature with verified weights,
line-grid seeding/audits that know the orientation population, campaign/profile
serialization and hashing for the new model, full runner/CLI integration, and
independent physical validation. No production default, schema, shared table
identity, or existing golden is changed by this slice.
