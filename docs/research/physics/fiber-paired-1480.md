# Paired frozen-transport fiber diagnostic (#1480)

Research-only driver: `checks/fiber_paired_1480.py`. No production defaults,
reflection selector, catalog schema, or model/table identity changes. This is
an evidence-generation harness, not a claim of validation or convergence.

## Arms and interpretation

| Arm | CXR intensity sum | BremsLib background |
| --- | --- | --- |
| A | Basal only, scalar B33 | Basal share removed |
| B | Full fiber, scalar B33 | Full-domain scalar share removed |
| C | Full fiber, projected uniaxial tensor | Full-domain tensor share removed |
| D | None | Original uncorrected tables |

One trajectory population is loaded or transported once, then staged once.
The line and continuum populations must be equal and complete. Every score
uses the same detector direction, geometry, and photon bin centers. Segment
fingerprints are checked before and after the study. All arms retain the
original transport (including its energy losses); D is an **amorphous
emission control**, not an independently transported amorphous target.

The scalar B33 is taken from the same #535 carbon thermal row as the tensor
rather than the rounded catalog B, so the basal term is identical in A/B/C.
The catalog value is also recorded. The production reflection enumerator is
called once with B=0 and the texture restriction temporarily lifted, then
restored. The production strength/extinction threshold still applies. The
basal set is the exact `00l` subset of the full candidate set, including both
signs; neither thermal weighting nor azimuth changes membership.

For each non-basal vector, the #535 thermal site supplies
`B_eff = 8 pi^2 [U_b (1 - cos^2 alpha) + U_c cos^2 alpha]`, where alpha is
its angle to c. Since HOPG has one element with a common tensor, applying
that scalar per reflection to the production CXR couplings is exactly the
uniaxial amplitude attenuation `exp(-g^T U g / 2)`. Equal projected B values
are grouped only to amortize kernel calls, not to omit symmetry mates.
Existing graphite site averaging and assumed room temperature are inherited
from #535. hBN is deliberately refused: its element-resolved tensor data
are not present in that thermal catalog as of this slice.

B and C average non-basal **intensities** over the same uniform fiber rolls:
`N` midpoint nodes on one 60-degree sector, the opt-in model's nodes at zero
offset. The reflection set is closed under 60-degree rolls, and equally spaced
full-circle nodes would alias whenever `gcd(N, 6) > 1`.
The basal score is invariant under those rolls, so it is evaluated once and
added to both averages. The stored per-azimuth arrays contain non-basal
spectra, not amplitudes. Existing mosaic settings of the source case are
retained. No coherence across crystallites or domain-size model is added.

## Continuum subtraction

The diagnostic reuses the production #1250 released universal shell kernel
`H(|g|; T, theta, x)`, which averages over **isotropic orientations**, not
conditional fiber orientations. Its interpretation is the lattice-coherent
share of the *amorphous* continuum. The CXR score itself uses the fiber
orientation quadrature. These are different averages on purpose; this slice
measures the full-domain extension of the existing continuum partition,
not a new derivation of a texture-conditioned diffuse cross section.

For A/B/C the shell weights are computed from the same reflection sets and
Debye--Waller factors used in their line scores, with the production `U_g`
and elemental allocation. For C, sum the shell weights over all projected-B
groups **before** passing them to `_lattice_share_tables`. Correcting and
clipping each group separately would not reproduce the full-set DDCS clamp.
Each continuum is scored directly from its own corrected tables, on the same
frozen segments. D uses the original tables. The line energy cutoff is
passed to the correction; missing tables and out-of-domain kernel coverage
are errors rather than silent fallback. Inspect `clamped_nodes` in the
summary: clipping indicates the model needs scrutiny, not automatically
successful physical subtraction.

CXR uses production bin means, whereas the smooth continuum is evaluated at
those same cell centers and integrated by the midpoint rule. Refine the
spacing to check continuum/edge quadrature and line-shape convergence.
Reported total is **CXR + BremsLib only**; other components (characteristic,
transition and diffracted radiation) are outside this comparison. The peak
FWHM is the contiguous half-height width of the dominant CXR peak; it is
null for zero or edge-truncated peaks. Absolute and relative differences are
reported; relative differences with a zero reference are null.

## Running

Start small (one fresh HOPG transport):

```bash
pixi run python checks/fiber_paired_1480.py \
  --energy 300 --thickness 5000 --tilt 45 --electrons 16 \
  --band 50 3000 --spacing 1 --cutoff 8 --azimuths 8 \
  --output ~/dev/pyrite-benches/1480/smoke
```

Prefer an existing trusted schema-2 trajectory artifact, for example a
`cleo_sub`, `bells_45deg`, or #1301 case with equal line/brem populations:

```bash
pixi run python checks/fiber_paired_1480.py \
  --artifact /path/to/trajectory.h5 \
  --band 50 3000 --spacing 1 --cutoff 8 --azimuths 8 \
  --output ~/dev/pyrite-benches/1480/cleo-8g-8phi
```

Artifact replay checks the original case, geometry and physics provenance
before applying the research-only emission overrides. The original artifact
and checkpoints are never written. The script loads the full artifact into
memory; start with small cases. Unsupported cases include channeling,
multilayer, split-cone, mixed transport, and unequal line/brem populations.
Do not silently trim these to force a comparison. The script does not
submit to SLURM; use the group's normal remote workflow for large runs.

Repeat on the **same artifact**, using fresh output directories, to converge
azimuth count (8, 32, 128), shifted quadrature (`--azimuth-offset 0.5`),
reciprocal cutoff (e.g. 8, 16, 24 inverse angstrom, within the released
kernel range), and photon spacing (1, 0.5, 0.25 eV). Unshifted midpoint grids
nest only when the count ratio is odd (8 and 32 do not). A high-energy upper band must be checked for
omitted peaks; a fixed band is not a statement of total emitted yield.
Tensor grouping is deliberately simple and may be expensive at high cutoffs.

Each completed directory contains `spectra.npz` (common axis, all four
CXR/background/total arrays, removed backgrounds, HKLs and per-azimuth
non-basal spectra), `summary.json` (source, fingerprints, reflection counts,
thermal provenance, continuum clamp diagnostics, yields, peak/centroid/FWHM,
S/B and timings), and `SHA256SUMS`. Run `sha256sum -c SHA256SUMS` there.
Retain raw evidence outside the repo; compare B-A (domain), C-B (thermal),
C-A (combined) and A/B/C-D (net yield relative to amorphous emission).

## Cutoff ladder

`--ladder C1 C2 ...` replaces arms A–D with the opt-in model
(`cxr-fiber-intensity-average`) over nested reciprocal cutoffs, on 60° sector
rolls (`--azimuths`). Each disjoint |g| band is scored once, and every cutoff
gets the matched continuum subtraction over its own full prefix domain. The
smallest rung must equal `score_fiber_case` (the `integrity` record) within
1e-9 of peak in FP64 or 1e-4 in FP32; otherwise the outputs are written with
`complete: false` and the script exits non-zero. Cases that `score_fiber_case`
refuses, and cutoffs outside the kernel, are rejected before line scoring.
`--components` also stores the PXR and CBS parts of the basal and non-basal
sums per cutoff (`g<C>_{basal,nonbasal}_{pxr,cbs}`; their sum differs from the
total by the PXR–CBS interference term). This route calls `mc_cxr_spectrum`
directly with the runner's single-slab options; the integrity check confirms
it reproduces the runner's total. Production CXR returns no
polarization-resolved output, so polarization is not recorded.

## Remaining work

hBN needs site-specific thermal tensors (#1482). Not done: bounded-block
artifact replay, detector acceptance integration, paired independent-seed
uncertainty, a texture-conditioned continuum subtraction and domain size
(#1485).
