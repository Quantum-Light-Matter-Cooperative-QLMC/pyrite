# Repaired multi-geometry oracle summary

Date: 2026-08-23  
Harness checkpoint: `33a463b5`  
Remote: configured `qlmc` host, NVIDIA GeForce RTX 5080, CUDA backend

This is decision evidence for issue #23, not a validation anchor or human
sign-off. It establishes no acceptance threshold.

## Reproduction

The four named geometry presets in `checks/pixel_reconstruction_oracle.py`
were run after `pyrite remote sync`. Each run used 4,000 line electrons,
2,000 bremsstrahlung electrons, 200,000 Timepix response samples, and fixed
seeds 1, 2, and 3:

```text
PYRITE_MC_BACKEND=cuda uv run --no-sync python \
  checks/pixel_reconstruction_oracle.py \
  --geometry <preset> \
  --n-electrons 4000 \
  --n-electrons-brem 2000 \
  --timepix-n-mc 200000 \
  --seeds 1 2 3 \
  --output agentdocs/tasks/feature/timepix-pixel-spectra/evidence/oracle-report.repaired.<preset>.json
```

Presets:

| preset | detector polar | target tilt | target tilt azimuth | purpose |
|---|---:|---:|---:|---|
| `baseline` | 60° | 30° | 0° | original coplanar geometry |
| `broken-symmetry` | 60° | 30° | 45° | non-coplanar control |
| `detector-on-g` | 20° | 20° | 0° | reciprocal-vector alignment danger case |
| `near-pole-not-g-aligned` | 20° | 30° | 0° | 10°-off-`g` control |

Each JSON file has 144 rows: three seeds, four angular shapes, and twelve
representative pixels. Every row records the geometry/statistics, exact
pixel-to-tile offset, tile occupancy, direct/reconstructed dominant peak and
FWHM, significant peaks, intrinsic and Timepix lineshape distance, and legacy
flux/centroid/continuum metrics. Exact pixel filter transmission is applied
before the Timepix response.

SHA-256:

```text
bfd8edcfa92fdb67d5638253ec16cdbb9e356762501298428c594429ba0e06ae  oracle-report.repaired.baseline.json
8641558f08b086aaa241d03dd5bcdbbe9ab6d2b28958d3567d830cc230b58945  oracle-report.repaired.broken-symmetry.json
9a05f65ad9506dc7a21f33203e3e2200099243b044c876d794f6203e3da0c696  oracle-report.repaired.detector-on-g.json
484719b73937a2e1b2798bf8f00496223d350a08184eff26714caa898aa93072  oracle-report.repaired.near-pole-not-g-aligned.json
```

## Results

Values below are the median `[minimum, maximum]` across the three per-seed
worst absolute errors over representative pixels whose coarse tile contains
more than one pixel. Singleton tiles are excluded. `(9,9)` is therefore the
exact identity case and is not accuracy evidence.

### Intrinsic line-flux error

| geometry | `(1,1)` | `(3,3)` | `(5,5)` |
|---|---:|---:|---:|
| baseline | 23.6% `[23.4, 23.8]` | 5.82% `[5.63, 5.90]` | 2.95% `[2.90, 3.11]` |
| broken symmetry | 22.2% `[22.1, 22.7]` | 5.49% `[5.25, 5.55]` | 2.38% `[2.30, 2.51]` |
| detector on `g` | 67.3% `[66.2, 68.8]` | 36.2% `[35.2, 36.6]` | 14.1% `[13.9, 14.8]` |
| near pole, not `g`-aligned | 127% `[119, 135]` | 20.3% `[20.1, 22.2]` | 12.8% `[12.4, 13.3]` |

### `(5,5)` diagnostics

| geometry | filtered line flux | intrinsic shape TV | Timepix flux | Timepix shape TV | largest dominant-peak switch |
|---|---:|---:|---:|---:|---|
| baseline | 2.95% `[2.90, 3.11]` | 0.0384 `[0.0371, 0.0395]` | 2.65% `[2.51, 2.92]` | 0.00269 `[0.00261, 0.00290]` | 1258 → 1252 eV |
| broken symmetry | 2.08% `[1.69, 2.29]` | 0.0396 `[0.0391, 0.0409]` | 2.56% `[2.23, 2.80]` | 0.00286 `[0.00260, 0.00295]` | 1258 → 1252 eV |
| detector on `g` | 57.5% `[57.1, 57.6]` | 0.0675 `[0.0641, 0.0691]` | 42.7% `[42.6, 42.8]` | 0.0220 `[0.0214, 0.0222]` | 1624 → 247 eV (seed 3, `corner_tr`) |
| near pole, not `g`-aligned | 19.5% `[19.3, 20.2]` | 0.0553 `[0.0533, 0.0571]` | 17.9% `[17.8, 18.5]` | 0.00210 `[0.00207, 0.00217]` | 1519 → 280 eV (seed 3, `center`) |

Continuum error at `(5,5)` remains much smaller: worst-per-seed medians are
0.139%, 0.300%, 0.0292%, and 0.0669% in table order. Line reconstruction,
not continuum, is the binding behavior in these scenes.

## Implementation-context review

- The repaired baseline and broken-symmetry flux results reproduce the
  historical qualitative ordering across all three seeds.
- `(5,5)` does not meet the earlier reviewer-proposed 1% line-flux bar in any
  nontrivial geometry. This comparison is informational because a human has
  not signed off that bar.
- Exact filter-before-response ordering is consequential, not cosmetic. In
  the detector-on-`g` case it amplifies the `(5,5)` worst flux error from about
  14% intrinsically to about 58% after filtering; the Timepix integrated error
  remains about 43%.
- Multi-seed line-flux maxima are fairly stable, but dominant-feature identity
  is not. Seed 3 changes the detector-on-`g` worst pixel from a 1624 eV direct
  feature to a 247 eV reconstructed feature. The non-`g`-aligned 20° control
  switches from 1519 eV to 280 eV. Full-window centroid or flux alone would
  conceal these failures.
- The 10°-off-`g` control remains strongly non-smooth. The danger cannot be
  limited to exact detector/`g` coincidence from this evidence.
- Uniform `(9,9)` gives zero error only because each fine pixel is its own
  singleton tile; it does not justify a coarse default.

No production physics, equations, units, coordinate conventions, or
normalization changed in this slice, so no validation marker or ledger row is
added. Human reconstruction-policy/tolerance review remains required before
generalizing `PixelScorer` or starting Slice 2 around a default policy.
