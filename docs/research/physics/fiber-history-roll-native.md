# Independent history rolls: device-vectorized Phase IIa

This is an experimental fixed-N, retained-transport, explicit-grid route, not
an advertised campaign sampling mode. Based on #1527 branch commit
`bbae4dd8173dbc059d1e434eb5f28754194d8210` and the Phase I check.

## Design

The existing sampled-mosaic route already accepts a different reciprocal
vector and polarization pair for each row in `_accumulate_reflection`. The
coupling tables depend on reflection and energy, not the orientation. The
new private `_fiber_history_rolls` request supplies one sector offset per
incident primary (indexed by global retained `elec_id`). After normal
cutoff clipping and escape-piece expansion, the per-hkl accumulation route
finds unique represented primaries, gathers their rolls on the active
backend, and evaluates `R_m @ R_z(phi) @ g` and the sigma/pi basis using
NumPy or CuPy. The case's base azimuth is already included in `g`. The
reciprocal norm is invariant and uses the same FP64-before-REAL convention
as sampled mosaic. No coordinate of the beam, detector or trajectory is
rotated. One table batch is built per reflection batch per scoring block,
not per roll. The existing recoil, complex PXR/CBS amplitude, attenuation,
physical-flight grouping and reduction are reused. Mosaic quadrature, when
active, retains its original weights; the basal contribution remains a
separate ordinary call. The original incident count normalizes every block.
The continuum is not evaluated or changed by this check.

This vectorizes orientation geometry and uses the existing GPU row-physics
kernels, but is **not** a fused segment-by-reflection CUDA/HIP kernel. It
still pays the per-hkl route's launch costs. Device FP64 geometry per
represented primary may be expensive on consumer GPUs. Compare measured
wall time before optimizing further. Memory scales with one reflection's
history and row vectors plus the existing bounded reflection tables, not
with all histories times all reflections times all energies. Retained
transport memory is not bounded independently of N; `--native-blocks`
partitions only scoring. The roll array itself is O(N incident primaries).

The check uses the Phase I diagnostic RNG and supplied angles for exact
same-roll oracle comparisons. It is not a new production RNG namespace.
No FiberAverage config, case identity, checkpoints, adaptive stopping,
line-grid policies, or scientific defaults change. The private route
rejects truncation audits, coherent/temporal/capture modes, non-HOPG,
non-bin-mean scoring, sampled mosaic, miscut, grooves, layers and detector
cone copies. Do not invoke it from a persisted campaign. Global primary ID
semantics across streamed/local-ID transport and restart remain unqualified.

## Validation

Start on CPU FP64 and NVIDIA using the same roll seed in scalar `I` and
native `N`. The existing deterministic Q arm is a convergence diagnostic,
not a reference for exact same-realization parity. The check reports
`scalar_same_roll_parity` (mass L1, peak L-infinity, integrated yield error).
Do not assume bitwise equality: the scalar oracle may take the batched
route, while N takes the per-hkl route; floating-point ordering and the
invariant norm can differ. Investigate unexpected error, not just total
mass. No acceptance tolerance is silently imposed by this research check.

```bash
pixi run --locked python -m pytest -q checks/test_fiber_history_native.py
pixi run -e nvidia python checks/fiber_randomized_rolls.py \
    --histories 32 --cutoff 4 --step 4 --energy-max 13000 \
    --quadratures 4 --reference 8 --shifted --strata \
    --orientation-seeds 101 102 --independent --native \
    --production-parity --output scratch/fiber-roll/native-smoke --plot
```

Repeat with `--native-blocks 3` and a different output stem. The NPZ native
spectra and roll assignments should be numerically consistent across block
partitions. Compare the existing JSON transport fingerprint first. Repeat
with `--mosaic --mosaic-route mc --mosaic-nodes 3` for the composition order.
Use the configured CPU environment with `PYRITE_MC_BACKEND=cpu` and the same
arguments for a CPU oracle. Test grouped flights explicitly before admitting
those configurations to production.

After small same-roll parity, omit `--independent` (scalar reference is
expensive) and increase `--histories` to 4096, 32768, then larger if memory
permits. `--native` uses the same independent roll distribution at every N;
`N_seed*` and `N_mean_nonbasal` are saved like the other arms. The native
warm-up is separated from measured timings. Compare the complete radiation
wall-time including basal and setup, as well as error at comparable cost.
Run several tilt angles and a larger energy band before making conclusions.

## Remaining gates

- Same-realization scalar/native parity on CPU FP64, CUDA and ROCm; validate
  the existing grouped-flight route and active deterministic mosaic.
- Profile table setup, orientation kernels, per-hkl launches and line
  reduction independently. Only then decide whether a batched/fused kernel
  is required to attain a material end-to-end speedup.
- Define a stable production counter stream and case identity/versioning;
  prove global-ID offset, chunk, retry and restart invariance. Retain existing
  deterministic default and matched continuum subtraction.
- Implement sampled audit/grid/stopping parity, including the #1527 bounded
  pilot/exact tail and independent brem prefix. Do not admit adaptive or
  automatic-grid production runs until this passes.
