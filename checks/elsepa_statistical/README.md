# ELSEPA node-mixture experiment (#895)

Study-only `--law quantile|mixture` switch around the #821 CUDA benchmark.
Production imports none of these samplers; do not reuse production checkpoint
identities for candidate output. Each invocation uses a fresh process.

`pilot.sbatch` runs the analytic CUDA preflight, then W 1 MeV at 4000
histories with seeds 89501–89503 in both precisions. Reduce the complete run:

```bash
pixi run python checks/elsepa_statistical/analyze.py evidence/elsepa895/JOB_ID \
  --output evidence/elsepa895/JOB_ID/summary.json
```

The reducer requires complete three-seed pairs, matched controls/table keys,
and intact primary-score digests. It reuses #838's headline selection and
paired 95% intervals at epsilon/3; zero-yield and non-headline diagnostics
do not establish equivalence. This bounded pilot is not the full-anchor gate.

`samplers.py` implements CPU and exact-CUDA candidates (FP64 and mixed table
precision). Given the existing polar uniform `x`, select the upper energy
node when `x < f` and invert at `x/f`; otherwise select the lower and invert
at `(x-f)/(1-f)`. Endpoint weights preserve `x`. Conditional on either node,
the remapped deviate is uniform, so this implements the independent mixture
law without consuming another counter-addressed draw. Selection and remapping
stay FP64 in the mixed build; its existing complementary-CDF inversion stays
FP32. The continuous-law proof does not certify FP32 inversion accuracy.

This applies only to **detailed elastic transport**. It is not a replacement
for the hard-tail conditional quantile in mixed elastic transport or cone
splitting. The runner refuses mixed-elastic switches and runtime kernels.
Cross-section interpolation is unchanged. The CPU candidate is tested on
analytic node distributions; CPU full-transport integration is not supplied.

`run.py` wraps `checks/geant4_performance/pyrite_bench.py`. `timing.py` records
CUDA events around every exact transport launch for both arms. Cold compilation
and warmup are reported separately. Warm request timing includes event-recording
overhead in both arms; event totals and retained rows give kernel ns/row.
No profiler replay timings enter this comparison.

Remote GPU only, from a clean synced checkout:

```bash
pixi run python checks/elsepa_statistical/run.py --law mixture w-1MeV-100um \
  --histories 4000 --seed 89501 --rows-per-round 1024 --device-staging \
  --transport-precision mixed --no-audit \
  --estimator continuous --cone-mode point --output evidence/mixture.json
```

Before adoption, run all six original #821 scenes with at least three seeds
in both precisions, alternate arm order, and evaluate per-primary paired
confidence intervals against the #838 margin at epsilon=2% (epsilon/3).
Report unresolved intervals explicitly. Timing and moment bounds alone cannot
establish headline-observable equivalence. Independent physics review remains
required; no production sampler, model marker, or default is adopted here.
