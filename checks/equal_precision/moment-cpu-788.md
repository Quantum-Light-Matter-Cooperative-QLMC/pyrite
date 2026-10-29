# #788: CPU total-radiative-moment transport diagnostic

This **diagnostic** complements the published CUDA prototype and uses the
existing `moment-policy-788.json` and `analyze_moment_transport.py`. It does
not add a production option or change default transport. The CPU per-electron
reference is selected through an isolated runtime copy of `cores.py`. Exactly
three frequent total-moment calls use the prepared table; the *fourth* call
at the end of a hard event, element selection, exact conditional photon
sampler, actual photon-energy debit and RNG addressing remain unchanged.

From a synchronized remote repository checkout with all source tables installed:

```bash
sbatch checks/equal_precision/moment-cpu-788.sbatch
```

For a short standalone check (use the same inputs for all three variants):

```bash
uv run python checks/equal_precision/benchmark_moment_transport_cpu.py \
  exact /tmp/788-exact-audit.json --replay si-300-100um-0deg \
  --core per-electron --histories 256 --seed 78811 \
  --batch-histories 256 --probe-electrons 0 --output /tmp/788-exact.json
```

Replace `exact` with `table` and `refined` and give each a different output
and audit filename. The batch job writes separate `.json`, `.npz`, audit,
CPU host and job records under `evidence/moment-cpu788/<jobid>/`, followed
by an analysis. The sample JSON contains source, policy and table digests.
The audit includes the original and transformed CPU core hashes, component
held-out error, preparation time, host table bytes and within-process replay
hashes. A re-run with the same seed can check reproducibility across
processes from the recorded array digests; no such agreement is assumed.

**Interpretation:** warm request time includes CPU transport and reductions;
Numba JIT/warmup and table construction are separate costs. Comparisons use
single-thread CPU per-electron transport, **not** the CUDA execution engine
and not equal-achieved-precision throughput. The pilot's 2,048 histories
per seed are a *screen* only. Zero-variance and rare-discrete observables
remain `unresolved` under the frozen policy; do not treat unchanged counts
as ensemble equivalence. Do not mix the CPU and CUDA `.json` roots when
running the paired analyzer, which keys cases by scene, seed and variant.
CPU/CUDA statistical comparisons and wider energy/material anchors remain
separate qualification work before production opt-in or default promotion.
