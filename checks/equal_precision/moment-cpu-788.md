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

## Held-out binary endpoints and wider CPU anchors

`moment-cpu-heldout-788.sbatch` freezes fresh seeds 78821–78823 and eight
cases in `moment-heldout-788.json` before collecting results. It includes
the Si/W pilot anchors, 100/800 keV and oblique anchors, a 10 keV photon
cutoff and the supported SiO2 case. Each policy uses 2048 histories per seed.
The existing three-seed pilot and its zero-variance rule remain unchanged.

`qualify_moment_fractions.py` qualifies only transmission and backscatter.
For paired binary outcomes A,B, absolute mean bias is at most P(A != B).
It bounds this probability with a one-sided exact binomial upper confidence
limit, allocating 0.01/32 failure probability to each of the 32 fixed
case/variant/endpoint comparisons. This yields simultaneous 99% coverage
by the union bound, assuming independent incident-primary pairs. Zero
observed mismatches still give a positive upper bound. No optional stopping,
pilot reuse, or shrinking the comparison family for missing cases is allowed.
An incomplete seed group remains unresolved. Source, physical table, ordered
stream fingerprints, score digests and fixed sample sizes must agree.
Hard counts and continuous means are outside this binary qualification;
passing these endpoints cannot open the whole production gate.

Six additional preflights compare repeated within-process replay and retain
scores from two independent processes under each policy. These records must
be checked after collection; merely running them is not a replay verdict.
The batch uses an isolated remote checkout and records source sync metadata.
Job failure or a timeout leaves the qualification incomplete.

### Interaction with #795

#795's element-batched photon-angle completion is separate WIP in
`issue-795-batched-photon`; its source, dirty documentation and job 2668
evidence must be preserved. Job 2669 retains scalar completion at its frozen
pre-rebase source `8db53f2d`. The current branch is rebased onto main including
#795 and #799; it is not the measured source of that job. Future runs on this
combined source require a separately identified protocol rather than reusing
the held-out protocol's scalar-completion exclusion.
The 2666 CPU ratios and held-out run cannot be multiplied by #795's gains.
Any integrated performance claim needs matched baseline/candidate requests
on the combined source, with completion provenance and peak memory retained.
