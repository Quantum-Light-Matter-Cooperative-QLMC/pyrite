# Mixed elastic CUDA kernel verification

## Scope and independence

Claims: the CUDA port of `mixed-elastic-split`, `mixed-elastic-soft-moments` and `mixed-elastic-hinge`, in the exact kernel at `359babde5`. The verified tree `~/pyrite-894-verify` on qlmc matches that commit file for file under `src/`. The branch was later rebased, which added `main`'s characteristic-scoring changes and no kernel change.

This page closes a review in two stages. An earlier fresh context wrote the probes and ran them as job 4039, on the qlmc RTX 5080, on released ELSEPA tables. That session left no verdict. A second fresh context, which wrote neither the kernel nor the probes, reviewed the outputs and the kernel source and wrote this page.

Independence limits: the second context read the probe scripts and outputs, the [re-verification](mixed-elastic-reverification.md) and the design note's CUDA paragraph before it read the kernel. It recorded its requirements first, but they were not blind to the 4039 results, and requirement K3 below was written knowing them. The probes and logs are scratch files, not committed: `scratch/verify-cuda/` in the verified tree.

## Requirements

Recorded before reading `_jit_kernel.py` or `_jit_mixed_device.py`.

- **K1, CPU contract.** Requirements R1-R7 of the [re-verification](mixed-elastic-reverification.md) hold in FP64: thinning at step bounds, clocks in optical depth, one energy for acceptance, element and quantile, a bound that includes both sides of a node, a step length fixed at its start, a surface that drops the hinge, and no extra draws at $C_1=0$.
- **K2, narrowed rates.** Rates and bounds may be FP32 only if acceptance moves by order $2^{-24}$ relative.
- **K3, hard quantile.** The hard deviate is $u=1-f_h(1-\xi)$. Below 1, FP32 spacing is $2^{-24}$, while $f_h$ reaches about $10^{-5}$ above 10 MeV, so a narrowed $u$ has few levels and can round to the top node, where $\mu=1$. The hard moments must still match. The detailed FP32 kernel uses the same lookup, so a bias here needs a detailed control.
- **K4, hinge.** Hinge moments stay FP64.
- **K5, state.** Step fields survive a continuation, and a capacity replay resets them, bit for bit.
- **K6, mixed off.** The generated source is unchanged.

## Verdict

| Claim, on the kernel | Verdict | First divergent term |
|---|---|---|
| `transport_precision="fp64"`, all three claims | **`rederived`**, within the limits below | none |
| `transport_precision="mixed"`, up to 3 MeV | **`rederived`**, within the limits below, with a measured tail bias on Si | $\mathrm{fl}_{32}(1-f_h(1-\xi))$ in the hard quantile |
| `transport_precision="mixed"`, 10 MeV and above on low Z | **`discrepancy`** against the Lewis first moment | same term |

The discrepancy is in `_elsepa_quantile_mu(..., table_real(R_ang))` on the hard branch. It is not in the split, the soft moments or the hinge. The detailed FP32 kernel should share it (see [the finding](#finding-fp32-hard-tail)), which job 4048 will test.

## Source-to-code comparison

| Requirement | Code | Result |
|---|---|---|
| K1 thinning and clocks | `_jit_kernel.py` mixed plan block; `mix_state` bounds overwrite `mu_hard` and `mu_rad`; optical depths reduced per row | matches, and FP64 rows match the CPU core |
| K1 one energy | elastic, inelastic and radiative proposals evaluated at `end_energy_keV` | matches |
| K1 bound | `_mixed_layer_hard_bound`: window ends plus $\Sigma(E_k)\max(f_{h,k-1},f_{h,k})$ at nodes | matches |
| K1 step | `_mixed_step_length`, `_mixed_step_end`, hinge at `mix_length` times a uniform | matches |
| K1 surface | the final branch clears the hinge and the step | matches |
| K2 | FP32 twins of the rate, bound and element lookups | passes, measured below |
| K3 | hard deviate computed in FP64, then narrowed by `table_real` | **fails at high energy**, see the finding |
| K4 | `_mixed_hinge_mu` is not an FP32 twin; moments and law are FP64, reading FP32 knots | matches |
| K5 | `mix_state` is a global per-history row, reset unless resuming | matches, measured below |
| K6 | generated source for seven switch sets in both precisions | byte-identical across trees |

## Measurements

All from job 4039 unless stated. Uncertainties are 1 sigma.

### CPU parity

Kernel FP64 against the CPU core, same streams, seed 909, default parameters. A history counts as identical when event kinds match and $L$ and $E_{\rm end}$ agree to $10^{-6}$ relative. Direction was not compared directly. It enters only through geometry, in the bounded cases.

| Case | Histories | Rows | Identical histories | Worst relative $L$ |
|---|---|---|---:|---|
| Si 30 to 5 keV | 4000 | 1,411,525 | all | $1.7\times10^{-10}$ |
| SiO2 30 to 5 keV | 4000 | 1,061,789 | all | $1.4\times10^{-10}$ |
| Si/W/Si stack, 60 to 5 keV | 4000 | 3,971,233 | all | $1.3\times10^{-8}$ |
| Thin Si, 30 keV | 4000 | 1,003,122 | all | $7.5\times10^{-9}$ |
| W 30 to 5 keV | 4000 | 4,309,861 | all | $7.3\times10^{-10}$ |
| Si 10 to 5 MeV | 2000 | 75,600 | all | $4.5\times10^{-12}$ |
| Shell, Si 30 to 5 keV | 1000 | 491,636 | all | $2.7\times10^{-11}$ |
| Radiative, Si 3 to 1 MeV | 500 | 46,661 | all | $1.4\times10^{-10}$ |
| Shell and radiative, Si 500 to 100 keV | 500 | 993,505 | all | $2.1\times10^{-10}$ |

Transmitted and backscattered counts match exactly in the two bounded cases. This parity is what carries the CPU re-verification over to the FP64 kernel.

### Moments against Lewis integrals

Reference integrals come from raw tables, without the kernel's helpers. Values are percent deviation from $1-\exp[-\int ds/\lambda_\ell]$, and for hard-event counts from the exact integral. Hard count, in both precisions and all 15 cases, is within $1.7\sigma$ of its integral, so the table lists only the moments.

| Case, $C_1=C_2$ | $\int ds/\lambda_1$ | FP64 $1-\cos$ | FP64 $1-P_2$ | Mixed $1-\cos$ | Mixed $1-P_2$ |
|---|---:|---|---|---|---|
| Si 30 keV, 0.05 | 0.30 | $-0.01\pm0.09$ | $+0.06\pm0.06$ | $-0.01\pm0.09$ | $+0.06\pm0.06$ |
| W 25.2 keV, 0.05 | 0.30 | $+0.05\pm0.11$ | $+0.07\pm0.07$ | $+0.05\pm0.11$ | $+0.07\pm0.07$ |
| W 30 keV, 0.05 | 0.30 | $+0.01\pm0.11$ | $-0.09\pm0.07$ | $+0.01\pm0.11$ | $-0.09\pm0.07$ |
| Si 30 to 24 keV, 0.2 | 0.85 | $-0.05\pm0.08$ | $-0.11\pm0.05$ | $-0.05\pm0.08$ | $-0.11\pm0.05$ |
| W 30 to 20 keV, 0.2 | 7.22 | $-0.10\pm0.09$ | $+0.12\pm0.07$ | $-0.10\pm0.09$ | $+0.12\pm0.07$ |
| SiO2 30 to 27 keV, 0.05 | 0.30 | $-0.19\pm0.12$ | $-0.13\pm0.08$ | $-0.19\pm0.12$ | $-0.13\pm0.08$ |
| Si 1 MeV to 900 keV, 0.05 | 0.20 | $+0.02\pm0.12$ | $-0.11\pm0.09$ | $+0.03\pm0.12$ | $-0.12\pm0.09$ |
| Si 3 to 2.6 MeV, 0.05 | 0.14 | $+0.02\pm0.12$ | $-0.05\pm0.09$ | $+0.13\pm0.12$ | $-0.09\pm0.09$ |
| W 3 to 1.5 MeV, 0.05 | 5.57 | $-0.01\pm0.08$ | $-0.02\pm0.06$ | $-0.02\pm0.08$ | $-0.03\pm0.06$ |
| W 10 to 5 MeV, 0.05 | 2.13 | $-0.12\pm0.09$ | $+0.05\pm0.06$ | $-0.08\pm0.09$ | $+0.02\pm0.06$ |
| Si 10 to 5 MeV, 0.05 | 0.34 | $-0.08\pm0.07$ | $-0.08\pm0.05$ | **$+0.49\pm0.07$** | $-0.14\pm0.05$ |
| Si 10 to 5 MeV, 0.2 | 0.34 | $-0.08\pm0.10$ | **$-0.31\pm0.07$** | **$+0.51\pm0.10$** | **$-0.37\pm0.07$** |
| Si 30 to 20 MeV, 0.05 | 0.06 | $+0.07\pm0.14$ | $+0.06\pm0.11$ | **$+7.22\pm0.21$** | $-0.30\pm0.11$ |

The FP64 residual at Si 10 MeV and $C=0.2$ ($-4.5\sigma$) has the sign of the second-order scheme residual found on the CPU core at $C=0.2$, where a track holds few hinges. Whether a step limit removes it here is not yet measured.

On complete steps, FP64, all 15 cases: hinge $\sum\mu/\sum\langle\mu\rangle$ is within $1.9\sigma$ of 1, with errors of 0.02-0.06%. Its second moment is within $1.6\sigma$ of 1. Moments of $\tau/s$ are within $1.8\sigma$ of $1/2$ and $1/3$. Each complete step holds exactly one hinge.

### Bitwise contracts

60 comparisons on released tables, both precisions: $C_1=0$ against detailed (with $C_2=0$ and $C_2=0.05$), continuation at three schedules and capacity replay, on the continuous, compound, stack, thin-exit and MeV cases. None differ.

The same probe failed on the shell, radiative, exact-radiative and shell-radiative cases, in both precisions (8 errors), from a bug in the probe's own comparison function when a result held a list. Those cases have no released-table evidence yet. `test_transport_mixed_elastic_cuda.py` covers them on a synthetic table, and its 21 tests passed in this job. Job 4048 reruns them with the bug fixed.

Urban straggling on the mixed path is refused with a clear error, as decided on 2026-11-06.

### Device rounding

| Quantity | Result, mixed precision |
|---|---|
| Rate, FP32 against FP64 | mean $-6\times10^{-7}$, rms $3$-$5\times10^{-4}$, worst 0.2-0.4 within a rounding of a node, where the frozen $f_h$ jumps. Largest integrated bias across a node window $3.4\times10^{-6}$ |
| Bound exceeded by the rate | single elements: never. Compounds: up to $3.4\times10^{-4}$ of samples, by at most $5.4\times10^{-8}$ relative |
| Element selection | rates to about $6\times10^{-7}$ relative, the lattice resolution |
| FP32 against FP64 rows, paired, at or below 30 keV | moments differ by at most $0.002\pm0.002$% |

These are all negligible against the Monte Carlo errors here. K2 passes.

(finding-fp32-hard-tail)=
## Finding: FP32 hard tail

Narrowing the hard deviate puts excess weight on large deflections, growing with energy as $f_h$ falls:

| Case, $C=0.05$ | Paired $1-\cos$, mixed minus FP64 | Paired $1-P_2$ | Events above $\mu=0.9$ per track, FP64 | Excess, mixed |
|---|---|---|---|---|
| W 3 to 1.5 MeV | $-0.006\pm0.011$% | $-0.002\pm0.008$% | $8.7\times10^{-3}$ | $+7.9\times10^{-4}$ |
| Si 1 MeV to 900 keV | $+0.012\pm0.002$% | $-0.005\pm0.001$% | $2.3\times10^{-4}$ | $+3.7\times10^{-5}$ |
| Si 3 to 2.6 MeV | $+0.105\pm0.004$% | $-0.034\pm0.003$% | $7.6\times10^{-5}$ | $+6.7\times10^{-5}$ |
| W 10 to 5 MeV | $+0.037\pm0.015$% | $-0.025\pm0.010$% | $1.9\times10^{-3}$ | $+1.7\times10^{-3}$ |
| Si 10 to 5 MeV | $+0.573\pm0.008$% | $-0.064\pm0.004$% | $8.6\times10^{-5}$ | $+1.5\times10^{-3}$ |
| Si 30 to 20 MeV | $+7.15\pm0.10$% | $-0.35\pm0.04$% | $1.3\times10^{-5}$ | $+2.8\times10^{-3}$ |

The device probes agree in order of magnitude. The relative change of the first moment on the top of the hard law, $\delta T_1/T_1^{\rm tot}$, is $1.3$-$1.7\times10^{-3}$ on Si at 3 MeV, $1.4$-$1.7\times10^{-2}$ at 10 MeV and $0.11$-$0.14$ at 30 MeV. At 30 MeV the same quadrature with an FP64 inversion on the FP32 lattice already gives 0.08-0.11, so most of it is rounding of the deviate and the rest is rounding of the tables.

A rounded deviate can only change where it falls in the top cells. It lands near $\mu=1$ with a probability of order $2^{-25}$ per detailed event, or $2^{-25}/f_h$ per hard event. A true moment of $T_1^{\rm tot}\approx2\times10^{-7}$ cannot absorb a $3\times10^{-8}$ weight at $\mu\approx1$. Near-reversal adds to $1-\cos$ and barely changes $P_2$, which fits the opposite signs in the table.

**Not specific to the port, by derivation.** For $\xi$ uniform on $(0,1)$, $u=1-f_h(1-\xi)$ is uniform on $(1-f_h,1)$. A detailed draw conditional on the same interval has the same law, and both pass through the same `table_real` and `_elsepa_quantile_mu`. So the mixed kernel's hard events have the law of the detailed FP32 kernel's events in that interval, and per unit path both carry the same excess. The soft part is FP64, so it adds none. Job 4048 tests this on the kernel: detailed and $C_1=0.05$ tallies at Si 10 to 9.5 MeV in both precisions. This page does not count the detailed kernel's bias as measured until then.

This fits `gpu-mixed-precision-tables`, whose CDF bound ($|\delta F|\le2u$) holds here. A bounded CDF shift does not bound a moment when the moment is comparable to $u$. That claim's gate stopped at 3 MeV and found no significant shift there.

**Effect on the frozen protocol.** Its scenes stop at 3 MeV and run both arms in the default mixed precision, so the shared part should cancel in mixed-against-detailed comparisons. If 4048 confirms that, the bias does not undermine the equivalence study. It is still an absolute bias of about 0.1% on the Si 3 MeV first moment, and it about doubles the rate of deflections above $\mu=0.9$ there, a small rate that matters only for rare-event observables.

## Filter checklist

| Item | Result |
|---|---|
| Notation across ledger, docstrings, note | pass; the design note's account of narrowing is accurate but does not mention the tail |
| Units at each boundary | pass; rows agree with the CPU core to $10^{-8}$ relative |
| Reference data | none new; tables are the released ELSEPA set |
| Tolerances | no fixed tolerance; deviations reported with errors |
| Cited evidence | job 4039 logs exist on qlmc; the tree matches `359babde5` |

## Limits of this check

- No comparison against PENELOPE. The soft and split derivations are the earlier pages', not redone here.
- Direction parity with the CPU core is indirect; job 4048 adds a direct check. Shell and radiative contracts on released tables are also pending there.
- Secondaries, positrons, exact-radiative parity, Mott tables and layer-crossing statistics beyond the stack case are not measured.
- The review covered the mixed branches of the kernel and the mixed device helpers, not the whole kernel. FP64 rows match the CPU core, which is the evidence for the rest.
- Timing is not addressed.
