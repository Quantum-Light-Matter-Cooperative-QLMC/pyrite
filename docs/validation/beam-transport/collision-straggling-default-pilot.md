# Collision-straggling default pilot

The 2026-10-23 pilot does **not qualify a default change**. It separates the
incremental shell soft-loss fluctuations from continuous/Urban fluctuations
under current transport models. Most near-range shell comparisons remain
statistics-limited. Continuous/Urban changes the sapphire control materially;
that is sensitivity to a different loss law, not evidence of better accuracy.
Existing `energy-loss-straggling` and `shell-soft-hard-transport` ledger statuses
remain unchanged. No new physics claim or human sign-off is asserted.

## Frozen workload

Code revision: `08b710f8f9513909b5d5ad9b542bc7a3b5ed05eb`, with the added study
scripts and documentation in the issue worktree. Transport source is unchanged.
See `checks/straggling_defaults/{protocol.json,study.py,analyze.py,run.sbatch}`
and its README for the complete reproduction contract.

The first subset uses 30 keV pencil electrons at normal incidence, a 5 keV
tracking threshold, 10 × 10 mm slabs, and thicknesses 0.05 and 0.8 of the
material's corrected-SBETHE CSDA range down to that threshold. The range only
sizes identical on/off slabs; it is not an independent range reference.
HOPG, silicon, MoS₂ and WSe₂ use shell soft/hard transport at 50 eV. Sapphire
has no packaged shell record and is the explicit continuous-stopping control.

All arms pin ELSEPA, corrected SBETHE, Kawrakow deflection, midpoint integration,
coupled BremsLib loss with a 1 keV hard-photon cutoff, exact transport, FP64 and
no numerical energy cap. No secondary tracks or line/detector estimators are
scored. Analog hard-photon counts above 10 keV are source diagnostics only.

Each arm has three independent seeds × 512 primaries after one identical-size
warmup. On/off seeds are disjoint. The Si 0.8-range case also has separate
on-state cap and collision-cutoff refinements and one-thread CPU on/off arms.
There are 26 complete groups, 78 sample records and 39,936 measured primaries;
the 20 baseline CUDA groups are complete. Every sample passes the segment
contract, terminal-population completeness and primary energy ledger. The
largest energy residual is below $4.5\times10^{-13}$ keV.

## Sensitivity results

Near-range transmission is normalized per incident primary. Intervals below
are pointwise normal approximations using independent-arm standard errors;
they do not have simultaneous coverage and are not accuracy intervals.
The frozen diagnostic margin is 2 percentage points for terminal fractions,
and 2% of the off-state conditional transmitted mean for energy.

| Material, 0.8 range | Off transmission | On transmission | On − off, percentage points (approx. 95% interval) |
| --- | ---: | ---: | ---: |
| HOPG, shell | 19.922% | 19.792% | −0.130 (−2.95, +2.69) |
| Silicon, shell | 3.190% | 2.799% | −0.391 (−1.60, +0.82) |
| MoS₂, shell | 0.260% | 0.326% | +0.065 (−0.32, +0.45) |
| WSe₂, shell | 0/1536 | 1/1536 | Unresolved; zero off-state sample variance |
| Sapphire, continuous | 1.628% | 5.859% | +4.232 (+2.90, +5.57) |

Sapphire's stopping fraction changes by −5.924 percentage points (approximately
−8.44 to −3.41). Its thin-film conditional energy variance changes from 0.132
to 0.919 keV². These demonstrate why Urban needs separate qualification.
They do not validate its distribution width against experiment.

The shell near-range energy populations are too small for tails: only 49/43
transmitted Si primaries, 4/5 MoS₂ primaries and 0/1 WSe₂ primaries (off/on).
The analyzer refuses conditional-mean intervals below 30 transmitted primaries;
this is a conservative reporting guard, not a newly calibrated acceptance
threshold. Every group's raw scores retain the selected counts and conditional
variance/quantiles. Variance and 1%/99% quantiles have no qualification intervals.
Thin-film shell mean-energy intervals fit the 2% diagnostic margin, but this
does not certify their variance or rare tails. Zero observed variance never
qualifies a margin.

The Si cap probes are active: 0.025% and 0.0125% produce about 1,891 and 3,990
energy substeps per primary, respectively, compared with zero at the default
cap. Total rows rise from about 449 to 2,340 and 4,440. Their near-range
transmitted populations remain only 37 and 38, so this ladder does not select a
converged default. Separate 100/200 eV cutoff probes remain underpowered too.

## Hardware and timing limits

The lab GPU is an NVIDIA GeForce RTX 5080, driver 610.47, CuPy 14.2.0, with
Python 3.14.6. Jobs request one CPU, one GPU and 24 GiB RAM, each bounded at ten
minutes. GPU calls synchronize around each timed interval. The retained warm
transport times are fixed-population diagnostics; no equal-precision speedup
is established. Si near-range CUDA off/on means are 0.643/0.464 s, and CPU
means 2.468/0.934 s. The changed loss/integration path and finite-population
statistics prevent interpreting this as a general straggling overhead result.
Host peak RSS is recorded; peak GPU allocation is not measured.

All six Si near-range CPU/GPU replica score sets agree at `rtol=1e-12`,
`atol=1e-10` in each field's units; this is one scene, not general parity.
The existing Urban exact/LUT and shell hardware modules pass **21/21 tests**,
including `test_cuda_sbethe_straggling_first_row_matches_host`, which compares
the applied loss through `E_end_keV`. The first run had two MoS₂ setup failures
because test isolation cleared `PYRITE_HOME`; the successful rerun redirects
platform data to the dedicated staged folder. Both logs are retained.

## Evidence and remaining scope

Dedicated remote checkout: `qlmc:~/pyrite-642-straggling`. Jobs: 2000 (Si),
2001 (initial anchors), 2002–2005 (HOPG, MoS₂, WSe₂, sapphire), 2006
(refinements), 2007 (CPU), 2008 (corrected anchors). Raw scores, hashes, exact
tables, source scripts, sync identities and scheduler/hardware logs are retained
under `evidence/straggling/`. Final archive:
`evidence/straggling-642-pilot-final.tar.gz`, SHA-256
`593ffa9aae91bc1384b6a4935a9d49785ca55fe83268eeb87731fe51bd75e178`.

Deferred coverage is explicit in the protocol: 5/100/300 keV, layered/grooved
cases, detector/line observables, rare-tail populations, independent experimental
or external-code accuracy, and held-out equal-precision timing. Reuse the
[equal-precision protocol](equal-precision-protocol.md) and #510's adapters for
those measurements. The pilot alone warrants no default, model marker, dataset
identity or omission-semantics change.

## Stage 2: energy coverage and larger populations

The 2026-11-12 replacement study at revision
`2506344f44dadb0273017f99c541fc823fe74406` uses
{download}`protocol-stage2.json <../../../checks/straggling_defaults/protocol-stage2.json>`:
5/30/100/300 keV, slabs at 0.05/0.3/0.6 of the corrected-SBETHE range,
and three independent seeds × 8192 primaries per on/off arm. The 5 keV
beam tracks down to 1 keV; other energies track down to 5 keV. Physics,
FP64, cutoffs and the absence of a numerical energy cap remain pinned.
This is a new sensitivity dataset; the deleted original stage-2 evidence
was not recovered or pooled.

All 120 groups (360 records, 2,949,120 measured primaries) are complete.
Raw-file and job manifests pass; the largest primary energy residual is
$3.53\times10^{-12}$ keV. The
{download}`compressed report <../check-records/straggling-defaults-stage2.json.gz>`
retains group moments, conditional variances/quantiles, pointwise intervals,
table identities, per-device assignment and separate overlap diagnostics.

NVIDIA RTX 5080 jobs cover silicon, HOPG and MoS₂; AMD Radeon 8060S jobs
cover WSe₂ and sapphire. Every on/off pair stays on one device. The common
`core="cuda"` record names the shared CuPy kernel implementation, not the
GPU vendor; the report's device map and raw `backend.txt`/allocation files
identify the ROCm runs. Do not compare materials' timing ratios as a
cross-device benchmark.

For the 48 shell-material comparisons, all transmission and backscatter
intervals lie within the predeclared ±2-percentage-point diagnostic margin.
Shell conditional transmitted-energy comparisons are inside their ±2% margin
in 39 cases and unresolved in nine; none is outside. Zero variance and small
transmitted populations remain unresolved by construction. These margins
bound the sampled on/off sensitivity at these settings, not model bias,
variance accuracy, rare tails or numerical convergence.

Sapphire's continuous/Urban control remains materially different. At 0.6 range,
its conditional transmitted-energy mean increases beyond the 2% diagnostic
margin at every energy:

| Beam energy (keV) | On − off mean (keV) | Pointwise approximate 95% interval (keV) |
| ---: | ---: | ---: |
| 5 | +0.467 | [+0.445, +0.489] |
| 30 | +2.018 | [+1.892, +2.143] |
| 100 | +5.716 | [+5.271, +6.162] |
| 300 | +15.299 | [+14.037, +16.562] |

The other sapphire conditional-energy comparisons include seven inside-margin
and one unresolved result. No experiment or independent transport code is used
here, so these differences do not establish improved accuracy.

### Cross-device overlap and preservation

A separate silicon 300 keV AMD run repeats all six NVIDIA arms with the same
seeds and table digests (18 records, 147,456 primaries). It is excluded from
the baseline counts and cannot be pooled as independent replicas. Of 180
per-replica score-field comparisons, 162 meet `rtol=1e-12, atol=1e-10`.
The remaining comparisons concern final energy and path length: maximum
absolute differences are $7.11\times10^{-10}$ keV and
$1.81\times10^{-5}$ Å, respectively. This records small arithmetic differences
in one overlap scene; it does not certify general CPU/GPU or cross-device
parity. AMD logs also report SharedSignalPool resource-leak warnings; complete
samples and checksums do not establish that runtime cleanup is correct.

The frozen raw archive is `straggling-642-stage2-20261112.tar.gz`, SHA-256
`046cb89d235f8abebb5bc30b019f51b051fac5eb1612c9aad7d01712c62af496`.
It includes source snapshots, protocol, scores, device/allocation records,
runtime scripts and job logs. Keep a copy outside disposable study checkouts.
NVIDIA baseline jobs are 12022 and 12024–12034; AMD baseline jobs are
12045–12052, with overlap 12043. Job 12021 failed before simulation and is
retained as failed-start provenance only.

The broader rollout gates remain open: layered/grooved coverage, active
step/cutoff convergence, detector/line observables, independently matched
accuracy references, rare-tail qualification, applied-loss anchors and
held-out equal-precision performance including setup and GPU peak memory.
No default, model marker or physics-ledger status changes follow from stage 2.
