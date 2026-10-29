# Radiative moment interpolation

Validation: `radiative-moment-interpolation`.

## Current default and remaining qualification

At the user's explicit request, coupled CPU/CUDA transport now defaults to
prepared float64 moment tables at the frozen $2\times10^{-4}$ relative budget.
This is a default-selection decision, **not** completed ensemble qualification
or a change to any physical validation status. Exact event-row totals, element
selection, conditional photon sampling, photon debit and RNG addressing remain
unchanged. The keyword-only advanced transport argument
`radiative_moments="exact"` retains the reference quadrature.

New coupled cases serialize `radiative_moment_model` with the versioned marker
`radiative-moments-v1/positive-linear-log-energy-v1/rtol2e-4`. Result row metadata
also retain the actual packed moment-buffer SHA-256. A legacy case lacking the
marker replays exact quadrature; it is not silently reinterpreted as a table
case. Prepared buffers belong to one transport pass and are uploaded once by
its batching owner; no process-global host/device table cache is introduced.
Preparation and out-of-coverage failures remain explicit. Production table
construction retains the exact reference packer's existing relative
$10^{-12}$ source-endpoint allowance, needed when incident-grid refinement
moves an intended endpoint by a floating-point step. At these admitted source
edges the reference uses its endpoint value; no additional physical energy
range is admitted. The standalone component preparation API remains strict
by default. All interpolation queries, including production queries, still
reject energies outside their explicitly prepared interval.

Held-out CPU job **2669**, frozen at `8db53f2d` before the #795/#799 rebase,
was cancelled at the user's request to release its 24 GiB reservation for
other GPU tests. The [interruption record](../check-records/equal-precision/moment-heldout-interrupted-788.json)
lists completed requests. Partial records remain in its isolated evidence directory;
missing fixed-seed groups remain unresolved. This run did not include #795.
Issue #788 stays open for complete held-out ensemble qualification, remaining
hard-count/continuous-observable limits, wider CPU/CUDA transport comparisons
and matched timing of the combined production source. No unfinished check is
counted as passing. The job's six Si preflights passed within-process replay
and matched score/trajectory hashes across the two independent processes for
exact, table and refined policies.

## Independent derivation

This verifier did not implement this claim. Before reading implementation
bodies, it read the ledger row, function/class/module docstrings, frozen
`checks/equal_precision/moment-policy-788.json`, and the existing independent
[BremsLib partition](bremslib-radiative-partition.md) and
[dielectric suppression](dielectric-suppression.md) derivations. This claim
starts from those ledgered moment definitions; it does not independently
revalidate the original BremsLib data or the bulk-medium approximation.

Let $T>0$ be kinetic energy, $k_c>0$ the photon cutoff,
$K=\min(k_c,T)$, and $\chi_a(T,k/T)$ the interpolated elemental scaled SDCS.
For fixed layer plasma energy $p$, define

$$
F(T,k)=\frac{k^2}{k^2+[(1+T/m_ec^2)p]^2},\qquad C_a=10^{-27}Z_a^2.
$$

For zero plasma energy use $F=1$, including the zero-photon endpoint in
first-moment integrals. The per-atom moments are

$$
S_{s,a}=C_a\int_0^K\chi_a F\,dk,\qquad
R_{h,a}=C_a\int_K^T\frac{\chi_a F}{k}\,dk,\qquad
S_{h,a}=C_a\int_K^T\chi_a F\,dk.
$$

The cross-section conversion gives $R_{h,a}$ in $\mathrm{cm^2}$ and both
first moments in $\mathrm{eV\,cm^2}$. For elemental number densities
$n_a$ in $\mathrm{cm^{-3}}$, layer moments are

$$
\mathbf M(T)=10^{-8}\sum_a n_a(S_{s,a},R_{h,a},S_{h,a}).
$$

Their units are respectively $\mathrm{eV/\mathring A}$,
$\mathrm{\mathring A^{-1}}$, and $\mathrm{eV/\mathring A}$.
Equivalently a density supplied in $\mathrm{\mathring A^{-3}}$ must first
be multiplied by $10^{24}$. At $T\le k_c$ both hard moments vanish;
the soft moment is the full first moment. At all covered energies
$S_s+S_h=S_{\mathrm{total}}$ for the exact reference.

For increasing nodes $T_i$, set

$$
w=\frac{\ln(T/T_i)}{\ln(T_{i+1}/T_i)},\qquad
\widehat{\mathbf M}(T)=(1-w)\mathbf M(T_i)+w\mathbf M(T_{i+1}).
$$

Within the closed node interval $0\le w\le1$: endpoints are reproduced,
nonnegative nodes imply nonnegative components, and zero moments require
no logarithm. Outside coverage there is no convexity guarantee and the
query must be refused. A cutoff node is needed when $k_c$ lies inside
coverage to preserve identically zero hard moments below the cutoff.
The interpolation is linear in moment values, not their logarithms.

The exact total first moment must be checked separately. Componentwise
first-moment errors each bounded by an absolute tolerance do not imply
their sum meets one absolute tolerance. The frozen qualification uses

$$
B_j(T)=a_j+r\lvert M_j(T)\rvert,\qquad
\lvert\widehat M_j-M_j\rvert\le B_j,
$$

with $r=2\times10^{-4}$ and
$(a_s,a_r,a_h)=(10^{-10},10^{-14},10^{-10})$ in the corresponding units.
Closure additionally requires

$$
\lvert\widehat S_s+\widehat S_h-S_{\mathrm{total}}\rvert
\le10^{-10}+r\lvert S_{\mathrm{total}}\rvert.
$$

Adaptive quarter/midpoint probes use one quarter of these budgets;
qualification uses a separate offset logarithmic grid. These are declared
numerical approximation budgets, not source-data uncertainty bounds.
Finite probes provide evidence on the tested grid, not an interval-wide
error theorem. Identity must include table contents, composition, plasma,
cutoff, coverage, projectile, model versions and numerical policy so reuse
cannot silently change the physical reference or approximation.

## Cheap filters before implementation inspection

Units pass: density and length conversion above are required. Limits pass:
endpoint reproduction, zero hard moments below threshold, and a separate
closure budget follow directly. Signs and conventions pass: convex weights
preserve nonnegativity and the kinetic-energy Lorentz factor is
$1+T/m_ec^2$. The approximation alone gives no bound on errors in transport
histories, samplers, or CPU/CUDA execution.

## Source-to-code comparison

`_exact_layer` multiplies packed density in $\mathrm{cm^{-3}}$ by
$10^{-8}$ and sums all three elemental moments. The packer converts
input $\mathrm{\mathring A^{-3}}$ density by $10^{24}$ and computes one
shared plasma energy for all constituents of each layer. These match the
derived quantity and units. `interpolate_moments` computes the convex
combination above in float64. The supported prepared-object evaluator
checks finite physical energies before logarithms and returns physical
endpoint values directly; this prevents logarithmic rounding from admitting
an adjacent out-of-range floating-point energy.

`_refine` uses the declared quarter/midpoint probes and one-quarter budgets.
`_validate` evaluates the fixed offset grid independently of those adaptive
probe positions. Its reference calls the same scalar exact evaluator used
to construct nodes, so this is independent sampling, not an independent
physical model. The tests additionally compare against
`build_radiative_partition`, a separate host moment implementation using the
same tabulated cross sections and physics. That is useful algorithmic
parity; it does not establish independent experimental agreement.

The first reviewed closure check incorrectly summed both absolute
first-moment budgets. The implementer corrected it during verification to
use one `policy.atol[0]`, matching the frozen closure budget. No factor of
two remains in the reviewed denominator.

Identity includes the interpolation-policy identifier, tolerances, maximum
nodes, held-out count, projectile label, cutoff, coverage, composition,
plasma, source keys/digests and BremsLib model marker. The digest additionally
hashes every actual packed quadrature input with shape/dtype and all
prepared grids/logs/moments. Custom data reusing source labels cannot alias
the same digest. Projectile is an explicit identity label: the caller must
supply already scaled positron tables; labeling electron tables as positron
does not perform or validate positron scaling.

The CUDA diagnostic evaluator implements the same convex formula and checks
physical endpoints before taking logarithms. Unsupported energies yield
explicit NaNs rather than a host exception; callers must reject them.
This source review is not hardware execution evidence. The uploaded object
retains the host digest and records its owning CUDA device.

## Numerical evidence and limits

Focused host anchors and documentation checks passed together: 17 tests,
8.64 s. The tests cover compound layers, both suppression settings, two
cutoffs, zero/tiny rates, endpoint coverage refusal, and identity changes.

An additional verifier-run calculation replaced the prepared elemental
scaled SDCS by a constant 10 mb. It did not use either partition helper to
calculate reference moments. For constant $\chi$, vacuum references are
proportional to $(k_c,\ln(T/k_c),T-k_c)$. With dielectric scale $a$, use

$$
G(k)=k-a\arctan(k/a),\qquad
(S_s,R_h,S_h)\propto
\left(G(k_c),\frac12\ln\frac{T^2+a^2}{k_c^2+a^2},G(T)-G(k_c)\right).
$$

The proportionality factor includes 10 mb, atomic number squared, cross-
section conversion and number density. Plasma energy was calculated from
SI constants as $\hbar\sqrt{n_e e^2/(\epsilon_0m_e)}$, with
$n_e=14\times0.05\times10^{30}\ \mathrm{m^{-3}}$. For Si density
$0.05\ \mathrm{\mathring A^{-3}}$, a 1 keV cutoff and kinetic energies
10, 13.721, 37.373, 113.171 and 200 keV, all ten vacuum/suppressed cases
passed. The largest component error was 0.114735 of its mixed budget;
physical endpoints matched the analytic moments at relative tolerance
$2\times10^{-13}$. This is a fresh analytic numerical check, not a retained
regression artifact or a released-table transport benchmark.

No CPU/CUDA runtime parity, production integration, ensemble equivalence,
transport speedup, or complete-request equal-precision speedup is certified
by these host component checks.

The first reviewed preparation path inherited a relative $10^{-12}$
table-endpoint admission tolerance from the existing packer. A verifier
probe confirmed that both adjacent floating-point energies outside the
source table were accepted. The implementer added strict physical table
bounds in this diagnostic preparation path and regression checks for both
adjacent outside endpoints. This resolves the stated no-clamping coverage
contract without changing the existing packer contract.

## Released-table component qualification

SLURM job 2637 ran the committed `592aa34f` prototype on 2026-10-29:
Ryzen 9 9950X3D, RTX 5080 (driver 610.47), Python 3.14.6,
NumPy 2.5.4 and CuPy 14.2.0. The allocation requested one CPU and 8 GiB;
the scheduler assigned two logical CPUs. BLAS/OpenMP/Numba threads were
limited to one. Job 2636 produced no measurements because the environment
lacked CuPy; the declared NVIDIA extra was restored before resubmission.

The reproduction command is
`uv run --no-sync python checks/equal_precision/qualify_moments.py OUTPUT.json`
inside the environment in `checks/equal_precision/moments-788.sbatch`.
The [complete numerical record](../check-records/equal-precision/radiative-moments-788.json)
retains source/table/policy digests and every timing repeat; published metadata
omits operator paths and hostnames.

All twelve component cases passed: Si, W and a mixed Si/W layer over
10 keV–100 MeV, photon cutoffs 1 and 10 keV, and relative budgets
$2\times10^{-4}$ and $10^{-4}$. Each preparation used 8191 fixed held-out
energies; a separate 65,536-energy grid also passed. The largest component
error was 0.249924 of its mixed budget. Held-out soft/hard closure passed.
Identical prepared float64 tables agreed between CPU and CUDA within
$10^{-24}+2\times10^{-12}|M|$; exact CPU/CUDA soft/rate probes separately
agreed within $10^{-20}+2\times10^{-10}|M|$. The two hardware regressions
passed, including repeated-call determinism and unsupported-energy NaNs.
These are component/runtime comparisons, not ensemble transport equivalence.

For 65,536 matched energies, five synchronized warm repeats at the default
budget gave the following medians (exact computes soft/rate, table computes
all three components):

| Cutoff | Layer | Exact GPU, ms | Table GPU, ms | Median ratio |
|---|---|---:|---:|---:|
| 1 keV | Si | 1.182 | 0.102 | 11.56 |
| 1 keV | W | 1.285 | 0.135 | 9.52 |
| 1 keV | Si/W | 2.306 | 0.103 | 22.34 |
| 10 keV | Si | 1.185 | 0.105 | 11.25 |
| 10 keV | W | 1.187 | 0.107 | 11.05 |
| 10 keV | Si/W | 2.220 | 0.114 | 19.42 |

All-three-layer preparation cost 0.205–0.227 s at the default budget;
upload cost 0.522–0.593 ms and table storage 93,240–120,480 bytes on each
of host and device. The tighter budget used 150,000–195,360 bytes and
0.216–0.236 s preparation. Build times can include remaining first-use
Numba specialization cost. Using the default medians, amortizing the entire
three-layer build against a single layer's savings requires about 93–211
such microbenchmark batches. This is a microbenchmark accounting exercise,
not a request-size threshold for transport.

Launch geometry was 128 threads/block with one query per thread. The exact
probe compiled to 92 registers/thread and zero local bytes/thread. No
profiler evidence establishes an exclusive FP64 or occupancy bottleneck.
There are no transported histories, batch/probe scheduling, useful/replayed
flight counts, or complete-request/equal-achieved-precision measurements in
this run. No transport speedup, baseline trajectory equivalence or wider
physical-model admission follows from these measurements.

### Diagnostic transport pilot (job 2644)

Job 2644 ran `checks/equal_precision/moment-transport-788.sbatch` from clean
source `fd265543` on the same GPU. An isolated kernel copy replaces exactly
the four frequent total-moment calls; the row-end event total, element
selection, conditional samplers, photon debit and RNG source stay exact.
Preflights at 256 histories passed for exact, table and refined variants on
both Si 300 keV/100 µm and W 300 keV/10 µm: in-process and separate-process
same-seed score hashes matched, the largest energy-ledger error was
$2.9\times10^{-12}$ keV and the largest photon-debit error was
$2.8\times10^{-14}$ keV.

The paired pilot used 5000 histories for each of seeds 78811–78813, with
99% Student-t intervals against the predeclared margins. Table and refined
moments changed continuous observables only at round-off scale: mean
photon-energy and exit-energy differences were below $10^{-9}$ keV per
primary, and path-length differences were about $10^{-5}$ Å. These
observables screen inside their margins. Hard-event count, transmission and
backscatter did not change for any history. The predeclared zero-variance
rule therefore reports them as `unresolved`, not passing. Median complete
warm request ratios (exact/table, all reductions included) were 1.44 (Si)
and 1.86 (W); refined gave 1.37 and 1.88. These are three-seed timings of
an isolated diagnostic kernel. They are not a production or
equal-achieved-precision speedup.
[Job record](../check-records/equal-precision/moment-transport-788.json).

Job 2647 (`moment-anchors-788.sbatch`, source `0c25658d`) compared exact
moments with default-budget tables on five wider anchors. These were
Si and W at 800 keV, Si at 100 keV/1 µm, W at 100 keV/100 µm and 75°
(backscatter-dominated), and Si at 300 keV with a 10 keV photon cutoff.
All replay and energy/photon-debit preflights passed. The pattern matched
job 2644. Continuous observables changed only at round-off scale and
screened inside their margins, except where no history changed. Hard-event
count, transmission and backscatter changed for no history and remain
`unresolved`. Median exact/table complete request ratios were 2.15
(Si 100 keV), 1.65 (Si 300 keV, 10 keV cutoff), 2.34 (Si 800 keV), 1.35
(W 100 keV, 75°) and 1.59 (W 800 keV). The same diagnostic-kernel caveats
apply.
[Job record](../check-records/equal-precision/moment-anchors-788.json).

Job 2650 (source `42e3d12c`) ran the same protocol on the compound matrix
case `sio2-800kev-tr0.3-0deg-compound`, a two-element Si/O layer table.
Replay and accounting preflights passed, with the largest energy-ledger
error at $6.5\times10^{-12}$ keV. Continuous observables again changed only at
round-off scale. Hard-event count, transmission and backscatter changed for
no history (`unresolved`). The median exact/table request ratio was 2.04.
[Job record](../check-records/equal-precision/moment-compound-788.json).
Opt-in production integration remains pending.

### CPU diagnostic transport pilot (job 2666)

Job 2666 ran `checks/equal_precision/moment-cpu-788.sbatch` on the remote
Ryzen 9 9950X3D, Python 3.14.6, with one requested CPU, 24 GiB and one
thread per numerical library. The sync record identifies a **dirty** checkout
at `c9a345e7`; recorded CPU core, wrapper, helper, exact sampler and policy
hashes match the committed diagnostic branch `7ea08549`. This establishes
those measured file identities, not whole-checkout equivalence. Job 2665
stopped before measurements because the host-metadata command could not find
`python` on PATH. Scheduler accounting storage was disabled.

The isolated CPU core replaces three frequent total-moment evaluations and
retains the fourth, row-end hard-event total. Conditional photon sampling,
actual photon debit and RNG addressing remain exact. Six 256-history
preflights passed within-process same-seed replay. Separate-process replay
was not tested. Across preflights and measured requests, maximum energy-ledger
and photon-debit errors were respectively $2.51\times10^{-12}$ keV and
$2.85\times10^{-14}$ keV.

Each scene/variant used three paired seeds, 2048 histories per seed
(6144 total), a 1 keV photon cutoff, 2048-history batches and no probe
histories. Exact, default-table and refined-table policies used the same
inputs and an independent same-workload warmup before each measurement.
All continuous checks screen inside the frozen 99% paired-pilot margins;
hard counts, transmission and backscatter changed for no history and remain
`unresolved`. Table refinement also preserves this pattern.

| CPU anchor | Exact median request, s | Table median request, s | Exact/table | Exact/refined |
|---|---:|---:|---:|---:|
| Si 300 keV, 100 µm | 33.208 | 2.990 | 11.11 | 11.12 |
| W 300 keV, 10 µm | 61.770 | 5.752 | 10.74 | 10.85 |

Request timings include reductions; preparation and warmup/JIT are separate.
Default-table builds in the measured runs took 0.049–0.051 s and used
9864 host bytes for Si and 9744 for W; CPU upload cost is zero. These
preparation observations are not a cold-cache benchmark.
The [complete CPU record](../check-records/equal-precision/moment-cpu-788.json)
retains every repeat, preparation cost, storage, source/table pins, process
peak RSS and accounting maxima. Recomputing the paired analysis from all
18 raw score archives reproduces the supplied analysis exactly. Published
metadata omit operator paths and hostnames.

These are diagnostic warm-request ratios. There is no production or
equal-achieved-precision speedup claim. Useful/replayed flight counters and
CPU profiling were not collected. The existing component CPU/CUDA comparison
does not qualify CPU/CUDA transport ensembles. Wider CPU anchors, held-out
ensemble qualification (including sparse discrete outcomes), separate-process
CPU replay and production identity integration remain open. The frozen pilot
rule is unchanged; unchanged outcomes are not promoted to passing.

## Verdict

- **Claim:** `radiative-moment-interpolation` —
  `montecarlo/transport/radiative_moments.py::prepare_radiative_moments`,
  `::interpolate_moments`, and the diagnostic device interpolant — existing
  ledgered BremsLib/dielectric moment definitions and convex interpolation
  in log kinetic energy.
- **Filters:** units pass; limits pass; signs/conventions pass. Reference
  transcriptions: no new external tabulated measurements. Tolerances:
  predeclared numerical approximation budgets. Hardware component evidence:
  job 2637 as recorded above; no external transport-code comparison.
- **Re-derivation:** matches — no remaining divergent factor, sign,
  exponent, unit, or interpolation convention after correction of the
  closure absolute budget and strict source-table coverage checks.
- **Verdict:** `rederived` for the interpolation equations and component
  scope. Device source agrees symbolically; job 2637 establishes the stated
  component runtime parity on the tested GPU and released-table grids.
- **Write-up:**
  `docs/validation/radiation-physics/radiative-moment-interpolation.md`.
- **Suggested ledger change:** set `rederived`; `anchored` is supported for
  the host synthetic component scope by the passing named regression module.
  Released-table component qualification passed; keep ensemble and production
  qualification explicitly pending.
  Only a human may set `signed-off`.

Strict admission exposed a separate fixture representation detail: staging
stored the synthetic nominal 10 keV endpoint as
$10.000000000000004\ \mathrm{keV}$. The test fixture now explicitly sets
its two intended endpoints to 10 and 200 keV; the implementation retains
strict comparison against the actual staged table interval. Released-table
diagnostics must derive coverage from the intersection of actual staged
elemental intervals. The implementer reports the corrected eight-test
component module passes in 8.56 s. The analytic constant-SDCS calculation
above is retained as a regression anchor.

Final focused rerun, explicitly including the four slow parametrizations:
`pyrite-dev test --slow tests/montecarlo/test_radiative_moments.py tests/dev/test_docs.py`:
19 passed in 7.17 s. This includes the retained
`test_constant_sdcs_against_analytic_moments` anchor for both suppression
settings. The author owns the complete docs build and rendered-math check.
