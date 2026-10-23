# CUDA axial MCWF

`AxialMCWF` accepts `backend="cpu"` (the default) or `backend="cuda"`.
CUDA executes both split-step propagation and thermal-diffuse/electronic
jump sampling and application. Wavefunctions, jump kernels, Fourier
coefficients, rejection proposals and recorded ensemble sums stay on the
device. Python receives jump indices and scalar control values; only the
final ensemble arrays return to the host. CUDA requests fail explicitly when
CUDA is unavailable. CPU execution does not import CuPy.

Install the `nvidia` extra in the GPU environment. Use the
[remote workflow](running-on-a-cluster.md) for GPU calculations and expensive
convergence studies. The [axial diagnostic guide](axial-diagnostics.md)
describes physical conventions and experimental limitations; choosing CUDA
does not change the numerical sink into physical escape.

## Solver selection and admission

```python
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF

solver = AxialMCWF(dissipator, kinetic_energy_eV, potential_eV, backend="cuda")
result = solver.run(
    entry,
    depths_ang=[0, 500, 1000],
    n_electrons=128,
    seed=17,
    step_ang=10,
    batch_size=128,
    memory_budget_bytes=512 * 1024**2,
)
```

The dissipator, potential and entry state are prepared as for CPU execution.
The result contains NumPy arrays. Arithmetic uses real float64 and complex128
on both backends, independently of the native transport's working precision.
`result.jump_history` maps `tds`, `sink` and `electronic` to cumulative raw
ensemble jump counts at each recorded depth. Divide by `n_electrons` for
counts per incident electron. These counts are not probabilities: an electron
can have several jumps. `sink_depths_ang` and `surviving` retain their existing
meanings. Hook absorptions appear separately in the final `jumps` dictionary.

`batch_size` is an upper limit; the default is 128. CUDA admission uses the
existing `PYRITE_MC_RESOURCE_POLICY`, half of currently free device memory,
and the optional `memory_budget_bytes`, taking the smallest limit. Immutable
jump data, FFT scratch and recorded reductions are reserved before admitting
wavefunction batches. A budget that cannot fit one electron fails before
uploading the jump data. Admission does not change the physical model.

## Random streams and reproducibility

CPU retains its NumPy Philox streams and draw order. CUDA uses counter-addressed
Philox4x32-10, using the complete existing 64-bit
`mcwf_stream_key(seed, electron)` as the key. The scalar draw index addresses
the counter; no per-device-thread random state is allocated. The backend/algorithm is part of the
execution provenance; CUDA and CPU do **not** use matched random draws.
Compare their ensemble statistics with declared Monte Carlo uncertainty,
not individual trajectories.

Within each backend, fixed seed and electron identity determine each stream.
Changing admitted batch size or adding recorded depths does not draw random
numbers or alter trajectories. Summation order can change ensemble reductions
at float64 roundoff. Sink-depth array ordering follows batch traversal; compare
its multiset when batch sizes differ. A new `run` starts electron IDs at zero;
separate calls are not a resumable trajectory-chunk interface.

Small CUDA batches check waiting-time thresholds in blocks of up to eight
steps. If a threshold crosses inside a block, propagation resumes from the
first crossing before any jump is drawn. Speculative steps consume no random
numbers. The block shrinks for larger electron batches to limit replayed FFTs.
Each step uses an unscaled inverse FFT with its `1/n` folded into the
potential phase, and one fused kernel for the waiting-time norm. These change
states and norms only at float64 roundoff: on the benchmark workloads jump
counts are identical to the unfused path and densities agree within `1e-13`.

## Device callbacks

The built-in density observable reduces the ensemble directly on CUDA.
Custom observables and jump hooks must set `cuda_compatible = True`, accept
CuPy arrays and return CuPy arrays (`None` may absorb a state in a hook).
Unmarked callbacks fail before CUDA allocation. There is no implicit host
wavefunction transfer for callbacks.

An observable may implement `sum_batch(states, tilts)` to return the **sum**
over normalized live states, with shapes `(electrons, nx, ny)` and
`(electrons, 2)`. The result has the scalar callback's output shape; incident
normalization remains the solver's responsibility. The diagnostic recorder
uses batched reductions for metrics, level density and reference coherence.
Its internal `prepare_batch`/`flush` protocol also allows deferred spectral
projections to fill per-depth accumulators before results are finalized.

Custom observables can declare `recording_bytes` (one copy of all recorded outputs) and
`scratch_bytes` (peak temporary work); CUDA admission reserves these in
addition to its density and propagation allowance, reserving three recording
copies for sums, normalization and stacking. Sampled mixture entries also
count toward immutable storage. Callbacks must keep their
allocations within those declarations. Diagnostic reducers declare these
budgets for their ensemble sketches. Observation callbacks must not mutate
input state or consume the trajectory RNG; hooks use their own independent
random domains. Tilts may contain device scalars after an electronic kick.

## Hardware verification

{source}`checks/axial_mcwf_cuda.py` measures identical CPU/CUDA silicon
workloads with explicit complex128 precision, seed, grid, depth and step,
one cold run and three warm repeats per backend. It includes a small case
to expose dispatch overhead, records setup and result transfer costs, and
reports device memory-pool use and host peak RSS. Its predeclared performance
target is a twofold warm solver speedup on the 48²-grid, 128-electron,
100-step workload. A target is not a measured speedup or an accuracy claim.

Stage through `pyrite remote sync`, then run the checker and
{source}`tests/montecarlo/test_channeling_axial_cuda.py` in a bounded SLURM GPU
allocation. The checker requires real CUDA hardware and never silently skips
or falls back. The focused tests compare deterministic propagation and jump
kernels, check device callback types, verify batch/recording independence and
compare stochastic populations and survival with a predeclared uncertainty
bound. Extend convergence evidence before interpreting a new physical case.

{source}`checks/axial_recording_performance.py` measures recording off,
eigenstate-only and combined-recording workloads. It records one cold run,
individual warm repeats, GPU utilization, memory, configuration and source
hashes, and saves paired observable arrays for before/after comparisons.
Run identical arguments in separate baseline and candidate checkouts inside
bounded GPU allocations. Profile separate invocations with Nsight Systems;
profiling timings should not be mixed with unprofiled repeats.

## Measured workload and limitations

On an RTX 5080, CuPy 14.2.0/CUDA runtime 13.2, Python 3.14.6 and eight
allocated CPU threads, the fixed silicon [001], 1 MeV, 48²-grid,
128-electron, 100-step case met the predeclared twofold **warm solver-call**
target. The solver call includes immutable-data upload, propagation, both
jump mechanisms, density reduction and the final result transfer.

| Workload | CPU warm median (range) | CUDA warm median (range) | CPU/CUDA |
| --- | --- | --- | --- |
| 48², 128 electrons, 1000 Å, 10 Å step | 409.85 ms (407.00–410.20) | 195.86 ms (195.01–198.69) | 2.093 |
| 12², 8 electrons, 100 Å, 10 Å step | 0.680 ms (0.580–0.708) | 6.131 ms (6.127–6.523) | 0.111 |

The small case is approximately nine times slower on CUDA; select CPU for
small diagnostics. The first representative solver call was 421.29 ms on CPU
and 363.20 ms on CUDA. These are first-call measurements with warm on-disk
compiler/library caches, not cold-driver startup claims. Common CPU
profile/dissipator/potential construction took 106.13 ms; adding construction
to the warm calls gives approximately 1.71 times complete setup-plus-solve
speedup. An initial direct device port missed the target (0.668 times), as did
intermediate candidates; measured TDS shell-envelope, categorical sampling and
immutable jump-data overhead drove the optimizations.

Final output transfer was 0.039 ms for the representative density array;
a same-sized synthetic upload took 0.082 ms. Device-pool reservation reached
40.29 MiB. This is the pool footprint, excluding allocations owned directly
by CUDA libraries; it is not a measured whole-device peak. Process peak RSS
was 1.21 GiB including setup and imported libraries. The raw
{source}`docs/validation/check-records/axial-cuda-performance.json` preserves
all sampled timings, initial target misses, settings, source hashes and the
ten passing hardware checks. These measurements establish this bounded
workload only, not a general CUDA crossover size or new physical convergence.

## Heavier workloads and single-precision tolerance

SLURM job 1659 reran the hardware checks on the same RTX 5080 (CuPy 14.2.0,
CUDA runtime 13.2, eight CPU threads) at branch commit `a2274fb0` (axial sources unchanged in its rebase `a54e5564`). All 33
tests in the axial CUDA, CPU-zone CUDA continuation, axial flux and
energy-MCWF refusal modules passed, with none skipped. The default benchmark
reproduced the recorded target: 413.0 ms CPU and 197.6 ms CUDA warm medians,
a 2.090 times speedup.

`checks/axial_mcwf_cuda.py --workload` then ran silicon [001] at 5 MeV over
1 µm (1,000 steps of 10 Å). CPU runs were capped at 64 electrons, so the
comparison is warm throughput per electron-step, not matched results:

| Cells, grid | CUDA electrons | CPU (electron-steps/s) | CUDA (electron-steps/s) | CUDA/CPU |
| --- | ---: | ---: | ---: | ---: |
| 1×1, 48² | 512 | 36,672 | 126,329 | 3.44 |
| 2×2, 64² | 512 | 29,626 | 112,742 | 3.81 |
| 3×3, 96² | 512 | 13,330 | 88,615 | 6.65 |
| 4×4, 128² | 1,024 | 7,122 | 65,776 | 9.24 |

The advantage grows with supercell size. CPU and CUDA use independent random
algorithms, so these runs make no accuracy comparison. The largest case
reserved 598 MiB of CuPy pool memory.

Production propagation stays complex128.
{source}`checks/axial_split_step_precision_cuda.py` measures what
complex64 would cost. It applies the same deterministic no-jump Strang step,
with absorptive loss, to the same batch of tilted Gaussian entries in both
precisions. A CPU complex128 run of one entry agrees with the device
reference to $6\times10^{-13}$–$3\times10^{-12}$ relative norm after
$10^4$ steps. The complex64 error grows roughly linearly with step count:

| Steps (10 Å) | Surviving norm | Relative norm error | Infidelity | Density L1 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.35–0.36 | $1.3$–$3.4\times10^{-5}$ | $\le1.2\times10^{-10}$ | $\le1.1\times10^{-5}$ |
| 1,000 | $\approx10^{-3}$ | $1.3$–$3.2\times10^{-4}$ | $\le5.5\times10^{-9}$ | $\le8.2\times10^{-5}$ |
| 10,000 | $\approx10^{-22}$ | $2.2$–$4.2\times10^{-3}$ | $\le8.4\times10^{-7}$ | $\le1.0\times10^{-3}$ |

Ranges cover all four grids (48² to 128²) and are batch maxima. An MCWF jump
resets the norm, and a no-jump span ends when the norm falls below a uniform
threshold, typically within about a hundred steps here. Over such spans the
complex64 norm error shifts a jump by far less than one step. Complex64 split
steps ran 1.70, 3.10, 5.16 and 7.06 times faster than complex128 for 48²,
64², 96² and 128² grids. This measures the no-jump kernel only; jump
selection and dissipator kernels remain float64.
Raw outputs: {source}`docs/validation/check-records/axial-cuda-heavy-precision.json`.

### Paired complex64 MCWF ensembles

{source}`checks/axial_mcwf_precision_cuda.py` runs the production CUDA batch
twice per seed chunk with identical Philox streams. One arm propagates in
complex128. The other stores the spectrum and phases in complex64, so its
split step, norm decay and jump clock are single precision; jump states are
promoted to complex128 for the float64 jump kernels. The predeclared
tolerance was $|\overline{\Delta}|+2\,\mathrm{SE}(\Delta)\le0.1\,
\mathrm{SE}_{128}$ for every observable and recorded depth. The observables
are survival, cumulative TDS, sink and electronic jumps per electron, and the
mean transverse potential energy per incident electron. SE₁₂₈ is the chunk
standard error of the complex128 ensemble mean.

SLURM job 1662 (RTX 5080, commit `fe6993fc`, exit 0, 5 min 26 s) ran silicon
[001] at 5 MeV with 10 Å steps and chunks of 64 electrons:

| Grid, depth | Electrons | Identical jump sequences | Tolerance | Warm time ratio c128/c64 |
| --- | ---: | ---: | --- | ---: |
| 2×2 cells, 64², 2 µm | 2,048 | 1,873 (91.5%) | fails from 0.21 µm | 1.06 |
| 4×4 cells, 128², 2 µm | 2,048 | 1,870 (91.3%) | fails from 0.45 µm | 1.75 |
| 2×2 cells, 64², 10 µm | 1,024 | 618 (60.4%) | fails from 1 µm | 1.07 |

**Complex64 fails the tolerance.** The failure comes from trajectory
divergence, not a resolved bias. Single-precision rounding eventually moves a
jump step or a rejection-sampling outcome. After that, the paired electron
follows an independent history, so SE(Δ) grows toward the unpaired value.
The earliest divergence came at 38 steps, the median near 1,100 steps over
2 µm. Of the 43 depth/observable pairs whose paired differences vary,
only the 0.1 µm mean potential energy on 64² shows a resolved shift,
$\approx8\times10^{-6}$ eV against a −8.7 eV mean, consistent with rounding.
No other $|\overline{\Delta}|/\mathrm{SE}(\Delta)$ exceeds 2. With dispatch and
float64 jumps still dominating, the whole-solver gain is modest. Production
axial MCWF therefore remains complex128. A statistically equivalent unpaired
ensemble comparison could still be designed, but this check gives no reason
to pursue the 1.06–1.75 times gain.
Raw outputs: {source}`docs/validation/check-records/axial-mcwf-complex64-1662.json`.
