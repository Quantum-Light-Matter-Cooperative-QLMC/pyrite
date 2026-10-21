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

## Device callbacks

The built-in density observable reduces the ensemble directly on CUDA.
Custom observables and jump hooks must set `cuda_compatible = True`, accept
CuPy arrays and return CuPy arrays (`None` may absorb a state in a hook).
Unmarked callbacks fail before CUDA allocation. There is no implicit host
wavefunction transfer for callbacks.

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
