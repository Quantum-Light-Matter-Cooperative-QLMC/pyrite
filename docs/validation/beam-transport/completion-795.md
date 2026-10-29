# Issue #795 — batched hard-photon completion

`complete_hard_radiative_events` still samples *exactly two* uniforms in the
canonical positive-photon row order using the request's original
`seed ^ 0xD6023FEA38B57C29` generator. The positive-photon row indices and
ordered uniforms are kept together when photons are subsequently grouped by
atomic number. The implementation makes one backend BremsLib **row** DDCS
interpolation per element/chunk over `(photon, theta)`, followed by one
backend-to-host transfer. A bounded `128 photon / 32768 angle-row` workset
prevents an all-photons cross-product allocation. It does **not** retain
DDCS stages globally or change the soft/hard energy partition or angular law.

The device interpolator uses `evaluate_bremslib_rows` with photon-major
repeated incident/photon energies and tiled angles; using the regular
`evaluate_bremslib` with a photon-energy vector would instead allocate a
quadratic `(photons * angles, photons)` grid. The host CDF retains the
original linear-in-theta cell mass, side-`right` boundary convention and
52 successive bisections. Direction and momentum algebra is batched except
for the tiny three-component transverse norms: the pre-batch
`np.linalg.norm(vector)` uses a different reduction path from
`np.linalg.norm(vectors, axis=1)` on some builds and can differ by 1 ULP.
The original scalar sampler remains as the executable reference; it is not
modified by the optimization.

## Lab verification on `qlmc`

Use a fresh issue worktree, synchronize it to an isolated `qlmc` checkout
using the maintained `uv run pyrite remote sync`, and install the locked CUDA
environment there (`uv sync --extra nvidia --locked`). **Do not run two
variants in the same process**. From the checkout on the remote machine:

```bash
sbatch checks/equal_precision/run_completion_795.sbatch
```

The job is pinned to the lab GPU node used by the existing equal-precision
SLURM scripts. Override `PYRITE_795_BASE_REF` if the pinned pre-#795 commit
is not in the checkout; that ref **must** contain the original scalar
completion. `PYRITE_795_OUT` changes the evidence directory (which the
script refuses to overwrite). `PYRITE_FP64=1` pins the equal-precision CUDA
path. Default are seeds `79501–79503`, anchors Si 300 keV/100 µm and W
300 keV/10 µm, 5000 and 20000 histories, chunks of up to 5000 histories,
and an equal complete warmup for every run. Variant order alternates by seed.
The run preserves `git HEAD`, source hashes, job allocation, GPU properties,
raw per-primary score arrays, emitted photon-row vectors, warmup and measured
times, and process RSS and CuPy allocator high water. **Allocator high water
is not a direct hardware VRAM peak**. Completion time is explicitly bounded by
synchronization. Total request time includes scoring/reduction but not warmup.

`benchmark_completion_795.py compare` demands **byte-identical** photon row
indices, directions, target recoil momenta, and per-primary source scores;
it terminates on discrepancies, reporting their count and maximum absolute
value for investigation. No tolerance is silently loosened. Its `summarize`
mode records the medians over warm independent processes, photon counts,
completion microseconds/photon, request time, and memory maxima. Keep the
per-run files and failed runs, rather than presenting only the medians. Do
not treat isolated completion-only gains as whole-request gains. If a CPU
baseline is needed, run the benchmark's `run` and `compare` commands under
`PYRITE_MC_BACKEND=cpu` on the CPU core with a distinct evidence directory.

## Lab results

Job 2668 compared baseline `42931cbb2b5fe71a2a8afd3fd00a69033a92c9ad`
with candidate `7df6092b13b9899a44727285b73df627b02a773f` on the lab GPU,
using the protocol above. Three independent warm repetitions used seeds
79501–79503. All 12 paired requests matched bitwise for photon row indices,
directions, target momenta, and per-primary source scores. Focused CPU tests
(37), CUDA tests (20), and the complexity check passed. The other CPU job
shared the node at submission; its possible timing effect is not quantified.

| Material, histories | Median photons | Completion baseline → candidate (s) | Completion µs/photon baseline → candidate | Request baseline → candidate (s) | Maximum candidate RSS / CuPy pool (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Si, 5000 | 89 | 0.270 → 0.102 | 3069 → 1139 | 2.778 → 2.591 | 1654 / 6212 |
| Si, 20000 | 325 | 1.017 → 0.414 | 3140 → 1274 | 10.950 → 10.377 | 1659 / 7375 |
| W, 5000 | 383 | 0.943 → 0.231 | 2461 → 598 | 9.361 → 8.553 | 2590 / 12402 |
| W, 20000 | 1513 | 3.853 → 0.943 | 2523 → 623 | 37.242 → 33.982 | 2590 / 19594 |

RSS and pool values are maxima across repetitions; the CuPy allocator high
water is not a direct hardware VRAM peak. Per-photon costs are medians of
individual request ratios. The [machine-readable summary](../check-records/equal-precision/completion-795-summary.json)
includes baseline memory and unrounded timings. Full individual results and
parity records remain in the isolated job evidence directory. The physical
`bremslib-radiative-partition` ledger status is unchanged; these results do
not establish a physics accuracy improvement or completed #611 streaming.
