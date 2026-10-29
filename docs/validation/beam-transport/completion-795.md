# Issue #795 — batched hard-photon completion (measurement protocol)

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

## Validation status

The patch changes implementation and adds synthetic Si/W mixed-element,
zero-photon, single-photon, cutoff/endpoint, multi-chunk and validation tests.
Its performance and CUDA bitwise gates are **not complete** until the above
remote measurements and focused CPU/CUDA tests are attached to the issue.
The physical `bremslib-radiative-partition` ledger status is unchanged. This
is not evidence of a physics accuracy improvement or completed #611 streaming.
