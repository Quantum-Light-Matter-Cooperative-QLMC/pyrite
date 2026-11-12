# Planar density-matrix solver benchmarks

[Machine-readable measurements](data/density-matrix-2026-11-11.json). Harness:
`checks/density_propagate_benchmark.py` (#1125). These figures are performance
evidence for `DensityMatrixSolver.propagate`, not physics validation.

## Protocol

Si(110) at 16.92 MeV with 2.2 mrad divergence and a TDS + electronic kernel.
Settings: `n_cells = 16`, 300 eV margin, 2 µm at `dz = 250 Å` (80 steps), no
stopping. Each H runs one warm-up solve and three timed repeats in one
process; kernel construction is excluded. The table lists medians, with the
repeat range in the JSON. Host: i7-13620H laptop (16 logical CPUs), shared
with other work. Baseline is `main` at `39bc51b04`; the candidate is the
`issue-1125-density-fft` branch on the same base. Both use NumPy and SciPy
with OpenBLAS 0.3.34.

## Result

| H | N | `n_grid` | main ms/step | branch ms/step | speedup |
|---:|---:|---:|---:|---:|---:|
| 24 | 49 | 74 | 8.99 | 4.77 | 1.9× |
| 32 | 65 | 98 | 14.92 | 6.53 | 2.3× |
| 40 | 81 | 122 | 24.50 | 9.32 | 2.6× |
| 56 | 113 | 170 | 56.81 | 19.06 | 3.0× |

`OPENBLAS_NUM_THREADS=4`. With all 16 BLAS threads, both arms are slower and
noisy at these sizes: main takes 13–123 ms/step and the branch 6.7–52 ms/step,
with repeat ranges up to ±40%. Batched matmuls this small do not scale past a
few BLAS threads, so set `OPENBLAS_NUM_THREADS` to about 4 for these solves.

End to end, the Kephart anchor (`checks/kephart_occupation.py --cases si110_17
si100_17 --margins 300 --orders 56 --cells 16 --dz 62.5`, 120 µm, 19,200 steps)
took 511 s and 364 s locally at 4 threads. The stored remote records took
2261 s and 1987 s. Occupation lengths agree with
`docs/validation/check-records/kephart_occupation_runs/` to 1.4e-5 relative,
the same in every case. That offset dates from upstream kernel revisions after
the recorded `2818b327`; on one revision, main and branch outputs differ by at
most 2e-15.

## What changed and why

At H = 24, cProfile put most of the solve in `jump()`, in batched dense
`A B A^dagger` products. Two candidate fixes were measured.

- **FFT for the plane-wave ↔ grid map.** Rejected. The `_waves` map is a
  zero-padded DFT plus a Bloch phase. An FFT form agrees with it to 1.5e-14,
  but at these sizes the batched 2-D transform of `(16, n, n)` arrays is no
  faster than the BLAS product. With `numpy.fft`, it was 2.1 ms vs 1.0 ms at
  H = 24 and 7.7 ms vs 1.7 ms at H = 40. `n_grid = 3H + 2` also has large
  prime factors (122 = 2·61). Even with `scipy.fft` workers and a fast length,
  it only roughly breaks even below H = 56.
- **Evolve only subspace slots.** Adopted. The cutoff is fixed at barrier
  + 300 eV, so the evolving subspace has about 33 slots per block whatever H
  is. The no-jump propagator is zero outside it. The step loop therefore holds
  `rho` on those slots and maps them straight to the cell grid, using
  `W V[:, active]`, which has shape `n_grid × 33`. Jump diagonals of the other
  states are still computed exactly for free-flux booking. The band trace uses
  a precomputed `W W^dagger`. Loss is `V^dagger C V` on the slots. This is
  exact: tests compare it with the full-slot path and the dense map.
- **`scipy.fft` in place for the Bloch convolution.** Adopted. On one worker
  it takes 1.2 ms against 3.1 ms for `numpy.fft` at `n_grid = 122`, and more
  workers are available through `scipy.fft.set_workers`. The kernel scale and
  the `1/n_grid^2` normalisation move onto the small outputs.

At H = 56 the restricted jump takes 8.7 ms. Of that, the free-flux diagonal of
the 80 out-of-subspace slots costs 2.3 ms, the two grid maps 2.2 ms and the
Bloch convolution about 1.3 ms; the rest is elementwise work.
