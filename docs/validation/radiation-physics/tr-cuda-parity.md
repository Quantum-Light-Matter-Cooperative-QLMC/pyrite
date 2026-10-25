# TR/CTR CUDA parity evidence

This is a hardware test record for the existing planar-stack TR/CTR component,
not an independent physics derivation or a DTR device-validation claim.

On 2026-10-18, SLURM job **1592** ran
`tests/montecarlo/test_transition_radiation_cuda.py` on an NVIDIA GeForce
RTX 5080 (16303 MiB), driver 610.47, CuPy 14.2.0, Python 3.14.6.
The isolated checkout was `/home/user/pyrite-issue-103` on the configured
`cluster-head` host. Source was clean revision
`fc547a9e1dc96cc350b2c262b6e7c40ff99aa7b4`, with the test-fixture correction
in this change applied afterward. The tested CUDA module SHA-256 is
`067f7d94ad5e051ba56b061dc0a7549e5c9083952b07b8208d1cae3da7e5aed2`.
The synced source digest was
`a966a3c14a3913680dbe1d714662e5815e41c3f808537ce36553b67ae120bee0`.

**Result: 16 passed in 3.77 s; no skips.** Tests cover:

- NumPy/CuPy intensity and complex-field parity on an absorbing oblique stack;
- crossing, per-history and bunch modes, both emitting faces, fused kernel
  and chunked array routes, on identical transported bilayer rows;
- bit-identical kernel reruns and reduction stability under launch changes;
- default CuPy dispatch and host result return through `case_transition_spectrum`.

Spectral parity uses relative tolerance `1e-10` and an absolute floor of
`1e-12` times the host peak for near-cancelled entries. The identical launch
requires bitwise equality; regrouped reductions use `rtol=1e-12` and
`atol=1e-13` times the peak.

The first valid submission, job 1591, passed the 14 field/kernel tests; two
runner tests failed before scoring because their independent transport setup
required an unavailable SBETHE HOPG table. The corrected fixture reuses the
seeded, transported C/W bilayer already used by crossing parity. It tests the
actual scoring component on shared trajectories without downloading unrelated
stopping/elastic tables. This is not evidence for the complete default
SBETHE/ELSEPA run pipeline. Job 1590 was a setup failure using the nonexistent
`cuda` extra; the actual environment uses `nvidia`.

The allocation requested one GPU, four CPUs, 8 GiB and ten minutes. The live
SLURM default was 1500 MiB per CPU; this job used explicit memory. The job had
`PYRITE_TEST_BACKEND=cuda`, `PYRITE_MC_GPU_POOL_FRAC=0.5`,
`PYRITE_TEST_WORKERS=1`, `OMP_NUM_THREADS=4`, and `OPENBLAS_NUM_THREADS=1`.
The command, after the normal `pyrite remote sync` into the isolated checkout
and explicit transfer of tests, was:

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run --frozen --extra nvidia --group test \
  pyrite-dev test tests/montecarlo/test_transition_radiation_cuda.py \
  -ra --durations=0 --junitxml=cuda-test-results.xml
```

The log and JUnit XML remain in that remote checkout:
`cuda-test-1592.log` and `cuda-test-results.xml`. SLURM accounting storage is
disabled, so no `sacct` peak-RSS figure is claimed. No production GPU jobs were
cancelled or modified. The experimental two-beam DTR solver is CPU-only and
is not part of these 16 tests.
