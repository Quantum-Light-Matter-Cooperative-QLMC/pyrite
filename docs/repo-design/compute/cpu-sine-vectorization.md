# CPU sine vectorization, fastmath and SVML (#1459)

The support-windowed sinc² line reduction
(`cxr/_node_window.py::_sincsq_window_reduce`) took 73–80 % of the 1-thread
wall time of the photon-case and layered audit cases (`checks/native_cpu_audit`,
#1432). This page records why `fastmath` and Intel SVML did not speed it up as
written, what replaced them, and the dependency-stack decision.

## Decision

- **Adopted by default:** `vector_sin` (`cxr/_vector_sin.py`), a branch-free
  port of the fdlibm reduction and kernels.
  - Accuracy: ≤0.79 ulp measured (libm: 0.52 ulp).
  - Bits are identical on every host checked.
  - `CXR_ESCAPE_MODEL` advanced to `.../fp64-spectrum-sum-v2`.
- **Rejected:** `fastmath` flags without SVML. They gain only 1–3 % once the
  loop vectorizes, and they would let LLVM contract or reorder operations
  differently per host.
- **Not default; opt-in proposed:** SVML. It needs Numba ≤0.61 with
  Anaconda's patched llvmlite; the costs are under
  [Dependency stack](#dependency-stack).
  - SVML's high-accuracy `_ha` sine (0.50 ulp) is no faster than
    `vector_sin`.
  - Its remaining 2.5x kernel gain comes only from the low-accuracy sine
    (3.3 ulp). LLVM selects that sine only under the full `fastmath=True` set:
    dropping `nnan` or `ninf` returns the kernel to about 0.5 s.
  - See [Opt-in SVML path](#opt-in-svml-path).

**Tolerance.** A changed default here must:

- be bit-deterministic across threads, Numba versions and CPU instruction sets;
- keep each sine within 1 ulp;
- move the reduced spectrum by no more than 1e-15 relative to the libm result;
- leave the error against an extended-precision reference unchanged.

`vector_sin` meets all four: 3.9e-16 maximum relative change, and 5.2e-14
against long double for both the old and the new kernel (summation-limited).

## Why SVML did nothing

SVML replaces scalar `sin` calls only inside loops that LLVM vectorizes, and
only in `parallel=True` kernels. The old kernel looped over
`range(lo[j], hi[j] + 1)`. A variable range start keeps Numba's
negative-index wraparound check, which blocks vectorization, so neither SVML
nor `fastmath` could act.

Looping over zero-based window slices (65ef52609) removes the check. Same
operations, same order: the output stays bit-identical, and the kernel gets
1.6x faster on stock Numba 0.68.

The loop then vectorizes everything except the libm `sin` call. That call is
what SVML substitutes, and what `vector_sin` replaces portably.

## Kernel measurements

Inputs: 18 captured reducer calls of the photon-case (Cu 30 keV), all
direct-mode full-grid windows of 400 nodes. Hardware: i7-13620H (AVX2), best
of 3. Call 4 has 454,629 lines; its time is at 1 thread.

| Kernel | Numba | Call 4 | 18 calls, 1 thread | 18 calls, 4 threads | Max rel. vs libm |
|---|---|---:|---:|---:|---:|
| original (`range(lo, hi+1)`, libm) | 0.68.1 | 2.75 s | 33.4 s | 9.5 s | 0 |
| zero-based slices, libm | 0.68.1 | 1.61 s | – | – | 0 (bit-identical) |
| slices + `fastmath` | 0.68.1 | 1–3 % faster | – | – | – |
| slices + SVML (`_ha`, strict) | 0.61.2 | 0.52 s | – | – | 1.2e-16 |
| slices + SVML + `fastmath` (`afn`/`arcp`/`contract`/`reassoc`/`nsz`) | 0.61.2 | 0.50 s | – | – | 1.1e-15 |
| slices + SVML + `fastmath=True` (low-accuracy sine) | 0.61.2 | 0.23 s | – | – | 7.5e-16 |
| **slices + `vector_sin` (default)** | 0.68.1 | **0.59 s** | **6.9 s** | **2.2 s** | 3.9e-16 |

`vector_sin` alone runs 6–8x faster than a libm loop on 4 M arguments. Its
errors against a long-double reference:

| Arguments | Max error |
|---|---:|
| uniform on ±1e5 | 0.77 ulp |
| uniform on ±4 | 0.78 ulp |
| doubles nearest k·π/2 and their neighbours | 0.50 ulp |

## Determinism

The `vector_sin` window-kernel outputs have the same SHA-256 under all of:

- 1 and 4 Numba threads;
- Numba 0.61.2 and 0.68.1;
- SVML on and off;
- the i7-13620H (AVX2, no AVX-512) and qlmc-queen's Ryzen 9 9950X3D (AVX-512).

On identical inputs, an array of 4 M `vector_sin` values is bit-identical
across the two CPUs. `test_vector_sin.py` pins a digest.

Under the SVML stack the slice kernel's libm path does change bits, because
SVML engages once the loop vectorizes. This is one more reason SVML stays out
of the default stack. Whether SVML gives the same bits on AVX2 and AVX-512 was
not measured, because no pkgs/main environment exists on the lab box; the
deferral makes that measurement unnecessary.

## Case wall time

Hardware: qlmc-queen (Ryzen 9 9950X3D, AVX-512), four bound CPUs, Slurm jobs
16006–16009. Each arm is `checks/native_cpu_audit.py --case NAME --repeats 2`.
Wall time is the mean of the two warm samples. The first, cold sample includes
JIT compilation and is excluded.

| Case | Numba threads | Base 9836de899 | `vector_sin` 4703ea1a4 | Speedup | Reducer, base → new |
|---|---:|---:|---:|---:|---:|
| photon-case (Cu 30 keV) | 1 | 26.9 s | 9.3 s | 2.9x | 20.6 → 2.7 s |
| photon-case | 4 | 10.9 s | 7.0 s | 1.6x | 5.3 → <1.4 s |
| layered (MoS2/Si 100 keV) | 1 | 91.4 s | 39.4 s | 2.3x | 66.2 → 8.9 s |
| layered | 4 | 43.7 s | 30.0 s | 1.5x | 17.2 → <6.2 s |

At 4 threads the reducer is no longer the top exclusive cost: `searchsorted`
leads. Removing the remaining reducer time entirely would save at most 23–29 %
of a 1-thread case. The 2.5x kernel gain of low-accuracy SVML is therefore
worth about 13–17 % of 1-thread case time (1.6 s of 9.3 s; 5.3 s of 39.4 s).

## Dependency stack

A dry run built Python 3.13.18, NumPy 2.2.5, Numba 0.61.2 and the SVML llvmlite
0.44.0 from Anaconda `pkgs/main`. Then:

- **Tests:** 517 of 518 sampled tests across spectrum, transport,
  energy-grid, materials, the JIT cache and the CLI pass. All imports succeed.
- **Floors:** `numpy>=2.4.6` and `numba>=0.66.0` were set to whatever was
  current at `uv add` time; nothing in the code needs them.
- **The one failure:** the JIT-cache cross-process test. Numba 0.61 stamps
  cached kernels by source mtime, so every remote sync or checkout recompiles
  every `cache=True` kernel.
- **Python 3.14 is lost:** Numba 0.61 supports Python <3.14 only.
- **Packaging:** uv and PyPI cannot express the stack. A default
  `pip install -e` silently upgrades to Numba 0.68 and drops SVML.
  conda-forge's Numba 0.61 has no SVML, and Numba 0.62 removed SVML upstream.
- **Licensing:**
  - Anaconda `pkgs/main` is free for academic institutions only for
    non-commercial, educational use, and for companies with 200 or fewer
    staff.
  - `intel-cmplr-lib-rt` is under the proprietary Intel EULA for Developer
    Tools, so it can be depended on but not vendored.
- **Migration:** an opt-in conda or pixi "svml" environment would take about
  1–2 days and 5–10 files. pixi needs a `conda-pypi-map` for `pkgs/main`, or
  it double-installs NumPy. A full move from uv to pixi touches 507 tracked
  files and all four CI workflows, about 1.5–3 weeks.

## Opt-in SVML path

Proposed; not implemented. All steps are measured except the runtime switch.

1. **Relax the floors** to `numpy>=2.2` and `numba>=0.61`.
   - With those floors, `pip install` inside a `pkgs/main` environment keeps
     the conda numpy 2.2.5, numba 0.61.2 and llvmlite 0.44.0 as already
     satisfied (dry run).
   - `uv.lock` still resolves to the latest versions, so the default install
     does not change.
2. **Ship one environment file,** `environment-svml.yml`:

   ```yaml
   name: pyrite-svml
   channels: [pkgs/main, nodefaults]
   dependencies:
     - python=3.13
     - numpy=2.2
     - numba=0.61
     - llvmlite=0.44
     - intel-cmplr-lib-rt
     - pip
     - pip:
         - pyrite-mc   # or "-e ." from a checkout
   ```

   - The user runs `conda env create -f environment-svml.yml`, then
     `numba -s | grep SVML`.
   - A from-checkout version of this file built in 17 s from a warm package
     cache, and all four SVML checks reported True.
3. **Add a runtime switch** `PYRITE_CPU_MATH=svml-fast`, set through the
   config table in `console/config.py`.
   - It compiles a second window kernel that calls `np.sin` with
     `fastmath=True`.
   - If `numba.config.USING_SVML` is false, it refuses with a clear message
     rather than silently running slower.
   - The kernel's inputs are finite by construction (`x = 0` is replaced
     before the division), so `nnan` and `ninf` are safe there.
4. **Give it its own identity.** The marker gains `/cpu-math-svml-fast-v1`,
   computed at import like `characteristic_model_marker()`. Provenance records
   the Numba version, SVML state and CPU instruction set.
   - SVML dispatches by instruction set, and AVX2 vs AVX-512 bits are
     unmeasured.
   - Checkpoints from the two modes therefore never mix.
5. **Cover it in CI or the lab:**
   - one job that builds the environment and runs the spectrum node-window
     tests and `test_vector_sin.py`;
   - a fix to the JIT-cache test for Numba 0.61's mtime stamps.

Costs carried by users of the opt-in:

- Python 3.13 only.
- Anaconda's terms (non-commercial academic use, or companies with 200 or
  fewer staff).
- Recompiles after every sync.
- A frozen stack with no upstream future, since Numba 0.62 removed SVML.

An alternative with no stack change is a reduced-accuracy `vector_sin`
variant. It would not need SVML at all; it is not yet measured.

Raw evidence (scripts, replay JSON, ISA probes, dry-run logs and report) is in
the benchmark archive `issue-1459/` with `SHA256SUMS.all`.
