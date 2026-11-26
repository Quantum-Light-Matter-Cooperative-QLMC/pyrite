# CPU kernels under `math="fast"` (#1469)

This page records which CPU kernels gain from the `fast` math mode of
[the math-mode contract](math-mode.md), what each candidate costs in accuracy,
and what was adopted. It follows
[CPU sine vectorization](cpu-sine-vectorization.md) (#1459), which made the
node-window reducer vectorize and rejected `fastmath` without SVML.

## Decision

- **Adopted, `fast` only:** the node-window reducer
  (`cxr/_node_window.py::_sincsq_window_reduce_fast`).
  - It computes `sinc²` with `sinc2_fast` (`cxr/_vector_sin.py`): a
    reduction by π with one degree-19 polynomial, no quadrant select and no
    head/tail correction, on the direct argument `a(E - E_r)`.
  - It compiles with `fastmath={"contract"}` (FMA).
  - Kernel speedup: 2.9x on qlmc-queen and 2.6x on the AVX2 laptop. The
    reduced spectrum moves by at most 2.9e-14 relative on real calls.
  - Complete cases: 1.30x on photon-case and 1.23x on layered (qlmc-queen,
    1 thread), with outputs within 2e-16 of their peak ([Validation](#validation)).
  - `reproducible` still runs the unchanged #1459 kernel. Identity pins and
    physics baselines pass without a re-mint.
- **Not adopted:**
  - The `reassoc` flag. Together with `nsz` it folds the rounding step
    `(x c + R) - R` of every Cody-Waite reduction to `x c`, which gives 90 %
    spectrum error. That rules out `fastmath=True` for any kernel that uses
    `vector_sin` or `sinc2_fast`.
  - SVML-fast (`np.sin` with `fastmath=True` in the pixi `svml` environment).
    It is no faster than `sinc2_fast` with `contract`, and it needs the frozen
    SVML stack.
  - Every other CPU kernel. The rest of the spectrum time is NumPy, not
    Numba, so `fastmath` cannot reach it.
  - Transport fastmath. Even `fastmath=True` gains at most 2 % on the
    transport-heavy case, and it changes the sampled histories (see
    [Survey](#survey)).

## Survey

Hardware: qlmc-queen, a Ryzen 9 9950X3D (AVX-512). Each arm ran on four
OS-bound CPUs with 1 Numba thread, as one
`checks/native_cpu_audit/fastmath_arms.py` process (job 16044, checkout
`e1f8dda78`). Times are exclusive cProfile seconds from the profiled call.
A Numba kernel's time is charged to its Python caller.

"Vectorizes" means the compiled kernel contains packed FP64 arithmetic
(`v…pd` on `ymm`/`zmm`). "Transcendentals" lists the math calls in its LLVM IR.

| Kernel (route) | photon-case | layered | transport-si300 | Kind | Vectorizes? | Transcendentals | Finite inputs? | Options |
|---|---:|---:|---:|---|---|---|---|---|
| `_sincsq_window_reduce` (node-window reducer) | 33 % | 27 % | – | Numba, `parallel` | yes (108 packed / 200 scalar FP64 ops; `zmm`) | `vector_sin` inline; libm `sin` only for windows past `VECTOR_SIN_LIMIT` | yes (`x = 0` replaced before the division) | cheaper sine, `contract`, SVML-fast: **adopted** |
| `ndarray.searchsorted` (line brackets, grids) | 18 % | 19 % | – | NumPy | – | – | – | none: not Numba |
| `_quantum_line_factors`, `_cxr_amp_sq_total_core` (line amplitudes) | 8 % | 6 % | – | NumPy array expressions | – | `sqrt` | – | none: a Numba port is a separate optimization |
| `_interp_gather2d`, `_interp_index`, `_interp_gather1d` (table interpolation) | 9 % | 7 % | – | NumPy gathers | – | – | – | none |
| `reflection_coupling_tables`, `_reflection_families` (lattice tables) | 1.5 % | 8 % | – | Python/NumPy | – | – | – | none |
| `_spectrum_tile` (fused BremsLib continuum) | 2.5 % | not in the top 100 | – | Numba, `parallel` | no (0 packed) | `exp`, `expm1`, `hypot` | not checked | too small to rank |
| transport core (`cores.make_cpu_transport_core.body`) | < 1 % | < 1 % | 66 % (the rest is the harness hashing 6 M segments) | Numba, scalar | no (5 packed / 1152 scalar) | `log`, `exp`, `sin`, `cos`, `sqrt`, `pow`, `atan`, `log1p`, `expm1`, `log10` | not established | flag sets measured: ≤ 2 %, rejected |

Shares are of the profiled call: 8.05 s for photon-case, 33.2 s for layered
and 2.92 s for transport-si300.

The eligible share is the node-window reducer alone: 27–33 % of a 1-thread
spectrum case. Everything else that ranks is NumPy. The transport core is
scalar and branchy, and every flag set leaves it within 2 % (below).

**Vectorization prerequisites (Plan item 2).** There is nothing to
restructure. The one ranked Numba loop already vectorizes since #1459. The
BremsLib tile and the transport core do not vectorize, but at 2.5 % and 2 %
potential neither justifies a restructure that would have to keep
`reproducible` bits.

## Speed and accuracy per arm

### Sine and sinc²

Each figure is the best of 15 runs over 4.18 M arguments at 1 thread:

- 3 M uniform on ±1e5;
- 1 M uniform on ±4;
- the doubles nearest `kπ/2` and their neighbours.

The ulp error is measured against `long double`. The `sinc²` rows give the
ulp error of `(sin y / y)²`. "Bits" says whether the output digest is the same
on both hosts.

| Arm | i7-13620H, AVX2 (ms) | 9950X3D, AVX-512 (ms) | max ulp, uniform | max ulp, near `kπ/2` | Bits |
|---|---:|---:|---:|---:|---|
| `vector_sin` (reproducible) | 10.5 | 3.33 | 0.77 | 0.50 | same |
| `sin_pi_u35` (reduction by π, one polynomial) | 6.4 | 1.81 | 2.33 | 5.0e4 (absolute ≤ 1.1e-16) | same |
| `sin_pi_u35` + `contract` | 5.3 | 1.31 | 2.31 | 5.0e4 | same |
| libm `np.sin` | 63 | 48 | 0.52 | 0.50 | same |
| SVML `np.sin` (`_ha`, strict) | 21 | 14 | 0.50 | 0.50 | same |
| SVML `np.sin` + `fastmath=True` | 3.7 | 1.58 | 3.24 | 1.0 | same |
| `(vector_sin(y)/y)²` (reproducible) | 19.7 | 3.49 | 3.63 | 3.35 | same |
| `sinc2_fast` | 5.9 | 1.83 | 6.61 | 8.4e4 | same |
| `sinc2_fast` + `contract` | 3.9 | 1.24 | 6.61 | 8.4e4 | same |

Near the zeros of `sin`, the reduction by π loses relative accuracy, while the
absolute error of `sin²` stays below 1e-15. A `sinc²` sum carries that error
at the scale of its peak, not of its zeros.

### Node-window reducer

The arms replay the 18 reducer calls captured from photon-case
(`reducer_replay.py`; 5.2 M lines and 2.1 G window nodes in all). Each arm is a
variant of the production source with only the named edit (`fast_variants.py`):

- `prod`: today's kernel (`vector_sin`);
- `u35`: `sin_pi_u35` in place of `vector_sin`;
- `sinc2`: `sinc2_fast` with the direct argument.

Times are the sum over the 18 calls of each call's best of 3, at 1 thread.
"Rel. change" is the largest relative change of a reduced spectrum against
`prod`. The long-double error is the largest relative error of call 0
against an extended-precision reference.

| Arm | queen (s) | laptop (s) | Speedup (queen) | Rel. change | Long-double error |
|---|---:|---:|---:|---:|---:|
| `prod` strict (reproducible) | 2.64 | 7.73 | 1.00x | 0 | 5.1e-14 |
| `prod` + `contract` | 2.21 | 6.35 | 1.20x | 5.4e-16 | 5.1e-14 |
| `prod` + `afn` / `nsz` alone | 2.65 | 7.6–8.0 | 1.00x | 0 (same bits) | 5.1e-14 |
| `prod` + `arcp,contract` | 1.98 | 6.34 | 1.34x | 4.0e-14 | 5.1e-14 |
| `prod` + `reassoc,nsz` | – | 5.18 | – | **0.92** | – |
| `prod` + `afn,arcp,contract,reassoc,nsz` | 1.03 | 3.99 | 2.57x | **0.92** | 0.84 |
| `prod` + `fastmath=True` | 1.02 | 4.80 | 2.60x | **0.92** | 0.84 |
| `u35` strict | 1.75 | 5.34 | 1.51x | 4.9e-16 | 5.1e-14 |
| `u35` + `contract` | 1.24 | 4.45 | 2.13x | 6.1e-16 | 5.1e-14 |
| `sinc2` strict | 1.33 | 3.29 | 1.99x | 2.8e-14 | 5.4e-14 |
| **`sinc2` + `contract` (adopted)** | **0.92** | **2.99** | **2.86x** | **2.8e-14** | **5.4e-14** |
| `sinc2` + `arcp,contract` | 0.91 | 2.75 | 2.91x | 2.8e-14 (same bits as `contract`) | 5.4e-14 |
| SVML `np.sin` strict (`_ha`) | 4.66 | 9.44 | 0.57x | 3.8e-16 | 5.1e-14 |
| SVML `np.sin` + `fastmath=True` (SVML-fast) | 0.99 | 4.57 | 2.66x | 2.8e-14 | 5.4e-14 |

The laptop's absolute times drift by up to 30 % between runs (the same `prod`
bits took 7.7 s and 10.0 s), so compare ratios within one host only.

Other findings from the replays:

- `reassoc` alone, `reassoc,afn`, `reassoc,arcp` and `reassoc,contract` stay
  within 3e-14. `reassoc` together with `nsz` breaks the reduction.
- **Cross-ISA bits.** The same captured inputs were replayed on both hosts
  (jobs 16048 and 16049 against the local runs). Every arm in the table gave
  identical output bits on AVX2 and AVX-512, under both Numba 0.61 (SVML) and
  Numba 0.68:
  - FMA contraction, because both CPUs have FMA;
  - SVML-fast, which #1459 left unmeasured.

  The `fast` contract still promises no parity across hosts. A host without
  FMA rounds `contract` differently.

### Complete cases

Global flag sets inject one `fastmath` set into every PyRITE `@njit`
(`fastmath_arms.py --fastmath`). That is an upper bound, not a production
variant, because it also compiles `vector_sin` with the flags. The rows below
are job 16044 on qlmc-queen at 1 thread. Wall time is the mean of 2 warm
samples; the spread between samples is under 0.5 %. "Change" is the largest
absolute change of any float output array, as a fraction of that array's
peak.

| Case | Arm | Warm wall (s) | Speedup | Reducer (s) | Change |
|---|---|---:|---:|---:|---:|
| photon-case | strict | 7.68 | 1.00x | 2.67 | 0 |
| photon-case | `afn` | 7.70 | 1.00x | 2.67 | 0 (same bits) |
| photon-case | `contract` | 7.20 | 1.07x | 2.21 | 1.3e-13 |
| photon-case | `arcp,contract` | 6.96 | 1.10x | 1.98 | 1.5e-13 |
| photon-case | `afn,arcp,contract,reassoc,nsz` / `True` | 9.41 / 9.44 | 0.81x | 1.67 | **0.68** |
| layered | strict | 31.47 | 1.00x | 8.85 | 0 |
| layered | `afn` | 31.25 | 1.01x | 8.87 | 0 (same bits) |
| layered | `contract` | 29.52 | 1.07x | 7.38 | 2.5e-11 |
| layered | `arcp,contract` | 28.73 | 1.10x | 6.58 | 2.6e-11 |
| layered | `afn,arcp,contract,reassoc,nsz` / `True` | 37.14 / 37.11 | 0.85x | 4.78 | **0.81** |
| transport-si300 | strict | 2.91 | 1.00x | – | 0 |
| transport-si300 | `afn` | 2.94 | 0.99x | – | 0 (same bits) |
| transport-si300 | `contract` | 2.90 | 1.00x | – | 2.4e-09 (histories change) |
| transport-si300 | `arcp,contract` | 2.89 | 1.01x | – | 2.4e-09 |
| transport-si300 | `True` | 2.86 | 1.02x | – | 1.5e-09 |

What the rows show:

- `fastmath=True` makes the spectrum cases slower (0.81–0.85x) as well as
  wrong. Their extra time goes to other kernels; the reducer itself does get
  faster.
- On the transport-heavy case every flag set stays within 2 %. FMA
  contraction already changes the sampled histories: the segment arrays
  differ at 2.4e-9 of their peak. So transport offers no payoff worth the
  RNG-consumption validation that #1468 requires.
- `contract` also moves the layered output by 2.5e-11 of its peak through
  kernels other than the reducer. That is why the adopted variant applies
  flags to the reducer alone.

Reducer variants alone (`--reducer`, everything else reproducible), on the
AVX2 laptop (i7-13620H, photon-case, 1 thread, 2 warm samples):

| Arm | Warm wall (s) | Speedup | Reducer (s) | Change |
|---|---:|---:|---:|---:|
| reproducible | 15.54 | 1.00x | 7.84 | 0 |
| `sinc2` strict | 12.17 | 1.28x | 3.39 | 3.3e-16 |
| `sinc2` + `arcp,contract` | 10.45 | 1.49x | 2.32 | 3.3e-16 |

## Validation

The adopted kernel against `reproducible` on complete cases (#1468's
validation, moved here). Each mode ran in its own process with its own Numba
cache. "Change" is the largest absolute change of any output array as a
fraction of its peak; "sum change" is the largest relative change of an
array's sum. CPU rows are qlmc-queen at 1 thread (jobs 16050/16051, 16054/16055;
mean of 2 warm samples). The GPU row is qlmc-ace, Radeon 8060S with FP64,
ROCm (job 16056).

| Case | Backend | `reproducible` (s) | `fast` (s) | Speedup | Change | Sum change |
|---|---|---:|---:|---:|---:|---:|
| photon-case | CPU | 7.64 | 5.90 | 1.30x | 1.7e-16 | 0 |
| layered | CPU | 31.06 | 25.27 | 1.23x | 1.1e-16 | 0 |
| channeling-demo (admitted planar case) | CPU | 13.36 | 13.03 | – | 0 (same bits) | 0 |
| layered | GPU (ROCm) | 5.71 | 5.70 | – | 0 (same bits) | 0 |

- Both modes are far inside Monte Carlo error. These cases' relative MC error
  is about 1e-2 or larger per bin (the channeling case's budget is 6 %), and
  the largest change is 1.7e-16 of the peak.
- The `reproducible` photon-case arrays are bit-identical to the
  pre-adoption run (job 16044), so adoption left `reproducible` unchanged.
- Channeling and the GPU path do not call the CPU node-window reducer, and no
  GPU kernel has a `fast` variant yet (#1463). Their outputs are identical in
  both modes, which confirms the mode switch changes nothing else.
- GPU photon-case did not run. The harness refuses a case whose two warm
  repeats differ, and on ROCm photon-case's repeats differ in both modes. That
  is run-to-run nondeterminism on that device, not a math-mode effect.

On this evidence the default is `fast` (#1469). The flip is a separate commit;
see [math mode](math-mode.md#default-flip).

## Reproduce

All scripts are in `checks/native_cpu_audit/`; `README.md` there lists them.

```bash
# complete cases under one arm each (qlmc-queen, clean isolated checkout)
sbatch -w qlmc-queen --mem=12G checks/native_cpu_audit/fastmath.sbatch cases asm contract afn arcp_contract no_nnan_ninf true
CASES="photon-case layered" sbatch -w qlmc-queen checks/native_cpu_audit/fastmath.sbatch cases strict reducer=sinc2:contract
PYRITE_MATH=fast CASES="photon-case layered" sbatch -w qlmc-queen checks/native_cpu_audit/fastmath.sbatch cases strict
PYRITE_MATH=reproducible CASES="photon-case layered" sbatch -w qlmc-queen checks/native_cpu_audit/fastmath.sbatch cases strict
# validation: channeling case under each mode, one GPU case, then compare
PYRITE_MATH=fast CASES=channeling-demo sbatch -w qlmc-queen checks/native_cpu_audit/fastmath.sbatch cases strict
CASES="layered" sbatch -w qlmc-ace checks/native_cpu_audit/math_gpu.sbatch
pixi run python checks/native_cpu_audit/compare_modes.py REPRODUCIBLE.npz FAST.npz
# sine micro + reducer replay (captures photon-case inputs once)
sbatch -w qlmc-queen checks/native_cpu_audit/fastmath.sbatch replay prod:strict sinc2:contract
PIXI_ENV=svml sbatch -w qlmc-queen checks/native_cpu_audit/fastmath.sbatch replay prod:strict npsin:true
```

Raw evidence (job logs, JSON, output arrays, the local AVX2 runs and
`SHA256SUMS.all`) is in the benchmark archive `issue-1469-cpu-fastmath/`.
