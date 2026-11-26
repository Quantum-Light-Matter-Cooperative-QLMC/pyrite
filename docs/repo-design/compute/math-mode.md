# Floating-point math mode

One setting, `math`, selects how strictly floating-point kernels compile
(#1468). This page is its contract: the values, where the setting lives, what
each value guarantees, how identity and provenance record it, and how a kernel
adopts the `fast` variant. Kernel adoption lives in the CPU (#1469) and GPU
(#1463) sub-issues.

## Values

| value | guarantee |
|---|---|
| `reproducible` | Today's guarantees, unchanged: no Numba `fastmath`, device kernels compiled with `_backend.exact_rawkernel_options` (`--fmad=false` / `-ffp-contract=off`, #888), the exact `vector_sin` (#1459), and bitwise parity wherever it is pinned now. Identity pins, physics baselines and validation evidence use it. |
| `fast` | Adopted kernels may use fast-math flags, approximate transcendentals (SVML, a reduced-accuracy sine, device intrinsics) and FMA contraction. See [the `fast` guarantee](#the-fast-guarantee). |

The built-in default is `reproducible`. The issue's goal is `fast` by default;
the flip waits for at least one adopted kernel and passing validation
([Default flip](#default-flip)).

## Where it lives and precedence

The mode resolves once per run, at one point: case lowering
(`campaign.lowering._with_run_options`) for the Python API, and
`dataset_identity` plus the legacy-sweep adapter for configured campaigns. Both
call `pyrite._math_mode.resolve_math_mode`:

1. an explicit value: `KernelTuning(math=...)` (`ExecutionConfig.math`) in
   Python, or a `math` key in a catalog profile or `[numerics.NAME]` preset
   (`pyrite profile numerics set PROFILE --math fast`,
   `pyrite numerics create|set NAME --math fast`);
2. the `PYRITE_MATH` environment variable;
3. the `numerics.math` key of the user config store (`pyrite config set numerics.math fast`);
4. the built-in default.

`pyrite cache status` prints the mode a run that sets none would use and its
source (JSON: `payload.math`). Remote job scripts forward `PYRITE_MATH`; a
remote box resolves its own config store, so pin `math` in the profile or the
environment when a locally predicted checkpoint stem must match the remote run.

## The `fast` guarantee

- **Determinism:** run-to-run identical on one host, build and thread count
  where that is cheap (no adopted kernel may reassociate a reduction whose order
  depends on scheduling).
- **No parity across** instruction sets, devices, GPU models, Numba/LLVM,
  SVML or driver versions.
- **Accuracy:** each adopted kernel's owner states its bound (ulp error of the
  approximated function and the relative change of the affected spectrum) in
  the kernel docstring and its design page, and tests it under `fast`.
- **Statistics:** Monte Carlo outputs are statistically equivalent to
  `reproducible` within their Monte Carlo error. Branchy transport kernels whose
  accept/reject decisions consume RNG may adopt only with validation evidence;
  reassociation that changes RNG consumption order is out of scope unless
  validated.
- `nnan`/`ninf` flags only where a kernel's inputs are finite by construction.

## Identity and provenance

`math` **enters case identity** (owner decision 2026-11-25, option a), unlike
the `KernelTuning` rounding knobs:

- `Case.math` is `"fast"` or absent. Absent is `reproducible`, so every
  existing case key, `case_content_key` and dataset `parameter_sha256` is
  unchanged.
- `dataset_identity` adds `resolved_parameters.math = "fast"` only for a fast
  run (divergence-only). A fast run therefore has its own checkpoint stem and
  content-store keys: resume never mixes modes, and switching modes recomputes.
  Scoring saved trajectories under a different mode is refused like any other
  case-field mismatch.
- Run provenance (`provenance["math"]`) records the mode, Numba version, SVML
  state (`numba.config.USING_SVML`), host CPU name and vector features, and the
  GPU model (`None` on the CPU).

Revisit option (b), provenance only, once validation shows statistical
equivalence.

## Kernel adoption

The runner enters `math_mode_scope(case_mode)` for every case
(`with_case_math_mode` on `run_case`, the reline mirrors and the emission
driver). The scope is process-wide, so runner worker threads see it; two
different modes cannot run at once in one process.

**Numba.** Decorate with `pyrite._math_mode.mode_njit(fast=FLAGS, **njit_options)`
instead of `njit`. `FLAGS` is `True` (all LLVM fast-math flags, which SVML's
low-accuracy transcendentals need) or the validated subset of `nnan`, `ninf`,
`nsz`, `arcp`, `contract`, `afn`, `reassoc`. The `reproducible` dispatcher is
exactly `njit(**options)`; the `fast` one compiles a renamed copy
(`<qualname>__math_fast`). The copy matters: Numba's on-disk cache index keys
on qualname, line and signature, not on `fastmath`, so two `cache=True`
dispatchers of one function would load each other's machine code. Calling the
kernel dispatches on `active_math_mode()`; an `njit` caller picks a dispatcher
with `kernel.for_mode(mode)` before it compiles.

**GPU.** Compile with `_backend.rawkernel_options(cp, mode, fast=FLAGS)`.
`reproducible` returns exactly `exact_rawkernel_options`; `fast` permits FMA
contraction (`--fmad=true` / `-ffp-contract=fast`) and appends the kernel's
validated flags. CuPy's cache separates option sets, but a builder that caches
its module (`lru_cache`) must key that cache on the mode.

Each adoption adds a test that `reproducible` is bitwise unchanged and that
`fast` stays within the stated bound.

## Tests and CI

`tests/conftest.py` sets `PYRITE_MATH=reproducible` for the whole suite, so
identity pins and physics baselines stay bitwise; `PYRITE_TEST_MATH=fast`
overrides it on purpose. `tests/test_math_mode.py` covers resolution,
dispatch, cache separation, compile options, identity and provenance.
`tests/montecarlo/runner/test_math_mode_equivalence.py` compares a representative
CXR + bremsstrahlung case and its transport tallies under both modes with
explicit Monte Carlo tolerances; the `math-fast` CI job runs it with
`PYRITE_TEST_MATH=fast` in the default and SVML environments.

## Default flip

Deferred. Flip to `fast` when at least one sub-issue has adopted a kernel and
the #1468 validation (photon-case, layered, channeling and one GPU case on
qlmc, within Monte Carlo error; evidence in `~/dev/pyrite-benches/`) passes.
The flip changes `DEFAULT_MATH_MODE` and the `numerics.math` built-in together,
bumps the affected `*_MODEL` markers, re-mints pins with
`pyrite-dev test-suite identity-pins --update-pins`, and names
`math="reproducible"` in the release notes as the way to get bit-for-bit
results. Until then no kernel differs between modes, so `fast` results equal
`reproducible` results bit for bit while carrying a distinct identity.
