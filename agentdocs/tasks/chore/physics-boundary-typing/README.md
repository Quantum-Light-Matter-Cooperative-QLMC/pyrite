# Physics boundary typing

## Problem and scope

`cxr-dev typecheck` (`ty`, default rules) passes clean, but that is a weak
signal: the physics packages are the least annotated in the tree, so there is
little for the checker to check.

Parameter annotation coverage, measured 2026-08-07:

| package | annotated params | |
| --- | --- | --- |
| `montecarlo` | 44 / 951 | **4.6%** |
| `detectors` | 10 / 151 | **6.6%** |
| `energy_grid` | 132 / 498 | 26.5% |
| `cli` | 251 / 815 | 30.8% |
| `materials` | 125 / 209 | 59.8% |

Of the 42 functions re-exported in the `montecarlo` / `detectors` / `materials`
`__all__` lists, **2 are fully annotated**, 2 partially, 38 not at all.

In scope: annotate the *public boundary* of `montecarlo`, `detectors`, and
`materials` — the names in those `__all__` lists, which is what `run.py`,
`sweep.py`, `scan.py`, and the CLI call — with unit-carrying scalar aliases and
explicit `| None`.

Out of scope, deliberately:

- **Blanket annotation of inner array helpers.** `np.ndarray` on every kernel
  argument cannot distinguish `(n_seg, N_g)` from `(N_g,)`, which is where the
  real defects are. It is churn that buys a number, not safety.
- **A stricter ty gate.** `ty check --error all` over the three packages
  produces 51 diagnostics: 43 `possibly-unresolved-reference`, 7
  `missing-override-decorator`, 1 `missing-type-argument`. The 43 are almost
  all the correlated-guard false positive (`spectrum.py:1977` — `layer_mu` and
  `inv_nz` are assigned under the same `layers is not None` test the use sits
  behind; ty cannot correlate the two conditions). Not worth gating on.
- Any physics or numerical behavior change. This is signatures only.
- `cli/` and `energy_grid/` annotation debt — separate, lower-stakes.

`ty` has no `disallow_untyped_defs` equivalent, so there is no configuration
that forces this; it is a writing task, and the acceptance check is the
annotation count plus the alias being used, not a checker flag.

## Why these three failure modes

The annotations should be chosen to catch what has actually broken here:

1. **Scalar dtype.** The `cupy.fuse` untyped-scalar zeroing bug was a host-side
   scalar typing mistake — tiny constants flushed to 0 in fused cores because
   they were passed untyped. A `REAL`/`np.float32` alias on those parameters
   states the contract at the call site.
2. **Units.** 278 `_ang`, 228 `_rad`, 206 `_eV`, 162 `_keV`, 92 `_um`, 80
   `_deg` occurrences across the three packages — the convention is carried
   entirely in parameter *names*. `NewType` aliases make `theta_deg` and
   `theta_rad`, or `E_eV` and `E_keV`, unmixable instead of merely
   differently-spelled.
3. **Optionals.** `beam.py`'s `len(row["referenced_by"])` crash on a `None`
   (fixed 2026-08-07 in `cfa7e6d`) is exactly an un-annotated optional crossing
   a boundary.

## Implementation path

New `src/cxr_mc/_types.py` (or extend an existing shared module if one fits
better) holding the unit `NewType`s and the `REAL` scalar alias. Then annotate
outward-in: `materials` (already 60%, smallest delta), `detectors`, then
`montecarlo`.

Likely owners: the `__all__` lists in `montecarlo/__init__.py` (64 names),
`materials/__init__.py` (13), `detectors/__init__.py` (4), and the call sites in
`run.py`, `sweep.py`, `scan.py`.

## Checklist

- [ ] A — Decide the alias vocabulary: which units get a `NewType`, whether
      `REAL` is `np.float32` or backend-dependent, and where the module lives.
      Write it down before annotating anything.
- [ ] B — `materials` public boundary (13 names) — smallest, use it to shake out
      the vocabulary.
- [ ] C — `detectors` public boundary (4 names) plus `_si_sensor` shared
      plumbing.
- [ ] D — `montecarlo` public boundary (64 names) — the bulk. Split further if
      one worker is too coarse.
- [ ] E — Scalar-dtype annotations on the CuPy kernel launch wrappers in
      `montecarlo/*_jit_kernel.py` host code (device bodies excluded — they are
      not Python at runtime).
- [ ] F — Document the convention: where the aliases live, when to use one,
      and that inner helpers are intentionally unannotated.

## Decisions and open questions

- **Decided:** boundary only, not blanket. **Decided:** no stricter ty gate.
- **Open:** `NewType` vs `Annotated` vs plain alias for units. `NewType` gives
  real mixing errors but requires explicit construction at every call site,
  which may be too invasive for `float` literals. `Annotated` documents without
  enforcing. Prototype both in slice B before committing.
- **Open:** does `REAL` belong in the new types module or beside the backend
  dispatch in `montecarlo/_backend.py`, which already owns dtype policy?
- **Open:** do the `@njit` signatures in `transport.py` / `geometry.py` /
  `groove.py` tolerate annotations without upsetting numba's type inference?
  Check before slice D touches them.

## Delegation slices and required skills

- A → `lead-task`; sets the vocabulary every other slice depends on. Do not
  parallelize B–E ahead of it.
- B, C → `implement-task-lite`; `scientific-library`.
- D → `implement-task`; `monte-carlo` + `scientific-library`.
- E → `implement-task`; `performance` (touches the fused-kernel scalar path
  that previously mis-typed constants).
- F → `documentation-maintenance`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev typecheck
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev verify
```

- All 42 public-boundary functions fully annotated (params + return), up from 2.
- Unit aliases used, not bare `float`, wherever a parameter name carries a unit
  suffix.
- No behavior change: golden/regression outputs bit-identical. Any diff is a
  bug in the annotation pass, not an accepted result.
- No new `ty: ignore` comments (currently 7 in all of `src/`).
- Inner array helpers left alone — a diff that annotates them is out of scope.

## Related

- `chore/test-coverage-baseline` — `montecarlo/_backend.py` appears in both
  (69% covered there, dtype policy owner here). Independently ownable; do not
  merge the two.
