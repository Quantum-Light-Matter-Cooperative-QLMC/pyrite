# Groove Verification Baseline Repair Design

**Date:** 2026-07-24  
**Base:** `bec8fb7` on `feat/cxr-blaze-grooved-sweep`

## Goal

Restore repository-wide verification without changing groove physics or
absorbing unrelated dirty-checkout work. Eleven full-suite failures reproduce
unchanged at pre-groove commit `9f0c977`; groove-focused integration passes
150 tests.

## Scope

1. Restore missing `cxr_mc.remote` rebrem facade aliases whose owning
   implementations and public tests already exist.
2. Reconcile analysis-app source-contract tests with intentional current
   action names and responsive control-row structure. Do not redesign UI.
3. Add the four intentional dataset-partial symbols already exported by
   `cxr_mc.results` to its frozen export contract.
4. Replace unsupported cross-Python bitwise float assumptions with tight
   numerical comparisons. Preserve exact keys, shapes, ordering, strings,
   integer values, and other nonnumeric structure.

## Numerical Contracts

- Lattice and serialized-golden floats compare with tolerances chosen above
  observed one-ULP parser differences but far below physical-data precision.
- Orientation matrices retain a near-machine-precision tolerance and exact
  shape. No production orientation calculation changes.
- Golden drift tests still fail for structural, categorical, ordering, or
  materially different numeric changes.

## Isolation

Work occurs only in `.worktrees/groove-verify-fixes` on branch
`fix/groove-verification-baseline`, based at `bec8fb7`. Existing dirty checkout
and shared qlmc deployment remain untouched. Testing uses tracked-only archives
in isolated qlmc directories with CUDA hidden and low CPU priority.

## Verification

Use existing 11 failures as RED evidence. Repair one cluster at a time and run
its focused tests remotely. If `notebooks/analysis_app.py` changes, run Marimo
check. Then run:

1. all 11 baseline regression tests;
2. 150-test groove integration suite;
3. full `scripts/dev.py verify`.

No completion claim unless final full verification exits zero.
