# 0023 — Line-grid selector rules

- **Status:** Accepted
- **Date:** 2026-11-17
- **Context:** [#1265](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1265), ADR-0013 (#1243, #1214 amendments), #1230

## Context

By #1214 the automatic line grid had four resolutions, two bandwidths, two
quadratures and windows, joined by rules that depended on one another. The
default resolution depended on the bandwidth: bin-mean got `sinc-shape` on the
measured axis and `sinc-oversampled` on the kinematic one. The fallback then
rewrote `sinc-shape` to `sinc-oversampled` and blocked `resonance-local`
entirely. `sinc-shape` also switched silently to `resonance-local` above 1M
points. As a result, the bin-mean peak bound was 1e-2 or 1.7e-2 depending on
which branch a case took, and the payload `sources` could not tell an explicit
resolution from a defaulted one. An enumeration of all 180 combinations is on
#1265.

## Decision

1. **Resolution depends on quadrature, never on bandwidth.**
   `sinc-oversampled` is the shape-resolving uniform step:
   - `W/8` under `node` (measured, #1214);
   - $(3/\pi)\sqrt{0.01}\,W \approx W/10.5$ under `bin-mean` (derived, #1230).

   `sinc-nyquist` is the area-only step `W`. `resonance-local` is the only
   non-uniform step. It needs `bin-mean`, and it implies `resonance-population`,
   which is recorded in `sources`.
2. **Bandwidth.** When unset, it is `resonance-population` where its truncation
   audit applies, and `kinematic-ceiling` elsewhere (windows, coherent emission,
   `max_dE_frac > 0`).
3. **One fallback rule.** Only a bandwidth with no recorded source may move: to
   `kinematic-ceiling` on routes the audit cannot see, or up the extension
   ladder after an audit refusal. Resolution, quadrature and budget never
   change.
4. **One recorded budget route.** A `sinc-oversampled` axis over `max_points`
   takes the #1257 node-local refinement at its own quadrature's step, recorded
   in the grid record's `local_axis`; only a local axis that also exceeds the
   budget refuses. No other route exists: the former silent `sinc-shape →
   resonance-local` overflow is gone. The default budget is 600 000 points
   under `node` and 1 000 000 under `bin-mean` (#944 Fourier reduction).
   (User decision 2026-11-18, reconciling #1257, which landed first.)
5. **`sinc-shape` is deprecated.** It is an alias of `sinc-oversampled` with
   `bin-mean`, kept for one release.

The selector table in `docs/repo-design/profile-settings.md` is rendered from,
and pinned by, `tests/energy-grid/test_line_grid_selector_table.py`.

## Consequences

- Default `node` cases are unchanged: same payload, same step.
- The bin-mean step tightens from `W/8` to `W/10.5` on kinematic axes, fallback
  axes, and channeling, transition-radiation and mixed cases.
  `LINE_RESOLUTION_DEFAULT_MODEL` moves to `-v2` so results scored on the old
  step re-key rather than resume.
- Default bin-mean cases on the measured axis keep the same step, but their
  payload changes from `sinc-shape` to `sinc-oversampled`, so they re-key.
  Axes that previously overflowed to `resonance-local` now take the
  node-local refinement, or refuse and name `sinc-nyquist` or an explicit
  `resonance-local` when that is also over budget.
- An explicit `resonance-local` with an unset bandwidth now records that
  bandwidth, so it re-keys and never falls back.
- Stored payloads naming `sinc-shape` are refused at resolution and must be
  re-resolved.
