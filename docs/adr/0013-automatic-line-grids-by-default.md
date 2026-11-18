# 0013 — Automatic line grids by default

- **Status:** Accepted
- **Date:** 2026-09-19

## Context

Bundled profiles carried fixed, derived line-grid artifacts and legacy catalog rows. Their resolution could become stale as the transport, lineshape, or grid policy evolved. Issue #125 measured three remaining stored 300 keV rows against fixed-trajectory refinement: hBN missed the intrinsic yield tolerance by 6.5%, diamond by 0.15%, and black phosphorus by 1.3%. Six other suspected materials already fell through to automatic resolution.

The existing case-local resolver derives a conservative kinematic bandwidth from the material and reflection set, then refines its sinc-Nyquist spacing from the run's trajectories. It therefore tracks the actual case and current policy without maintaining a second set of bundled numerical defaults.

## Decision

All bundled profiles use automatic line grids. The packaged catalog contains no `energy_grid_refs` and no legacy `[energy_grids.*]` rows.

Immutable derived artifacts, legacy rows, and explicit fixed grids remain supported as opt-in inputs. Their precedence and schemas do not change.

## Consequences

- Every bundled material follows one case-local bandwidth and resolution policy, including newly added beam energies.
- Catalog case and dataset identities change once because the stored coordinates disappear from resolved configuration.
- Existing bremsstrahlung coordinates are preserved as ordinary profile overrides; this decision changes line grids only.
- Automatic grids can contain more points than a measured derived artifact; users may still install an artifact when that performance trade is useful.
- Shipped artifact payloads that no bundled profile references are removed.

## Amendment (2026-11-16, #1243)

The automatic bandwidth is now `resonance-population`: each case's axis stops
where its measured resonance population leaves at most 1e-4 of the line yield
above it, never above the clamped `kinematic-ceiling` stop. Measured on the
cleo_sub MoS₂ member, the closed-form axis spent most of its scoring cost on
nodes no selected line reaches (see #1243 for the A/B table). Cases whose
routes the truncation audit cannot gate (feature windows, coherent emission,
channeling, transition radiation, mixed transport, flight-grouped substeps)
keep `kinematic-ceiling`. The stop is measured over
the built reflection families, while automatic selection may score
higher-order reflections above it; a defaulted axis that then fails the audit
is rescored on `kinematic-ceiling` from the same trajectories rather than
refused. Automatic cases otherwise change identity once through their policy
payload.

## Amendment (2026-11-16, #1214)

The automatic resolution is now `sinc-oversampled`: the sinc-Nyquist step
divided by 8. The Nyquist step keeps a line's area but undersamples its node
peak and FWHM (HOPG 1 MeV: peak −31 %, FWHM +78 % at production population);
eight nodes per step keep both inside 1 % on every #1143 scene, and six or
seven do not at 1 MeV. Under the #1243 measured axis the finer step fits the
600 000-point budget on every measured scene. An axis over the budget refuses;
`sinc-nyquist` is the explicit opt-out. Feature-window policies keep
`sinc-nyquist`, whose backbone they refine themselves.
Bin-mean cases on the measured bandwidth default to `sinc-shape` instead
(2026-11-17): its spacing bounds an isolated line's bin-mean peak to 1 %, where
bin means at the ÷8 step are bounded only to 1.7 %. Their kinematic fallback
takes `sinc-oversampled`.

## Amendment (2026-11-17, #1257)

An over-budget `sinc-oversampled` axis no longer refuses. It is refined
locally: the ÷8 step inside merged windows around every measured line narrower
than eight maximum spacings (24 eV), and the 3 eV maximum spacing elsewhere. Each
window carries a halo, and a wider line's halo is refined whole wherever it
meets a window, so no spacing join falls inside any line's halo. Node
quadrature then misplaces at most 1e-4 of a line's mass per side
(`line-grid-node-local-refinement`). The Si 1 MeV kinematic axis drops from
2.58 M uniform nodes to 338 k; the measured four-family axis drops from 975 k
to 399 k. Where the uniform axis fits, nothing changes. Only a local axis that
also exceeds the budget refuses. Whether the local axis should become the
default everywhere is left open, pending the #1257 shape qualification.
