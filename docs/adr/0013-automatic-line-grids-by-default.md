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
