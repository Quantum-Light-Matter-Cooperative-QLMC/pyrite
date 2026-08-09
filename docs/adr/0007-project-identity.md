# 0007 — PyRITE project identity

- **Status:** Accepted — 2026-08-09
- **Date:** 2026-08-09
- **Rationale:** [`docs/project-identity-rfc.md`](../project-identity-rfc.md)

## Context

The old `cxr-mc` identity is narrower than the project's radiation,
electron-transport, detector, materials, analysis, and validation scope. Its
distribution, import, commands, environment, paths, and persisted protocols
cannot be renamed safely as one text substitution.

## Decision

- Display **PyRITE**, expanded as **a Python toolkit for Radiation from
  Interactions and Transport of Electrons**, with the tagline **Coherent X-ray
  radiation and electron transport in crystals.**
- Use `pyrite` for the repository and user CLI, `pyrite-dev` for contributor
  tooling, and `pyrite-xray` for the distribution.
- Keep `cxr_mc` as the canonical public import and pickle namespace; do not add
  `pyrite` or `pyrite_xray` import facades.
- Keep `cxr` and `cxr-dev` as installed compatibility executables through at
  least 0.4.0 when the canonical commands are introduced in 0.2.0. Normal
  invocation warns; completion-mode invocation stays silent.
- Make `PYRITE_*` environment names canonical while retaining corresponding
  `CXR_*` aliases without a scheduled removal. Resolve per-call > `PYRITE_*` >
  `CXR_*` > canonical store > legacy-store fallback > built-in.
- Move mutable application state to Click's app directory named `pyrite` using
  canonical-first reads, legacy fallback, and atomic copy-on-first-write. Use
  `platformdirs` with app name `pyrite` for cache/data paths. Do not move
  workspaces automatically.
- Keep existing `cxr.*`, `cxr.lock.json`, remote-frame, checkpoint, manifest,
  cache, and pickle identifiers as stable protocols. New schema families use
  `pyrite.*`.
- Name new release displays/artifacts for PyRITE, while leaving prior artifacts
  immutable. Do not publish or rename external resources until authenticated
  owners confirm availability and authority.

## Consequences

Brand, install name, import name, and protocol prefixes intentionally differ.
Later implementation must test both canonical and retained compatibility
surfaces. The old GitHub name remains reserved for redirects, and current
private-repository visibility means the proposed repository name still needs
credentialed confirmation.
