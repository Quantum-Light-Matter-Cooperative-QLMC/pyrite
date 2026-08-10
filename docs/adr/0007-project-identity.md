# 0007 — PyRITE project identity and compatibility

- **Status:** Accepted — 2026-08-09
- **Date:** 2026-08-09

## Context

The `cxr-mc` name no longer reflects the project's broader scope across radiation, electron transport, materials, detectors, analysis, and validation. However, the old identity also appears in public Python imports, commands, environment variables, persisted data, wire protocols, workspaces, and scientific terminology, so a global rename would create unnecessary compatibility breakage.

## Decision

The visible project identity is **PyRITE**, expanded as **a Python toolkit for Radiation from Interactions and Transport of Electrons**, with the tagline **Coherent X-ray radiation and electron transport in crystals.**

Use `pyrite` for the repository and user command, `pyrite-dev` for contributor tooling, and `pyrite-xray` for the Python distribution. Keep `cxr_mc` indefinitely as the canonical public import and pickle namespace.

`PYRITE_*` environment variables and `pyrite` application-state locations become canonical while corresponding legacy `CXR_*` names and state remain readable according to the project's compatibility rules. Existing user workspaces are not renamed automatically.

Existing persisted `cxr.*`, `cxr.lock.json`, remote-frame, checkpoint, manifest, cache, and related protocol identifiers remain stable; new schema families use `pyrite.*`. Historical uses of `cxr-mc` and scientific uses of CXR/PXR terminology are not rewritten merely for branding.

The `cxr` and `cxr-dev` executables remain compatibility aliases through their defined deprecation window. External repository or package-name changes require authenticated confirmation before execution.

## Consequences

Brand, distribution, import, command, environment, and protocol identities intentionally do not all share one spelling. This avoids a costly Python namespace and persisted-data migration while providing a consistent new public identity.

Compatibility beh