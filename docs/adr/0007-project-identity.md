# 0007 — PyRITE project identity and compatibility

- **Status:** Accepted — 2026-08-09; compatibility retired in 0.3.0
- **Date:** 2026-08-09

## Context

The `cxr-mc` name no longer reflects the project's broader scope across radiation, electron transport, materials, detectors, analysis, and validation. However, the old identity also appears in public Python imports, commands, environment variables, persisted data, wire protocols, workspaces, and scientific terminology, so a global rename would create unnecessary compatibility breakage.

## Decision

The visible project identity is **PyRITE**, expanded as **a Python toolkit for Radiation from Interactions and Transport of Electrons**, with the tagline **Coherent X-ray radiation and electron transport in crystals.**

Use `pyrite` for the repository, user command, and Python import and implementation namespace; use `pyrite-dev` for contributor tooling and `pyrite-mc` for the Python distribution. The implementation lives under `src/pyrite/`.

New pickles identify their `pyrite.*` module owners. Version 0.3.0 removes the transitional `cxr_mc` import namespace; legacy pickles naming that package require migration with an earlier release.

`PYRITE_*` environment variables and `pyrite` application-state locations are the only supported runtime identities from version 0.3.0. Existing user workspaces are not renamed automatically.

Existing persisted `cxr.*`, `cxr.lock.json`, remote-frame, checkpoint, manifest, cache, and related protocol identifiers remain stable; new schema families use `pyrite.*`. Historical uses of `cxr-mc` and scientific uses of CXR/PXR terminology are not rewritten merely for branding.

Version 0.3.0 removes the transitional `cxr` and `cxr-dev` executables. External repository or package-name changes require authenticated confirmation before execution.

## Consequences

Repository, command, import, environment, and mutable-state identities now share the `pyrite` spelling. Persisted schema and wire identities intentionally remain unchanged so existing datasets and remote job records remain readable.
