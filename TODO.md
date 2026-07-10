# TODO — feature/elsepa-port

This branch carries one backlog item; the full triaged backlog lives on `main`.
(Supersedes the older `feature/elsepa-transport`, which predates the
`cxr_model`→`cxr_mc` rename + the `montecarlo` package split.)

## pyelsepa / ELSEPA transport (P2)

*EVALUATED; adapter landed, image now buildable, validation reproduced.*

Evaluate replacing the hardcoded NIST Mott transport tables in
`src/cxr_mc/data/mott_transport_cross_sections/` with on-demand ELSEPA
([github.com/eScatter/pyelsepa](https://github.com/eScatter/pyelsepa), checked out at
`C:\dev\pyelsepa`). NB: *electron*-scattering data — separate from the xraydb (photon)
migration; xraydb cannot supply it.

**Done:**
- `dev/elsepa_tables.py` — drop-in adapter (writer/reader/compare are pure +
  unit-tested against the real `_load_mott_transport`; a gated
  `elsepa_transport_cross_section` driver runs ELSEPA via pyelsepa).
- Build is now **tarball-free**: `pyelsepa/docker/Dockerfile.modern` clones
  `github.com/eScatter/elsepa` (Apache-2.0) instead of the paywalled
  `adus_v1_0.tar.gz`; isolated venv at `C:/dev/pyelsepa/elsepa-venv` built + patched.
- **Validation reproduced** vs NIST: C (Z=6) 2.19% max rel, Si (Z=14) 4.42% max rel.
  Full recipe + results in [`docs/elsepa-transport.md`](docs/elsepa-transport.md).

**Remaining (why this branch stays open):**
- The `elsepa` image/venv live outside the repo (`C:/dev/pyelsepa`) and can't be
  reproduced in CI — the driver stays gated behind an actionable error; the NIST
  tables + analytic screened-Rutherford fallback remain in force at runtime.
- Extend table regeneration to elements with no NIST table (e.g. W, currently on the
  SR fallback).
- Decide whether to wire the adapter in at all, vs. keep it as offline tooling.

NB: pyelsepa depends on **Pint**, worth weighing if a units library is ever evaluated for
`cxr_mc`. (The old `claude_WIP.txt` scratchpad bullet is dropped — the file is not tracked on
this branch, so there is nothing to strip before merge.)
