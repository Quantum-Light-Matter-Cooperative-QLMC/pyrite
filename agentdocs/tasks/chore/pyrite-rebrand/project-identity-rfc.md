# Superseded RFC: PyRITE project identity and compatibility

This accepted matrix is a point-in-time task record recovered after the flat
`docs/` layout was retired. The active task and any supersession are in
[`README.md`](README.md); the durable high-level decision is
[`ADR-0007`](../../../../docs/adr/0007-project-identity.md).

- **Status:** Accepted — 2026-08-09
- **Decision record:** [ADR-0007](../../../../docs/adr/0007-project-identity.md)
- **Task record:** [`README.md`](README.md)
- **Related current reference:**
  [development workspace](../../../../docs/repo-design/development-workspace.md)

## Scope

This RFC fixes the names and compatibility rules for converting the project
from `cxr-mc` to **PyRITE**. It distinguishes visible brand from distribution,
import, command, environment, filesystem, and persisted protocol identities.
It is the approval boundary for the later implementation slices; it does not
itself move packages, add commands, migrate state, or rename the repository.

The conversion must not rename CXR when it denotes coherent X-ray radiation,
PXR/CBS terminology, equations, validation IDs, or literature. It must not
rewrite accepted historical records merely to eliminate the old spelling.

## Canonical identity and compatibility matrix

| Surface | Canonical identity | Compatibility identity | Policy |
|---|---|---|---|
| Visible brand | `PyRITE` | `cxr-mc` in historical and migration text only | Public current prose and UI move to `PyRITE` in slice F. Do not use `Pyrite` or `PYRITE` as the display brand. |
| Long form | `a Python toolkit for Radiation from Interactions and Transport of Electrons` | none | Use the exact capitalization shown. The article remains lowercase when it follows `PyRITE:`. |
| Short tagline | `Coherent X-ray radiation and electron transport in crystals.` | none | Exact sentence, including final period. |
| GitHub repository | `Quantum-Light-Matter-Cooperative-QLMC/pyrite` | `Quantum-Light-Matter-Cooperative-QLMC/cxr-mc` redirect | Rename only in authenticated slice G. Keep the old GitHub name unclaimed for as long as its redirect is part of compatibility. |
| Lowercase infrastructure stem | `pyrite` | `cxr-mc` only where compatibility or history requires it | New checkout, container, remote checkout, temporary, config, cache, and data names use `pyrite`. |
| Python distribution | `pyrite-xray` | no new `cxr-mc` compatibility distribution | PEP 503 normalized name is `pyrite-xray`; wheel/dist-info stem is `pyrite_xray`. Recheck and control the name before the first publication. Existing source installs named `cxr-mc` remain historical; slice C updates lock/build metadata. |
| Python import namespace | `pyrite` | `cxr_mc` compatibility namespace | Step I supersedes the original `cxr_mc`-canonical decision. The implementation lives only under `src/pyrite/`. A thin compatibility package maps old root, package, deep, and private imports to the identical canonical module objects; do not add a `pyrite_xray` facade. Retain `cxr_mc` indefinitely unless a later ADR defines a removal policy. |
| User command | `pyrite` | installed `cxr` compatibility executable | Add in 0.2.0. Normal `cxr` invocation warns once on stderr and names `pyrite`; completion-mode invocation is silent. The executable is omitted from canonical docs/examples and is removable no earlier than 0.4.0 after two published minor releases. Help, completion, exit, stream, and JSON behavior otherwise match. |
| Developer command | `pyrite-dev` | installed `cxr-dev` compatibility executable | Same 0.2.0 to 0.4.0 minimum window and normal-invocation warning. Contributor docs use `pyrite-dev` after slice F. |
| Shell completion protocol | `_PYRITE_COMPLETE` for `pyrite` | `_CXR_COMPLETE` through the `cxr` executable window | New managed blocks use `pyrite`; the completion remover recognizes both managed block forms. `_CXR_COMPLETE=... cxr` suppresses the rebrand warning so generated completion remains protocol-clean. |
| Environment prefix | `PYRITE_*` | corresponding `CXR_*` aliases | New names are canonical in 0.2.0. Old names have no scheduled removal because environment use is not discoverable; removal requires a later ADR and at least a two-minor warning window. |
| Config precedence | per-call > `PYRITE_*` > `CXR_*` > new store > legacy-store fallback > built-in | current `CXR_*` and `cxr-mc` store | Equal new/old environment values are accepted silently. Different values select `PYRITE_*` and warn once on stderr. An explicit per-call value suppresses the irrelevant conflict. |
| User config directory | Click platform app directory for `pyrite`; `config.toml` beneath it | Click platform app directory for `cxr-mc` | Read canonical first. If absent, read legacy. First write imports legacy state atomically, writes canonical, and leaves legacy recoverable. If both exist, canonical wins and the legacy store is not merged. |
| User cache directory | `platformdirs.user_cache_path("pyrite", appauthor=False)` | current `XDG_CACHE_HOME` or `~/.cache`, then `cxr-mc` | Add `platformdirs` as the shared resolver. Read canonical first, then the exact legacy root on a miss; write canonical only. Cache entries may be regenerated. |
| User data directory | `platformdirs.user_data_path("pyrite", appauthor=False)` | none currently | Reserve for future non-config application data; there is no legacy data root to migrate. Existing workspace artifacts are not user-data-directory data. |
| Workspace root selector | `PYRITE_HOME` | `CXR_HOME` | Precedence follows the environment rule. The selected directory is never renamed automatically. |
| Workspace artifacts | existing `checkpoints/`, `energy-grid-artifacts/`, profiles, archives, and explicit paths | all valid old layouts already accepted by readers | Keep names and formats. A repository rebrand is not authority to move user workspaces. |
| Persisted schema and wire IDs | existing `cxr.*`, `cxr.lock.json`, `CXR_REMOTE_V1`, and `CXR_REMOTE_FRAME_END` | same spellings are canonical protocol identifiers | Keep indefinitely. Readers and writers do not translate or rewrite these identifiers. |
| Python pickles | new writes use module paths under `pyrite` | existing paths under `cxr_mc` | Current releases load old `cxr_mc.*` pickles through the import compatibility layer, including ordinary `pickle.load`. Old releases are not required to load newly written `pyrite.*` pickles. Persisted schema IDs remain unchanged. |
| Release display | `PyRITE X.Y.Z` | prior release titles remain historical | Tags remain `vX.Y.Z`. Do not rewrite old releases. |
| Release files | `pyrite_xray-X.Y.Z` wheel/sdist stems; `pyrite:X.Y.Z` container tag; GitHub artifacts under repository `pyrite` | prior files remain immutable | Do not emit new release artifacts named `cxr-mc` after cutover. Publication remains outside this task. |

The command removal target is deliberately distinct from the pre-existing CLI
redesign aliases, which were deprecated in 0.1.0 and target 0.3.0. Rebranding
aliases begin their own window when `pyrite` ships in 0.2.0.

## Environment inventory and mapping

Every maintained environment variable below maps by replacing the leading
`CXR` with `PYRITE`. This rule preserves the remainder exactly; for example,
`CXR_MC_BACKEND` becomes `PYRITE_MC_BACKEND`, not `PYRITE_BACKEND`.

| Class | Legacy names accepted as aliases |
|---|---|
| User context and remote | `CXR_HOME`, `CXR_PROFILE`, `CXR_REMOTE_HOST`, `CXR_REMOTE_DIR`, `CXR_REMOTE_UV`, `CXR_REMOTE_GPU_VENDOR` |
| Backend and resources | `CXR_FP64`, `CXR_MC_BACKEND`, `CXR_MC_RESOURCE_POLICY`, `CXR_MC_SYCL_DEVICE`, `CXR_MC_TRANSPORT_CORE` |
| Advanced execution tuning | `CXR_MC_BREM_CHUNK`, `CXR_MC_SPEC_CHUNK`, `CXR_MC_SPEC_BUDGET_MB`, `CXR_MC_FREE_EVERY`, `CXR_MC_FREE_WATERMARK_MB`, `CXR_MC_GPU_OOM_RETRIES`, `CXR_MC_GPU_POOL_FRAC`, `CXR_MC_GPU_SHARE`, `CXR_MC_PIPELINE_WORKER_MEM_MB`, `CXR_MC_WORKER_MEM_MB`, `CXR_MC_TIMING`, `CXR_MC_NSYS`, `CXR_MC_NSYS_PYSTACK`, `CXR_MC_DEBUG` |
| App/process handoff | `CXR_ANALYZE_INITIAL`, `CXR_VIEWER_INITIAL`, `CXR_LOCAL_DASHBOARD` |
| Developer/test controls | `CXR_LOCAL_SWEEP_OK`, `CXR_ONLINE_TESTS`, `CXR_RUN_INTEL_SYCL_TESTS` |

App/process handoff and advanced tuning variables are not promoted to stable
user API merely by receiving aliases. New generated processes emit only the
`PYRITE_*` spelling; readers accept both under the common precedence rule.
Developer/test controls follow the same mapping so documented commands and
hooks have one identity.

`CXR_REMOTE_V1` and `CXR_REMOTE_FRAME_END` are not environment variables. They
are stable wire framing tokens and remain unchanged. `_CXR_COMPLETE` is a Click
shell-completion control covered separately in the matrix.

## Persisted data and directory behavior

The current mutable config is `config.toml` under
`click.get_app_dir("cxr-mc")` and contains `profile.current`, `remote.target`,
and `workspace.root`. The migration uses whole-store fallback, not per-key
merging:

1. Read the `pyrite` store when it exists.
2. Otherwise read the `cxr-mc` store without modifying it.
3. On the first state-changing command, copy the parsed legacy document into a
   temporary file under the canonical directory, apply the requested change,
   and atomically replace the canonical file.
4. Never delete or rewrite the legacy store automatically.

The same rule applies to non-regenerable app state such as saved initial
materials. It covers the current `analysis-default`, `viewer-default`,
`validation-defaults.json`, and `acp-servers.json` files under the legacy Click
app directory.

Regenerable cache entries use
`platformdirs.user_cache_path("pyrite", appauthor=False)` for canonical writes.
The viewer-render legacy fallback reproduces its current algorithm exactly:
`Path($XDG_CACHE_HOME)` when set, otherwise `Path.home() / ".cache"`, followed
by `cxr-mc/viewer-renders`. It reads canonical first and legacy only on a miss;
it never copies or deletes the legacy cache. There is no current separate user
data owner; future data uses
`platformdirs.user_data_path("pyrite", appauthor=False)`.

The Sphinx-only `tempfile.gettempdir()/cxr-mc-matplotlib` `MPLCONFIGDIR` is not
persisted user state. Slice F changes its temporary stem to `pyrite-matplotlib`
without fallback or migration.

Workspace and remote paths are values, not brands. A configured old checkout
such as `/home/.../cxr-mc` remains valid. The new default/example checkout stem
is `pyrite`, but no command silently moves local or remote directories.

The backend setup command's workspace `.env` file follows the environment
rule. It reads `PYRITE_MC_BACKEND` before `CXR_MC_BACKEND`. On an explicit
backend-setting write, it replaces the managed legacy key with the canonical
key while preserving unrelated lines; it does not otherwise rewrite `.env`.

The following persisted identities remain exact protocol names:

- `cxr.campaign-lock.v1`, `cxr.case-content-key.v1`,
  `cxr.case-manifest.v1`, `cxr.checkpoint-manifest.v2`, and
  `cxr.dataset-identity.v1`;
- `cxr.energy-grid-artifact.v1`, `cxr.energy-grid-gc-metadata.v1`,
  `cxr.performance.v1`, and `cxr.zhai-cache.v3`/`v4`;
- `cxr.lock.json` and every supported `cxr.*` JSON result schema;
- `CXR_REMOTE_V1`, `CXR_REMOTE_FRAME_END`, and existing `cxr.*` profiling/NVTX
  labels.

New schema families created after the rebrand use `pyrite.*`. Existing schema
families keep their `cxr.*` prefix when versioned; a branding change alone does
not justify a protocol-version bump.

## Inventory and ownership

The inventory used exact-text search for `cxr-mc`, `cxr_mc`, user-visible
`cxr`, `CXR_*`, `cxr.*`, GitHub URLs, app-directory calls, build names, and
release metadata. Symbol ownership was checked at the CLI, config/path, remote,
checkpoint, and developer-tool boundaries. The broad occurrence count makes a
global replacement unsafe: the same token may be public identity, import API,
wire protocol, scientific terminology, generated output, or history.

| Classification | Current owners and examples | Conversion rule |
|---|---|---|
| Public canonical | `README.md`, `docs/index.md`, Sphinx metadata, app chrome, CLI version banner | Change to PyRITE in F, after command/package work defines live behavior. |
| Packaging/import | `pyproject.toml`, `uv.lock`, `src/pyrite/`, the thin `src/cxr_mc/` compatibility bootstrap, package-smoke and API/export tests | Step I makes `pyrite` canonical, retains identity-preserving legacy imports, and verifies wheel and editable installs. |
| CLI contract | `pyproject.toml` scripts, `src/pyrite/cli/`, `src/pyrite/_dev.py`, completion, deprecation registry, CLI generators and snapshot | Add canonical commands and tested aliases in D; regenerate owned artifacts. |
| Config and mutable paths | `src/pyrite/paths.py`, `cli/_config.py`, app launchers, Plotly render cache | Implement dual-read/canonical-write migration in D/E. |
| Persisted/on-disk | checkpoint manifests, campaign locks, energy-grid artifacts, performance records, Zhai caches, remote frames | Preserve the listed identifiers and old valid artifacts in E. |
| Remote/generated commands | `src/pyrite/remote/` script and path builders | Emit canonical distribution/command/env names while accepting configured old checkout paths. |
| Generated documentation/data | CLI reference, CLI deprecations, repository-map generated region, API autosummary, CLI contract snapshot, lockfile | Change generators/owners first; regenerate only in their implementation slice. |
| Repository/release | GitHub URLs, clone instructions, Docker tag/examples, package metadata, CI and release settings | Prepare repository-contained changes in F; authenticated rename only in G. |
| Scientific/internal | coherent-X-ray terminology, CSS selectors, `cxr.*` instrumentation labels | Keep unless separately justified. |
| Historical | accepted RFC/ADR context, archived agent records, old release artifacts, migration examples | Keep and label by context; do not rewrite history. |

The Dockerfile also contains pre-existing stale references to the removed
`packages/cxr-mc-tests` workspace. That defect belongs to the later container
update; it is not evidence for retaining that distribution.

Maintained repository URL owners for slice F are: `README.md` (clone URL),
`docs/running-on-a-cluster.md` (cluster clone URL), `docs/index.md` (source
link), `src/cxr_mc/devtools/cli_reference.py` (generated-reference source URL),
`docs/repo_map.md` (README/docs/TODO links), and `docs/atomic-data-sources.md`
(raw-file link). Generated CLI references change through their owners.
Historical URLs in accepted decisions or archived agent records remain
historical unless they purport to be a current navigation link.

## External evidence and gates

Read-only authoritative checks on 2026-08-09 returned:

- [`pyrite` on PyPI](https://pypi.org/pypi/pyrite/json) exists (version 0.1,
  attributed by its metadata to Mark Ramm), so it is unavailable as this
  project's distribution;
- the PyPI JSON endpoints for
  [`pyrite-xray`](https://pypi.org/pypi/pyrite-xray/json) and
  [`cxr-mc`](https://pypi.org/pypi/cxr-mc/json) returned HTTP 404, so neither
  had a public project page at the time of the check;
- the anonymous GitHub repository API returned HTTP 404 for both the
  [proposed `pyrite` path](https://api.github.com/repos/Quantum-Light-Matter-Cooperative-QLMC/pyrite)
  and the
  [current private `cxr-mc` path](https://api.github.com/repos/Quantum-Light-Matter-Cooperative-QLMC/cxr-mc).
  Because the known current repository is private, the proposed 404 cannot
  exclude a private-name collision or prove rename authority.

HTTP 404 is availability evidence, not a reservation or ownership proof. Before
publication, an authenticated owner must confirm or create `pyrite-xray` and
recheck PyPI's normalized-name rules. Before slice G, an authenticated
organization owner must confirm the `pyrite` repository name and rename rights.
No package reservation, publication, repository rename, remote change, or push
is authorized by this RFC.

## Consequences

- Step I accepts the source namespace migration cost so repository, command,
  and import identities agree. Existing pickles remain readable; new pickles
  intentionally identify canonical `pyrite.*` owners.
- Users get a memorable `pyrite` command and deterministic environment/config
  migration without losing old workspaces or artifacts.
- Stable automation schemas retain `cxr.*`; visible product identity does not
  leak into protocol versioning.
- Distribution and repository cutovers remain gated on credentialed ownership
  checks. Repository redirects are part of compatibility, so `cxr-mc` must not
  be reused while they are required.
- Repository/package structure cleanup landed before implementation began, so
  the conversion proceeds against the regrouped owners inventoried here.
