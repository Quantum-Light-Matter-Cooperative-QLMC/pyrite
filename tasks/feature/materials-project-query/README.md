# Materials Project query

## Problem and scope

The repository has no maintained Materials Project query path, while catalog
work needs a reproducible way to retrieve source crystal metadata.  Define and
implement a small, explicit query workflow authenticated by a local, ignored
`.env` file.  Never commit API keys, query responses containing credentials, or
make package import depend on network access.

Scope: query configuration, an intentional command or developer entry point,
normalization of retrieved source data into the catalog-ingestion workflow, and
offline regression coverage.  This task does not add materials, turn runtime
simulation into an online service, or change the packaged catalog schema unless
the selected ingestion contract demonstrably requires it.

## Implementation path and likely owners

- `src/cxr_mc/materials/`: source-data retrieval/normalization boundary.
- `src/cxr_mc/cli/`: only if a user-facing `cxr` command is selected.
- `scripts/` or a dedicated developer-facing module: one-shot provenance
  retrieval, if that is the least invasive interface.
- `tests/`: mocked client and missing/invalid-key behavior; catalog fixtures
  remain offline.
- `docs/` and `.env.example`: non-secret setup, provenance, and failure
  recovery guidance.

## Checklist

1. Locate previous ad-hoc Materials Project access and identify the required
   query inputs/outputs for catalog work; record the chosen public interface.
2. Define `MP_API_KEY` loading from a local `.env` without logging the value;
   preserve explicit process-environment precedence and a clear missing-key
   error.
3. Add the minimal optional client dependency or isolated HTTP adapter only
   after selecting its supported API endpoint and response contract.
4. Implement deterministic retrieval/normalization with request identifiers,
   source URL/version, and retrieval-date provenance suitable for committing
   derived catalog inputs.
5. Add offline mocked tests for successful query, missing key, malformed
   response, and network/API failure.  Keep normal imports, catalog loading,
   and full offline tests network-free.
6. Add `.env.example` and user/developer documentation; regenerate CLI
   reference if a public `cxr` command is added.

## Decisions and open questions

- Default assumption: `MP_API_KEY` is local developer configuration, supplied
  through `.env` or the environment, never a repository setting.
- Decide whether this is a developer-only retrieval tool or a supported `cxr`
  command after locating intended callers.  Prefer developer-only if it only
  creates catalog source artifacts.
- Confirm query target(s), required fields, and provenance retention before
  choosing `mp-api` versus a direct API adapter.
- Do not perform live API calls in ordinary CI; optional credentialed checks
  must be explicit.

## Delegation slices and required skills

- Discovery/interface design: `repo-orientation`, `scientific-library`.
- CLI surface, if selected: `cli-ui-ux`, `documentation-maintenance`.
- Offline failure-path coverage: `regression-testing`.
- Material catalog output changes, if any: `regen-golden`,
  `physics-review`.

## Acceptance checks

- Missing credentials fail locally with actionable guidance and no secret
  exposure; environment override behavior is tested.
- Valid mocked API response becomes a validated, provenance-bearing result.
- API, malformed-response, and transport failures are deterministic and
  actionable.
- Default catalog imports and test suite make no network calls.
- `.env.example` contains no credential; `.gitignore` continues to exclude
  `.env`.
