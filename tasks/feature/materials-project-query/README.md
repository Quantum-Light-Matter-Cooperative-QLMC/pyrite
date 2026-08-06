# Materials Project query

## Problem and scope

The existing live external-database validation path calls
`crystals.Crystal.from_mp(mp_id, api_key=...)` from
`tests/external_db_fixtures.py`.  Local `.env` loading is incomplete, but that
is secondary: `crystals` 1.7 constructs the legacy
`https://materialsproject.org/rest/v2/materials/<id>/vasp/cif` request.  A
configured key reaches that request, which returns 403 (not authorized for the
operation).  Replace this unsupported query boundary with a currently supported
Materials Project retrieval contract; retain secure local `.env` loading for
the developer/test path.  Never commit API keys, query responses containing
credentials, or make package import depend on network access.

Scope: test-fixture and `scripts/refresh_external_cif.py` configuration,
deterministic live-query diagnostics, and offline regression coverage.  This
task does not add materials, introduce a new public `cxr` command, turn runtime
simulation into an online service, or change the packaged catalog schema.

## Implementation path and likely owners

- `tests/external_db_fixtures.py`: `MP_API_KEY` resolution, supported MP
  retrieval adapter, and query error classification.
- `tests/materials/test_crystal_external_db.py`: online-test gate and explicit
  credential/query failure behavior.
- `scripts/refresh_external_cif.py`: reuse fixture resolution, preserve cached
  MP data when unavailable, and report actionable diagnostics.
- `tests/`: mocked key-resolution and `Crystal.from_mp` failure coverage;
  catalog fixtures remain offline.
- `docs/` and `.env.example`: non-secret setup and recovery guidance.

## Checklist

1. Reproduce the credentialed MP-only online-test failure with an explicit key
   in a local `.env`; record the sanitized status/endpoint behavior.  Confirm
   that the `crystals` 1.7 legacy REST-v2 request, rather than key discovery,
   is the primary failure.
2. Add a shared local key resolver that loads `.env` for this developer/test
   path only, gives exported `MP_API_KEY` precedence, and never logs its value.
3. Select and implement a supported MP structure/query endpoint and minimal
   client/HTTP adapter.  Do not retain `Crystal.from_mp` for live MP retrieval
   unless a credentialed reproduction proves it supports the selected contract.
4. Preserve the current `CXR_ONLINE_TESTS=1` opt-in.  Distinguish unavailable
   credentials (skip) from a configured-key query/API failure (fail with an
   actionable message), so broken MP access cannot silently skip.
5. Route `scripts/refresh_external_cif.py` through the same resolver and retain
   cached entries on unavailable credentials or failed fetches.
6. Add offline mocked tests for `.env` loading, environment precedence,
   missing key, supported-adapter response parsing, and authorization/API
   failure. Keep default catalog imports and ordinary tests network-free.
7. Add `.env.example` and developer documentation; do not add a new `cxr`
   command.  Add only the dependency justified by the chosen supported
   endpoint.

## Decisions and open questions

- `MP_API_KEY` is local developer configuration, supplied through `.env` or the
  environment, never a repository setting. Exported environment wins.
- Existing owner is test/fixture tooling, not runtime material loading or the
  public CLI. Do not widen scope absent reproduction evidence.
- The checked installed `crystals` 1.7 source sends its MP request to
  `/rest/v2/materials/<id>/vasp/cif`; the observed credentialed 403 makes that
  adapter unsuitable unless direct reproduction disproves it.  Choose a
  supported endpoint/client before implementation; do not guess payload shape.
- Do not perform live API calls in ordinary CI; credentialed checks remain
  explicitly gated by `CXR_ONLINE_TESTS=1`.

## Delegation slices and required skills

- Fixture/query diagnosis: `investigating-changes`, `repo-orientation`.
- Offline failure-path coverage: `regression-testing`.
- Documentation and non-secret env template: `documentation-maintenance`.

## Acceptance checks

- A key in local `.env` is used by MP-only online tests and the refresh script;
  exported environment wins and no secret appears in output.
- Missing key skips only MP-only live cases; a configured-key query failure is
  surfaced as a deterministic actionable failure.
- Default catalog imports and ordinary tests make no network calls.
- `.env.example` contains no credential; `.gitignore` continues to exclude
  `.env`.
