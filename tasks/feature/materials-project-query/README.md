# Materials Project query

## Problem and scope

The existing live external-database validation path calls
`crystals.Crystal.from_mp(mp_id, api_key=...)` from
`tests/external_db_fixtures.py`.  It reads `MP_API_KEY` only from the process
environment, so a local ignored `.env` is not loaded; MP-only records skip when
the shell has no exported key, and credentialed runs cannot reliably exercise
the intended query.  Repair this established validation/fixture workflow with
secure local `.env` loading.  Never commit API keys, query responses containing
credentials, or make package import depend on network access.

Scope: test-fixture and `scripts/refresh_external_cif.py` configuration,
deterministic live-query diagnostics, and offline regression coverage.  This
task does not add materials, introduce a new public `cxr` command, turn runtime
simulation into an online service, or change the packaged catalog schema.

## Implementation path and likely owners

- `tests/external_db_fixtures.py`: `MP_API_KEY` resolution and MP query error
  classification.
- `tests/test_crystal_external_db.py`: online-test gate and explicit
  credential/query failure behavior.
- `scripts/refresh_external_cif.py`: reuse fixture resolution, preserve cached
  MP data when unavailable, and report actionable diagnostics.
- `tests/`: mocked key-resolution and `Crystal.from_mp` failure coverage;
  catalog fixtures remain offline.
- `docs/` and `.env.example`: non-secret setup and recovery guidance.

## Checklist

1. Reproduce the credentialed MP-only online-test failure with an explicit key
   in a local `.env`; capture the `Crystal.from_mp` exception/API behavior
   without disclosing the key.
2. Add a shared local key resolver that loads `.env` for this developer/test
   path only, gives exported `MP_API_KEY` precedence, and never logs its value.
3. Preserve the current `CXR_ONLINE_TESTS=1` opt-in.  Distinguish unavailable
   credentials (skip) from a configured-key query/API failure (fail with an
   actionable message), so broken MP access cannot silently skip.
4. Route `scripts/refresh_external_cif.py` through the same resolver and retain
   cached entries on unavailable credentials or failed fetches.
5. Add offline mocked tests for `.env` loading, environment precedence,
   missing key, and `Crystal.from_mp` failure. Keep default catalog imports and
   ordinary tests network-free.
6. Add `.env.example` and developer documentation; do not add a new `cxr`
   command or an API client dependency unless reproduction proves `crystals`
   itself cannot support the query.

## Decisions and open questions

- `MP_API_KEY` is local developer configuration, supplied through `.env` or the
  environment, never a repository setting. Exported environment wins.
- Existing owner is test/fixture tooling, not runtime material loading or the
  public CLI. Do not widen scope absent reproduction evidence.
- `crystals>=1.7,<2` already supplies `Crystal.from_mp`; do not add `mp-api`
  until a reproduced API incompatibility justifies replacement.
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
