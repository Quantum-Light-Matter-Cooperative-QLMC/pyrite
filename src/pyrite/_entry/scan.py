"""Headless CXR scan entry point -- thin ``python -m`` shim over pyrite.runs.scan.

The remote box invokes ``python -m pyrite._entry.scan <material>`` inside its
uv-synced checkout. The real logic -- and the rationale for the __main__ guard
(spawn/forkserver re-import the entry module per worker) -- lives in pyrite.runs.scan.
Prefer the installed CLI: ``cxr run [PROFILE] -m <material>``.
"""

from pyrite.materials import MaterialConfigError

try:
    from pyrite.runs.scan import main
except MaterialConfigError as exc:
    raise SystemExit(str(exc)) from None

if __name__ == "__main__":
    raise SystemExit(main())
