"""Headless CXR scan entry point -- thin ``python -m`` shim over cxr_mc.scan.

The remote box invokes ``python -m cxr_mc._entry.scan <material>`` inside its
uv-synced checkout. The real logic -- and the rationale for the __main__ guard
(spawn/forkserver re-import the entry module per worker) -- lives in cxr_mc.scan.
Prefer the installed CLI: ``cxr run [PROFILE] -m <material>``.
"""

from cxr_mc.materials import MaterialConfigError

try:
    from cxr_mc.scan import main
except MaterialConfigError as exc:
    raise SystemExit(str(exc)) from None

if __name__ == "__main__":
    raise SystemExit(main())
