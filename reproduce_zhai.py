"""Headless Zhai/supplementary Monte-Carlo cache populator -- thin shim to
checks/anchor_figures.py::reproduce_all.

Kept at the repo root so cxr_mc.remote (which runs ``python
reproduce_zhai.py`` on the GPU box) and muscle-memory ``python
reproduce_zhai.py`` keep working from a checkout with no install. Mirrors
scan.py's shape: sys.path shim + thin __main__ guard, real logic lives in
checks/. Populates checkpoints/zhai_reproduction/ with every cache the
validation app's Zhai sections can hit -- no figures, no display -- so a
later ``cxr remote check --pull`` (or a plain local ``cxr check``) sees an
instant cache hit.

Run (defaults match the validation app's own UI defaults, so a pulled cache
is guaranteed to hit locally):

    python reproduce_zhai.py
    python reproduce_zhai.py --ne 20000 --ne-brem 200 --ne-supp 200 --refresh
"""

import argparse
import os
import sys

_CHECKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checks")
if _CHECKS not in sys.path:
    sys.path.insert(0, _CHECKS)

from anchor_figures import reproduce_all  # noqa: E402  # type: ignore[reportMissingImports]


def main(argv=None):
    ap = argparse.ArgumentParser(description="populate the Zhai reproduction cache")
    ap.add_argument(
        "--ne", type=int, default=20_000, help="Fig.1c anchor line electrons per energy"
    )
    ap.add_argument(
        "--ne-brem",
        type=int,
        default=200,
        help="Fig.1c anchor bremsstrahlung electrons per energy",
    )
    ap.add_argument(
        "--ne-supp",
        type=int,
        default=200,
        help="supplementary electrons per polar-tilt spectrum",
    )
    ap.add_argument(
        "--tmd-azimuth",
        type=float,
        default=0.0,
        help="exploratory azimuth in degrees for TMD studies whose azimuth is unreported",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="recompute even if a matching cache already exists",
    )
    ap.add_argument("--cache-dir", default=None, help="override the cache directory (testing)")
    args = ap.parse_args(argv)

    results = reproduce_all(
        ne=args.ne,
        ne_brem=args.ne_brem,
        ne_supp=args.ne_supp,
        tmd_exploratory_azimuth_deg=args.tmd_azimuth,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
    )
    for label, path, cache_hit in results:
        status = "cached" if cache_hit else "computed"
        print(f"{label:16s} {status:9s} {path}")


if __name__ == "__main__":
    main()
