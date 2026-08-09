"""Headless Zhai/supplementary Monte-Carlo cache populator -- thin ``python -m``
shim over src/cxr_mc/apps/anchor_figures.py::reproduce_all.

The remote box invokes ``python -m cxr_mc._entry.reproduce_zhai`` inside its
uv-synced checkout. Populates checkpoints/zhai_reproduction/ for every cache the
validation app's Zhai sections hit -- no figures, no display -- so a later ``cxr
remote pull --preset zhai`` (or plain local ``cxr check``) sees an instant cache hit.

Run (defaults match the app's own UI defaults, so the pulled cache is guaranteed
to hit locally):

    python -m cxr_mc._entry.reproduce_zhai
    python -m cxr_mc._entry.reproduce_zhai --ne 20000 --ne-brem 200 --ne-supp 200 --refresh
"""

import argparse

from ..apps.anchor_figures import reproduce_all


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
        help="TMD exploratory azimuth (deg)",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="recompute even when a cache hit exists",
    )
    ap.add_argument("--cache-dir", default=None, help="override cache directory")
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
