"""Thin Feranchuk-versus-Zhai diagnostic over the canonical anchor pipeline.

The maintained configuration, current ``BeamSpec``/``Sweep`` case construction,
detector response, and versioned cache live in :mod:`anchor_figures`. This
script retains the useful analytic-versus-transport count comparison without a
second set of geometry constants or Monte-Carlo calls.
"""

import argparse

from tabulate import tabulate

from pyrite.validation.anchor_figures import (
    ZhaiAnchor,
    ZhaiCacheMiss,
    cached_model_spectra,
    feranchuk_line_flux,
)


def comparison_rows(anchor: ZhaiAnchor, model: dict) -> tuple[list[list], list[list]]:
    """Return supported line-count rows and analytic/Monte-Carlo ratios."""
    rows: list[list] = []
    ratios: list[list] = []
    targets = (
        (anchor.thick_film_ang, "29 nm film", model["film"]),
        (anchor.thick_bulk_ang, "1 mm bulk", model[anchor.energies_keV[-1]]),
    )
    for thickness_ang, label, record in targets:
        analytic = (
            feranchuk_line_flux(
                anchor,
                anchor.energies_keV[-1],
                thickness_ang,
            )
            * anchor.per_nA
        )
        line_counts = record["line_flux_per_e"] * anchor.per_nA
        note = "" if thickness_ang < 1e4 else "outside Eq.(6) validity"
        rows.extend(
            [
                [
                    "A idealized Feranchuk",
                    label,
                    analytic,
                    note,
                ],
                [
                    "B Monte Carlo (Zhai)",
                    label,
                    line_counts,
                    "",
                ],
            ]
        )
        ratios.append(
            [
                label,
                analytic / line_counts,
                (
                    "should be ~1 (idealized model valid here)"
                    if thickness_ang < 1e4
                    else "idealized bulk extrapolation overshoots"
                ),
            ]
        )
    return rows, ratios


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="compare canonical Zhai Monte Carlo against Feranchuk Eq. 12"
    )
    parser.add_argument("--ne", type=int, default=20_000)
    parser.add_argument("--ne-brem", type=int, default=200)
    args = parser.parse_args(argv)

    anchor = ZhaiAnchor()
    try:
        model, cache_hit, cache_path = cached_model_spectra(
            anchor,
            ne=args.ne,
            ne_brem=args.ne_brem,
            cache_only=True,
        )
    except ZhaiCacheMiss as exc:
        parser.exit(75, f"{exc}\n")
    rows, ratios = comparison_rows(anchor, model)
    detector = anchor.detector
    print(
        tabulate(
            [
                ["crystal / reflection", "graphite (002), beam || c-axis"],
                ["beam energy", f"{anchor.energies_keV[-1]:.1f} keV"],
                [
                    "detector",
                    f"theta_obs = {detector.observation_angle_deg:g} deg, "
                    f"full polar span = {detector.polar_acceptance_deg:g} deg, "
                    f"{detector.solid_angle_sr:g} sr",
                ],
                ["cache", f"{'hit' if cache_hit else 'computed'}: {cache_path}"],
            ],
            headers=["setup", "value"],
            tablefmt="github",
        )
    )
    print()
    print(
        tabulate(
            rows,
            headers=[
                "model",
                "target",
                "line\n[cts/s/nA]",
                "note",
            ],
            tablefmt="github",
            floatfmt=(None, None, ".1f", None),
        )
    )
    print()
    print(
        tabulate(
            ratios,
            headers=["target", "A/B line counts", "interpretation"],
            tablefmt="github",
            floatfmt=(None, ".2f"),
        )
    )


if __name__ == "__main__":
    main()
