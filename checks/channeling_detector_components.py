"""Baseline: what a planar channeling case puts into physical detectors (#1148).

One planar case (default: the admitted ``channeling_demo`` profile, Si(1-10),
17 MeV, 2 um) is transported once with the channeling zone and once without
it (same seed, amorphous). Every intrinsic component the detector records is
then scored at the profile's forward direction and at the Bragg-mirror
direction ``n = v - 2 (v.g) g / g^2`` of each reflection in the case that is
off the plane's own row (own-row reflections give no Bragg-angle line with
the beam in the plane):

- ``spec``: native PXR/CBS lines (own row already dropped on zone rows, #1055);
- ``spec_characteristic``: EEDL shell-ionization lines;
- ``brem``: incoherent bremsstrahlung on the line grid;
- ``spec_channeling``: the expected-value CR source (rows play no part).

The channeling-on transport is scored on all rows, on zone rows only and on
native rows only, so the zone's share of each component is explicit. Native PXR/CBS on zone
rows uses a plane-wave path (straight mean segment, no angular spread);
incoherent brem and characteristic use the planar flux weights (#1148).
This check separates the detector contributions under those approximations. It is a report, not a gate: exit 0 when every number is
finite and the zone/native split adds up to the full-row score (incoherent
sums are additive over rows), 1 otherwise.

    PYRITE_MC_BACKEND=cpu uv run python checks/channeling_detector_components.py \
        --electrons 50 --json scratch/channeling_detector_components.json

Band integrals are photons/sr per incident electron over the case line grid.

Evidence for the #1148 baseline; no ledger claim yet.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np

COMPONENTS = ("spec", "spec_characteristic", "brem", "spec_channeling")


def _profile_case(profile: str, material: str):
    from pyrite import api
    from pyrite.campaign import profiles
    from pyrite.campaign.config import default_settings, material_sweep

    sweep = material_sweep(material, catalog_profile=profile)
    settings = profiles.profile_run_settings(default_settings(), profile, sweep=sweep)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cases = api.build_configured_cases(sweep, settings)
    if len(cases) != 1:
        raise SystemExit(f"profile {profile!r} lowers to {len(cases)} cases; expected one")
    case = cases[0]
    return dict(case.to_dict() if hasattr(case, "to_dict") else case)


def _bragg_directions(case, beam, e_lo, e_hi):
    """Bragg-mirror directions of off-own-row reflections with a line in band."""
    from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG, reciprocal_g_vector
    from pyrite.montecarlo.channeling.axial_case import _rotation
    from pyrite.montecarlo.runner.planar_zone_lines import is_own_row

    plane = case["channeling"]["plane_hkl"]
    rotation = _rotation(case)
    lattice = CRYSTALS[case["crystal"]]["lattice"]
    out = []
    for hkl in case["hkl_list"]:
        if is_own_row(hkl, plane):
            continue
        g, _ = reciprocal_g_vector(hkl, lattice)
        g = np.asarray(rotation @ g if rotation is not None else g, dtype=float)
        n = beam - 2.0 * (beam @ g) * g / (g @ g)
        n /= np.linalg.norm(n)
        denominator = 1.0 - beam @ n
        if denominator <= 0.0 or not np.isfinite(denominator):
            continue
        energy = HBARC_EV_ANG * abs(beam @ g) / denominator
        if e_lo <= energy <= e_hi:
            label = "(" + " ".join(str(int(h)) for h in hkl) + ")"
            out.append((label, n, float(energy)))
    return out


def _subset(segs, keep):
    from pyrite.montecarlo.runner.planar_zone_lines import _rows

    return _rows(segs, np.flatnonzero(keep))


def _band(spectrum, grid):
    return float(np.trapezoid(np.asarray(spectrum, dtype=float), grid))


def _score(runner, case, transport, n_hat):
    out = runner._spectrum_case(case, {**transport, "n_hat": n_hat})
    grid = np.asarray(out["E_grid"], dtype=float)
    return {key: _band(out[key], grid) if out.get(key) is not None else None for key in COMPONENTS}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="channeling_demo")
    parser.add_argument("--material", default="silicon")
    parser.add_argument("--electrons", type=int, default=50, help="Ne = Ne_brem")
    parser.add_argument("--max-bragg", type=int, default=4, help="Bragg directions kept")
    parser.add_argument("--json", default=None, help="write the record here")
    parser.add_argument(
        "--node-sampling",
        action="store_true",
        help="force node-sampled lines (aliases sub-eV zone lines on coarse grids)",
    )
    args = parser.parse_args(argv)

    from pyrite.montecarlo import runner
    from pyrite.montecarlo.geometry import tilted_geometry

    case = _profile_case(args.profile, args.material)
    if case.get("channeling") is None or case["channeling"].get("model") != "planar":
        raise SystemExit(f"profile {args.profile!r} is not a planar channeling case")
    case.update(Ne=args.electrons, Ne_brem=args.electrons)
    if args.node_sampling:
        case.pop("line_quadrature", None)
    else:
        # A straight 2 um zone row radiates a sub-eV line; node sampling on a
        # 50 eV grid aliases it, so band integrals use exact bin masses.
        case["line_quadrature"] = "bin-mean"
    off = {key: value for key, value in case.items() if key != "channeling"}

    on_t = runner._transport_case(case, transport_core="per-electron")
    off_t = runner._transport_case(off, transport_core="per-electron")
    zone = np.asarray(on_t["segs"]["channeling_zone"], dtype=bool)
    grid = np.asarray(on_t["E_grid"], dtype=float)
    beam, _ = tilted_geometry(
        case["theta_obs_rad"], np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )  # fmt: skip
    beam = np.asarray(beam, dtype=float)
    directions = [("forward", np.asarray(on_t["n_hat"], dtype=float), None)]
    bragg = _bragg_directions(case, beam, float(grid[0]), float(grid[-1]))
    bragg.sort(key=lambda item: -item[2])
    kept: list[float] = []
    gamma = 1.0 + float(case["E0_keV"]) / 510.99895
    for label, n_hat, energy in bragg:
        if len(kept) >= args.max_bragg or any(abs(energy - e) < 1.0 for e in kept):
            continue  # symmetry-equivalent reflection
        kept.append(energy)
        directions.append((label, n_hat, energy))
        # PXR vanishes on the exact mirror direction and peaks ~1/gamma off it;
        # probe the peak in and out of the diffraction plane.
        normal = np.cross(beam, n_hat)
        normal /= np.linalg.norm(normal)
        in_plane = np.cross(normal, n_hat)
        for tag, axis in (("+in", in_plane), ("+out", normal)):
            probe = n_hat + axis / gamma
            directions.append((label + tag, probe / np.linalg.norm(probe), energy))

    zone_t = {**on_t, "segs": _subset(on_t["segs"], zone)}
    native_t = {**on_t, "segs": _subset(on_t["segs"], ~zone)}
    record = {
        "profile": args.profile,
        "electrons": args.electrons,
        "line_quadrature": case.get("line_quadrature", "node"),
        "zone_rows": int(zone.sum()),
        "rows": int(zone.size),
        "zone_path_fraction": float(
            np.asarray(on_t["segs"]["L_ang"])[zone].sum() / np.asarray(on_t["segs"]["L_ang"]).sum()
        ),
        "fates": on_t["segs"]["channeling"]["fates"],
        "directions": [],
    }
    ok = True
    for label, n_hat, energy in directions:
        row = {
            "label": label,
            "n_hat": [float(v) for v in n_hat],
            "bragg_eV": energy,
            "on_all": _score(runner, case, on_t, n_hat),
            "on_zone": _score(runner, case, zone_t, n_hat),
            "on_native": _score(runner, case, native_t, n_hat),
            "off": _score(runner, off, off_t, n_hat),
        }
        for key in ("spec", "spec_characteristic", "brem"):
            total, part_z, part_n = (row[arm][key] for arm in ("on_all", "on_zone", "on_native"))
            finite = all(np.isfinite(v) for v in (total, part_z, part_n, row["off"][key]))
            additive = np.isclose(part_z + part_n, total, rtol=1e-9, atol=1e-30)
            ok &= bool(finite and additive)
        record["directions"].append(row)

    print(
        f"{args.profile}: Ne={args.electrons}, zone rows {record['zone_rows']}/{record['rows']}, "
        f"zone path fraction {record['zone_path_fraction']:.3f}, fates {record['fates']}"
    )
    header = (
        f"{'direction':>14} {'component':>20} {'off':>11} {'on':>11} {'on/off':>7} {'zone%':>6}"
    )
    print(header)
    for row in record["directions"]:
        name = row["label"] + ("" if row["bragg_eV"] is None else f"@{row['bragg_eV']:.0f}eV")
        for key in COMPONENTS:
            on, off_v, zone_v = row["on_all"][key], row["off"][key], row["on_zone"][key]
            if on is None:
                continue
            ratio = on / off_v if off_v else float("nan")
            share = 100.0 * zone_v / on if on and zone_v is not None else float("nan")
            off_text = "-" if off_v is None else f"{off_v:11.3e}"
            print(f"{name:>14} {key:>20} {off_text:>11} {on:11.3e} {ratio:7.3f} {share:6.1f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
