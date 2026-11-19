"""Formerly pinned family versus automatic selection on one profile case (#1302).

Usage:
  reference_pins_1302.py run MATERIAL E0_KEV TILT_DEG HISTORIES H,K,L OUT_JSON
  reference_pins_1302.py table OUT_DIR

``run`` resolves one case of the ``PROFILE`` catalog profile (default
``cleo_sub_1302``; profile, beam and detector come from the user config
directory, so remote jobs set ``XDG_CONFIG_HOME`` to a shipped copy) as
``pyrite run`` does, fixes its population at ``HISTORIES`` and runs it once
through ``run_cases``. The catalog crystal must be unpinned, so the spectrum
phase runs the production automatic selection. At that selection the probe
also evaluates the same in-band yield callable on the formerly pinned family
``{+H,K,L; -H,K,L}`` alone (and its lattice-share continuum on net-metric
radiators). The production pass is skipped (``PROBE_PRODUCTION=1`` keeps it).

Recorded: the selected cutoff, reflection count, tail, convergence and shell
table; the pinned family's line yield and net yield; and their shares of the
kept selected total. The pinned-only run scored exactly that family, so its
share is the fraction of the automatic line yield the old pin retained.

Remote only (GPU, multi-minute selections). Engineering evidence for
``reflection-yield-tolerance-selector``; no accuracy verdict.
"""

import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PROFILE", "cleo_sub_1302")
os.environ.setdefault("PROBE_PRODUCTION", "0")
PROBE = runpy.run_path(str(Path(__file__).with_name("reflection_selection_subsample_1281.py")))
PRODUCTION = os.environ["PROBE_PRODUCTION"] != "0"


def instrument(family, records):
    """Evaluate ``family`` with each production selection's own yield callables."""
    from pyrite.montecarlo.runner import reflection_cutoff as rc

    select = rc.select_reflection_cutoff
    production = rc._selected_spectrum
    mates = [tuple(family), tuple(-int(i) for i in family)]

    def wrapped(in_band_yield, shells, tolerance, *, share_yield=None, background=0.0):
        t0 = time.perf_counter()
        result = select(
            in_band_yield, shells, tolerance, share_yield=share_yield, background=background
        )
        selection_s = time.perf_counter() - t0
        pinned_lines = float(in_band_yield(list(mates)))
        pinned_share = None if share_yield is None else float(share_yield(list(mates)))
        kept_lines = float(sum(row[2] for row in result.shells))
        kept_shares = float(sum(result.shares)) if result.shares else None
        record = dict(
            selection_s=selection_s,
            cutoff_invang=result.cutoff_invang,
            reflections=len(result.hkl_list),
            digest=PROBE["_digest"](result.hkl_list),
            tail_fraction=result.tail_fraction,
            converged=result.converged,
            shells=[list(row) for row in result.shells],
            shares=list(result.shares),
            background=result.background,
            family=mates,
            family_in_selected=all(m in {tuple(h) for h in result.hkl_list} for m in mates),
            family_lines=pinned_lines,
            family_share=pinned_share,
            kept_lines=kept_lines,
            family_line_fraction=pinned_lines / kept_lines if kept_lines > 0 else None,
        )
        if pinned_share is not None and kept_shares is not None:
            kept_net = kept_lines - kept_shares
            record.update(
                family_net=pinned_lines - pinned_share,
                kept_net=kept_net,
                family_net_fraction=(pinned_lines - pinned_share) / kept_net
                if kept_net != 0
                else None,
            )
        records.append(record)
        return result

    def skipped(case, evaluate, grid):
        return production(case, evaluate, grid) if PRODUCTION else np.zeros(np.shape(grid))

    rc.select_reflection_cutoff = wrapped
    rc._selected_spectrum = skipped


def run(material, e0_kev, tilt_deg, histories, family, out_json):
    from pyrite.materials import CATALOG
    from pyrite.montecarlo.runner.scheduling import run_cases

    if CATALOG.crystal(CATALOG.material(material).crystal_key).hkl_families:
        raise SystemExit(f"{material} is still pinned; this probe needs automatic selection")
    hkl = [int(i) for i in family.split(",")]
    PROBE["precision_override"](int(histories))
    case, _changes, n = PROBE["fixed_case"](material, float(e0_kev), float(tilt_deg), histories, 0)
    selections: list[dict] = []
    timings: list = []
    record = dict(
        issue=1302,
        code_sha=os.environ.get("PROBE_SHA"),
        profile=os.environ["PROFILE"],
        material=material,
        E0_keV=float(e0_kev),
        tilt_deg=float(tilt_deg),
        histories=n,
        family=hkl,
        production=PRODUCTION,
        error=None,
        timings=timings,
        selections=selections,
    )
    instrument(hkl, selections)
    t0 = time.perf_counter()
    try:
        run_cases([case], max_workers=0, progress=False, on_timing=timings.append)
    except Exception as exc:  # record failure as evidence
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_s"] = time.perf_counter() - t0
    Path(out_json).write_text(json.dumps(record, indent=1, default=PROBE["_json_default"]))
    print(json.dumps(summary(record), default=PROBE["_json_default"]))


def summary(record):
    keys = (
        "cutoff_invang",
        "reflections",
        "tail_fraction",
        "converged",
        "family_in_selected",
        "family_line_fraction",
        "family_net_fraction",
    )
    return dict(
        case=f"{record['material']} {record['E0_keV']:g} keV {record['tilt_deg']:g} deg "
        f"N={record['histories']} family={record['family']}",
        error=record["error"],
        selections=[{k: s.get(k) for k in keys} for s in record["selections"]],
    )


def table(out_dir):
    for path in sorted(Path(out_dir).glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("issue") == 1302:
            print(json.dumps(summary(record)))


if __name__ == "__main__":
    command, *rest = sys.argv[1:]
    if command == "run":
        run(*rest)
    elif command == "table":
        table(*rest)
    else:
        raise SystemExit(__doc__)
