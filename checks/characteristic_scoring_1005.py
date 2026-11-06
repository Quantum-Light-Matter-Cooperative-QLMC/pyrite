"""Characteristic-scoring split for #1005 on the #872 streamed harness.

Usage:
  characteristic_scoring_1005.py run MODE NE BLOCK OUT_JSON MATERIAL [E0_KEV [CORE [THICKNESS_ANG [SEED]]]]
  characteristic_scoring_1005.py compare REF_JSON OTHER_JSON... [--rtol R]

``run`` is ``checks/streamed_scoring_memory.py run`` (same case, same JSON)
plus synchronized timers inside ``mc_characteristic_spectrum``: ``total``
(every call), ``clip`` (``_clip_segments_to_cutoff``), ``escape``
(``segment_escape_paths``), ``accumulate`` (``_accumulate_line_yields``, the
per-segment scoring) and its ``interp`` share
(``_interpolate_shell_cross_sections``). ``setup`` is ``total`` less ``clip``,
``escape`` and ``accumulate``: the trajectory-independent table, profile,
response and attenuation preparation. Each timed callable also opens an NVTX
range (``char.<name>``) so an Nsight trace attributes kernel launches and
copies to it. ``MATERIAL`` ``A/B`` builds a layered target: a 1 um film of A
on a THICKNESS_ANG substrate of B.

``compare`` checks the three spectra against a reference run at ``--rtol``
(default 0: bitwise), relative to each spectrum's peak.
"""

import json
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamed_scoring_memory as harness  # noqa: E402

CHAR_TIMERS = ("total", "clip", "escape", "accumulate", "interp")


def _layered_case(material, ne, e0_kev, thickness_ang=1.0e4, seed=None):
    import pyrite as pr
    from pyrite import api
    from pyrite.detectors import EnergyBins

    film, substrate = material.split("/")
    numerics = pr.Numerics(
        cxr_trials=ne,
        brem_trials=ne,
        energy_model="midpoint",
        energy_bins=EnergyBins(
            cxr=np.linspace(1000.0, 20000.0, 400), brem=np.arange(1000.0, e0_kev * 1e3, 1000.0)
        ),
    )
    physics = pr.Physics(
        bremsstrahlung_model="bremslib",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
    )
    target = pr.geometry.Stack.on_substrate(
        film,
        1.0e4,
        substrate,
        thickness_ang,
        tilt_deg=30.0,
        footprint=pr.geometry.Footprint(5.0, 5.0),
    )
    scene = pr.Scene(
        pr.Beam(energy_keV=e0_kev, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        target,
        pr.Detector(),
        physics=physics,
    )
    case = api.build_case(scene, numerics)
    case = replace(case, E_cut_lines_keV=5.0, E_cut_brem_keV=5.0)
    return case if seed is None else replace(case, seed=int(seed))


def _nvtx(name):
    try:
        from cupyx.profiler import time_range

        return time_range(f"char.{name}")
    except Exception:
        return nullcontext()


class _CharTimers:
    def __init__(self):
        self.seconds = dict.fromkeys(CHAR_TIMERS, 0.0)
        self.calls = dict.fromkeys(CHAR_TIMERS, 0)

    def wrap(self, name, fn):
        def timed(*args, **kwargs):
            harness._sync()
            t0 = time.perf_counter()
            try:
                with _nvtx(name):
                    out = fn(*args, **kwargs)
                    harness._sync()
                return out
            finally:
                self.seconds[name] += time.perf_counter() - t0
                self.calls[name] += 1

        return timed


def _instrument(timers):
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.spectrum import characteristic as char

    char._clip_segments_to_cutoff = timers.wrap("clip", char._clip_segments_to_cutoff)
    char.segment_escape_paths = timers.wrap("escape", char.segment_escape_paths)
    char._accumulate_line_yields = timers.wrap("accumulate", char._accumulate_line_yields)
    char._interpolate_shell_cross_sections = timers.wrap(
        "interp", char._interpolate_shell_cross_sections
    )
    runner.mc_characteristic_spectrum = timers.wrap("total", runner.mc_characteristic_spectrum)


def run(mode, ne, block, out_path, material, *rest):
    timers = _CharTimers()
    _instrument(timers)
    if "/" in material:
        harness._case = _layered_case
    harness.run(mode, ne, block, out_path, material, *rest)
    s = timers.seconds
    split = {
        **{f"char.{k}": v for k, v in s.items()},
        "char.setup": s["total"] - s["clip"] - s["escape"] - s["accumulate"],
    }
    record = json.load(open(out_path))
    record["char_s"] = split
    record["char_calls"] = timers.calls
    with open(out_path, "w") as handle:
        json.dump(record, handle)
    print(json.dumps({"char_s": split, "char_calls": timers.calls}))


def compare(ref_path, *others, rtol=0.0):
    ref = json.load(open(ref_path))
    worst = 0.0
    for path in others:
        other = json.load(open(path))
        assert other["n_segments"] == ref["n_segments"], path
        for key in harness.SPECTRA:
            a, b = np.asarray(other[key]), np.asarray(ref[key])
            peak = max(float(np.max(np.abs(b), initial=0.0)), np.finfo(float).tiny)
            rel = float(np.max(np.abs(a - b), initial=0.0)) / peak
            worst = max(worst, rel)
            print(f"{path} {key}: max |diff|/peak {rel:.3e}")
    print(f"worst {worst:.3e} (rtol {rtol:g})")
    return 0 if worst <= rtol else 1


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:])
    elif sys.argv[1] == "compare":
        args = sys.argv[2:]
        rtol = 0.0
        if "--rtol" in args:
            i = args.index("--rtol")
            rtol = float(args[i + 1])
            del args[i : i + 2]
        sys.exit(compare(*args, rtol=rtol))
    else:
        raise SystemExit(__doc__)
