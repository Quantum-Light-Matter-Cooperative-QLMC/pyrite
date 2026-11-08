"""Bit stability of device line masses and spectra on identical input (#879); remote GPU only."""

import numpy as np

from pyrite._backend import _to_cpu, xp
from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.montecarlo import runner

sweep = Sweep(
    material="hopg",
    thickness_ang=1e4,
    beam=BeamSpec(energy_keV=30.0),
    tilt_deg=30.0,
    n_electrons=8,
    n_electrons_brem=8,
)
base = dict(build_cases(sweep)[0])
base.pop("line_grid_policy")
case = {**base, "Ne": 40, "Ne_brem": 52}
tp = runner._transport_case(case, transport_core="per-electron")
grid = tp["E_grid"]
axis = np.array([grid[0], grid[-1]])
segs = runner._segments_on_device(tp["segs"])
masses, specs = [], []
for _ in range(30):
    a = {
        "start_eV": float(grid[0]),
        "stop_eV": float(grid[-1]),
        "n_electrons": 40,
        "mass_only": True,
    }
    runner._cxr_for_segments(
        segs, axis, case, tp["n_hat"], None, None, coherent=False, Ne=40, truncation_audit=a
    )
    masses.append(np.asarray(_to_cpu(a["electron_mass"]), dtype=float))
for _ in range(10):
    specs.append(
        np.asarray(
            runner._cxr_for_segments(
                segs, grid, case, tp["n_hat"], None, None, coherent=False, Ne=40
            )
        )
    )
m = np.array(masses)
print("backend", getattr(xp, "__name__", "?"))
print("distinct mass vectors", len({r.tobytes() for r in m}), "of", len(m))
print(
    "primaries varying",
    int((m.max(0) != m.min(0)).sum()),
    "max rel spread",
    float(((m.max(0) - m.min(0)) / np.maximum(m.mean(0), 1e-300)).max()),
)
print("distinct spectra", len({s.tobytes() for s in specs}), "of", len(specs))
w = xp.asarray(np.random.default_rng(0).random(2_000_000))
i = xp.asarray(np.random.default_rng(1).integers(0, 40, 2_000_000))
print(
    "distinct raw bincount",
    len({_to_cpu(xp.bincount(i, weights=w, minlength=40)).tobytes() for _ in range(20)}),
    "of 20",
)
