"""Time pieces of one _main_line_keV workload (issue #646 audit)."""

import json
import resource
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info

from pyrite.materials.channeling import longitudinal_kinematics
from pyrite.materials.channeling.axial import axial_profile, axial_states
from pyrite.montecarlo.channeling.axial_emission import axial_line_catalog

uvw = tuple(int(c) for c in sys.argv[1])
energy_MeV = float(sys.argv[2])
order = int(sys.argv[3]) if len(sys.argv) > 3 else 16
info = threadpool_info()[0]["num_threads"]
t = {}
t0 = time.perf_counter()
prof = axial_profile("silicon", uvw)
t["profile"] = time.perf_counter() - t0
gamma, _ = longitudinal_kinematics(energy_MeV * 1e6)
t0 = time.perf_counter()
prof.extrema()
t["extrema"] = time.perf_counter() - t0
t0 = time.perf_counter()
st = axial_states(prof, gamma, max_order=order)
t["one_axial_states"] = time.perf_counter() - t0
n = len(st.energies_eV)
h = np.random.default_rng(0).standard_normal((n, n))
h = h + h.T
t0 = time.perf_counter()
np.linalg.eigh(h)
t["bare_eigh"] = time.perf_counter() - t0
t0 = time.perf_counter()
cat = axial_line_catalog(prof, energy_MeV * 1e6, max_order=order, max_bandwidth_eV=0.5)
t["catalog"] = time.perf_counter() - t0
t0 = time.perf_counter()
for ln in cat.lines:
    cat.yield_per_ang(ln, np.eye(len(cat.levels[ln.initial].states)))
t["yields"] = time.perf_counter() - t0
print(
    json.dumps(
        {
            "uvw": sys.argv[1],
            "E": energy_MeV,
            "order": order,
            "basis": n,
            "blas_threads": info,
            "profile_max_order": prof.max_order,
            "levels": len(cat.levels),
            "lines": len(cat.lines),
            "times_s": {k: round(v, 3) for k, v in t.items()},
            "maxrss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        }
    )
)
