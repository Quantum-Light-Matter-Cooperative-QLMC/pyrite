"""Hashes of analog output for one transport core, to compare across commits (#879)."""

import argparse
import hashlib

import numpy as np

import pyrite
from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.montecarlo import runner

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--core", required=True)
args = parser.parse_args()
print("# package", pyrite.__file__)
sweep = Sweep(
    material="hopg",
    thickness_ang=1e4,
    beam=BeamSpec(energy_keV=30.0),
    tilt_deg=30.0,
    n_electrons=64,
    n_electrons_brem=64,
)
policy = dict(build_cases(sweep)[0])
fixed = {key: value for key, value in policy.items() if key != "line_grid_policy"}
for label, case in (("policy", policy), ("fixed", fixed)):
    out = runner.run_case(case, transport_core=args.core)
    for key in ("E_grid", "spec", "spec_characteristic", "brem_wide"):
        digest = hashlib.sha256(np.asarray(out[key]).tobytes()).hexdigest()[:20]
        print(args.core, label, key, digest)
