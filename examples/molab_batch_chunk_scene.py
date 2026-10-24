"""MoLab example for joint transport-batch / CXR-chunk tuning.

Keep this factory and its physical/numerical setup FIXED throughout a study.
Electron and spectrum batch sizes are configured separately via environment.
"""

import numpy as np

from pyrite import (
    Beam, CxrReflections, Detector, EnergyBins, Footprint, Numerics,
    Physics, PointBunch, PointSpot, Scene, SensorAcceptance, Slab,
)


def make_scene() -> Scene:
    return Scene(
        beam=Beam(
            energy_keV=50.0,
            transverse=PointSpot(),
            longitudinal=PointBunch(),
        ),
        target=Slab(
            "hopg", thickness_ang=2000.0, tilt_deg=30.0,
            footprint=Footprint(5.0, 5.0),
        ),
        detector=Detector(
            observation_angle_deg=90.0,
            acceptance=SensorAcceptance(1000.0, 35.0, 35.0),
        ),
        physics=Physics(emission="incoherent", brem_source="mc"),
        numerics=Numerics(
            cxr_trials=8192,
            brem_trials=2048,
            cxr_reflections=CxrReflections(n_families=2),
            energy_bins=EnergyBins(cxr=np.linspace(250.0, 1250.0, 1001)),
            # Do not specify cxr_chunk: the tuner controls it with an env var.
        ),
    )
