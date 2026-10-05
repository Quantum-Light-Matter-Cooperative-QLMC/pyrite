"""Build the opt-in trajectory capture for one scanned material (issue #159)."""

from __future__ import annotations

import os


def scan_trajectory_capture(args, identity, stem, material, fidelity):
    """``TrajectoryCapture`` under ``args.trajectories/<stem>`` with scene and provenance.

    The scene snapshots the profile's physical filters/detector (#205),
    independently of counting policy.
    """
    from ..campaign.observation import profile_trajectory_scene
    from ..montecarlo.trajectories import TrajectoryCapture

    profile = identity.get("catalog_profile", "standard")
    return TrajectoryCapture(
        root=os.path.join(os.fspath(args.trajectories), stem),
        overwrite=bool(getattr(args, "overwrite_trajectories", False)),
        scene=profile_trajectory_scene(str(profile), getattr(args, "detector_id", None)),
        provenance={
            "material": material,
            "checkpoint_stem": stem,
            "catalog_profile": profile,
            "fidelity": fidelity,
            "parameter_sha256": identity["parameter_sha256"],
            "dataset_identity": identity,
        },
    )
