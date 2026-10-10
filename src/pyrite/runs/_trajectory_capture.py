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

    if os.environ.get("PYRITE_TRAJECTORY_SIZE_REPORT") == "1":
        from pathlib import Path

        base = Path(os.environ["PYRITE_HOME"]).resolve()
        if not Path(args.trajectories).resolve().is_relative_to(base):
            raise ValueError("trajectory root escapes the remote checkout through a symlink")
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


class CaptureSizeReport:
    """Warn once using the first completed capture; no segment arrays are read."""

    def __init__(self, capture, cases, *, threshold_bytes=10 * 1024**3):
        self.capture = capture
        self.histories = sum(case["Ne"] for case in cases)
        self.threshold_bytes = threshold_bytes
        self.reported = False

    def completed(self, case):
        if self.reported:
            return
        from ..console.output import emit_diagnostic
        from ..montecarlo.trajectories import read_trajectory_header

        path = self.capture.path_for(case)
        if not path.exists():
            return
        header = read_trajectory_header(path)
        segments = header["segment_count"]
        if not segments:
            return
        size = path.stat().st_size
        projected_segments = segments * self.histories / case["Ne"]
        projected = int(projected_segments * size / segments)
        self.reported = True
        emit_diagnostic(
            f"trajectories: first case {segments} segments, {size / segments:.1f} B/segment; "
            f"projected capture {projected} bytes ({projected_segments:.0f} segments). "
            "Estimate scales the first case by requested histories; geometry, energy, and "
            "adaptive stopping can change actual size."
        )
        if projected > self.threshold_bytes:
            emit_diagnostic(
                f"warning: projected trajectory capture exceeds {self.threshold_bytes} bytes; "
                "leave captures remote and pull selected cases or export selected histories"
            )
