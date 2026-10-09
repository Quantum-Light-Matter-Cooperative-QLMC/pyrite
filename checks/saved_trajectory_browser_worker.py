"""Instrument the real launch command without adding application probe hooks."""

import json
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch


def main():
    output = Path(sys.argv[1])
    tick = perf_counter()
    report = {"phases_s": {}, "filter_calls": []}

    from pyrite.cli.commands import app_trajectories as command
    from pyrite.trajectory_viewer import render, server

    report["imports_s"] = perf_counter() - tick

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    def phase(name, function):
        def measured(*args, **kwargs):
            start = perf_counter()
            value = function(*args, **kwargs)
            report["phases_s"][name] = perf_counter() - start
            save()
            return value

        return measured

    original_viewer = render.CaptureViewer

    def viewer(*args, **kwargs):
        instance = phase("scene_build", original_viewer)(*args, **kwargs)
        report["renderer"] = instance.plotter.render_window.ReportCapabilities()
        report["segments"] = instance.plan.segments
        apply = instance.apply_filters

        def filters(**controls):
            start = perf_counter()
            count = apply(**controls)
            report["filter_calls"].append(
                dict(controls=controls, seconds=perf_counter() - start, cells=count)
            )
            if report["renderer"] == "Display ID not set":
                report["renderer"] = instance.plotter.render_window.ReportCapabilities()
            save()
            return count

        instance.apply_filters = filters
        save()
        return instance

    with (
        patch.object(command, "plan_selection", phase("selection_scan", command.plan_selection)),
        patch.object(command, "load_selection", phase("array_load", command.load_selection)),
        patch.object(render, "CaptureViewer", viewer),
        patch.object(server, "build_server", phase("server_ui_build", server.build_server)),
    ):
        command.command.main(args=["launch", *sys.argv[2:]], prog_name="pyrite app trajectories")


if __name__ == "__main__":
    main()
