"""Actual running server callback/picking smoke; intentionally no browser client.

uv run --extra trajectory-viewer python checks/saved_trajectory_server_probe.py \
  capture.h5 /tmp/server.png /tmp/server.json
"""

import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

from pyrite.trajectory_viewer.render import CaptureViewer
from pyrite.trajectory_viewer.selection import load_selection, plan_selection
from pyrite.trajectory_viewer.server import build_server


def main():
    plan = plan_selection(sys.argv[1])
    viewer = CaptureViewer(plan, load_selection(plan), off_screen=True)
    server = build_server(viewer, output_dir=Path(sys.argv[2]).parent)
    state = server.state
    failures = []

    async def smoke(**_):
        try:
            counts = []
            transitions = []
            for energy, generation, time in (
                (0.0, 99, 1e9),
                (10.0, 99, 1e9),
                (10.0, 0, 1e9),
                (0.0, 99, 0.0),
                (0.0, 99, 1e9),
            ):
                tick = perf_counter()
                with state:
                    state.energy_min = energy
                    state.generation_max = generation
                    state.time_max = time
                await asyncio.sleep(0.02)
                counts.append(viewer.visible.n_cells)
                transitions.append(perf_counter() - tick)
            assert counts[0] == counts[-1] == plan.segments
            assert 0 < counts[2] <= counts[1] < counts[0]
            assert counts[3] < counts[0]
            # Project one captured midpoint through the current render camera,
            # then exercise the same server-side cell picker used by the UI.
            renderer = viewer.plotter.renderer
            point = viewer.visible.get_cell(0).center
            renderer.SetWorldPoint(*point, 1.0)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            server.controller.pick_track({"position": {"x": x, "y": y}})
            picked = json.loads(state.picked)
            assert picked["track_id"] >= 0 and picked["full_segments"] > 0
            with state:
                state.clipping = True
                state.clip_x = float(point[0])
            await asyncio.sleep(0.02)
            clip_count = viewer.visible.n_cells
            assert 0 < clip_count <= plan.segments
            viewer.screenshot(sys.argv[2], overwrite=True)
            Path(sys.argv[3]).write_text(
                json.dumps(
                    dict(
                        client_connected=False,
                        mode="VtkRemoteView",
                        browser_delivery_verified=False,
                        counts=counts,
                        state_callback_yield_s=transitions,
                        picked=picked,
                        clipping_cells=clip_count,
                    ),
                    indent=2,
                )
                + "\n"
            )
        except Exception as error:
            failures.append(error)
        finally:
            await server.stop()
            viewer.close()

    server.controller.on_server_ready.add(lambda **kw: asyncio.create_task(smoke(**kw)))
    server.start(host="127.0.0.1", port=0, open_browser=False, show_connection_info=False)
    if failures:
        raise RuntimeError("server callback smoke failed") from failures[0]


if __name__ == "__main__":
    main()
