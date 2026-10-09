"""Connected-browser regression and timing probe for the saved trajectory viewer.

Launches ``pyrite app trajectories launch`` and drives it from a real headless
Chrome through Playwright: every control is operated by mouse input, so
widget bindings, browser event serialization, image delivery and downloads
are exercised end to end (the server-only probe cannot see those)::

    LIBGL_ALWAYS_SOFTWARE=1 uv run --extra trajectory-viewer --with playwright \\
        python checks/saved_trajectory_browser_probe.py capture.h5 --output trial.json

Uses the installed Google Chrome (``--channel chrome``); ``--channel chromium``
needs ``playwright install chromium``. The capture must contain secondaries.
By default server and client are co-located over loopback. ``--server-url``
attaches through an existing SSH tunnel; a separate remote supervisor samples
that server. Both local server and Chrome process trees are sampled. Latency
ends at observation of a new decoded image, not compositor paint or FPS.
Not a physics check; no ledger records.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image


def _secondary_fixture(work):
    """Small deterministic inspection fixture; no transport is simulated."""
    from pyrite.montecarlo.trajectories import write_trajectory_artifact

    midpoints, directions = [], []
    for z in (0.0, 4.0):
        midpoints.extend([[i + 0.5, 0, z] for i in range(3)])
        directions.extend([[1, 0, 0]] * 3)
        midpoints.extend([[3, i + 0.5, z] for i in range(3)])
        directions.extend([[0, 1, 0]] * 3)
    return write_trajectory_artifact(
        work / "secondary-fixture.h5",
        dict(
            r_mid=np.array(midpoints, dtype=float),
            v_hat=np.array(directions, dtype=float),
            L_ang=np.ones(12),
            electron_id=np.repeat([0, 1], 6),
            track_id=np.repeat(np.arange(4), 3),
            parent_id=np.repeat([-1, 0, -1, 2], 3),
            generation=np.tile(np.repeat([0, 1], 3), 2),
            E_start_keV=np.tile([20.0, 15.0, 10.0, 12.0, 8.0, 2.0], 2),
            t_start_ang=np.tile(np.arange(6.0) * 2997.924580, 2),
            t0_ang=np.zeros(12),
        ),
        case=dict(
            name="browser-fixture",
            crystal="Si",
            E0_keV=20.0,
            thickness_ang=5.0,
            tilt_deg=0.0,
            tilt_azim_deg=0.0,
        ),
    )


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _tree_rss_mib(pid):
    """Resident memory of a process and its descendants (Linux /proc)."""
    total, pending = 0, [pid]
    while pending:
        current = pending.pop()
        try:
            status = Path(f"/proc/{current}/status").read_text()
            children = Path(f"/proc/{current}/task/{current}/children").read_text().split()
        except OSError:
            continue
        total += next(
            (int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS")),
            0,  # exited/zombie child: Linux status has no resident pages
        )
        pending.extend(int(child) for child in children)
    return total / 1024


class _Resources:
    """Sample isolated server/Chrome process trees; summed RSS counts shared pages."""

    def __init__(self, server_pid):
        self.roots = {"server": server_pid} if server_pid is not None else {}
        self.peaks = {"browser": 0.0, **({"server": 0.0} if server_pid is not None else {})}
        self.samples = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.wait(0.05):
            for name, pid in list(self.roots.items()):
                self.peaks[name] = max(self.peaks[name], _tree_rss_mib(pid))
            self.samples += 1

    def snapshot(self):
        return {f"{name}_tree_rss_mib": _tree_rss_mib(pid) for name, pid in self.roots.items()}

    def close(self):
        self.stop.set()
        self.thread.join()
        return dict(
            peak_tree_rss_mib=self.peaks,
            interval_s=0.05,
            samples=self.samples,
            method="Linux /proc VmRSS sum over each root and all descendants; shared pages counted repeatedly",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("capture", type=Path, nargs="?")
    parser.add_argument(
        "--fixture-secondary",
        action="store_true",
        help="Generate a deterministic fixture and require secondary-parent picking.",
    )
    parser.add_argument(
        "--require-secondary",
        action="store_true",
        help="Continue picking until a secondary with a loaded parent is found.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", default="chrome")
    parser.add_argument("--screenshots", type=Path, help="Directory for page screenshots.")
    parser.add_argument(
        "--server-url",
        help="Attach through an existing SSH tunnel; server ownership and resources stay external.",
    )
    parser.add_argument(
        "--capture-sha256",
        help="Verified remote input hash for attach mode when the capture is not local.",
    )
    # Unrecognized options pass to `launch`, e.g. --max-segments 2000000.
    args, launch_args = parser.parse_known_args()
    if (args.capture is None) != args.fixture_secondary:
        parser.error("provide a capture or --fixture-secondary")

    from playwright.sync_api import sync_playwright

    port = _free_port()
    # Keep all worker reports and exports alongside the raw result, rather
    # than relying on an independently cleaned system temporary directory.
    work = Path(tempfile.mkdtemp(prefix=".saved-viewer-probe-", dir=args.output.parent.resolve()))
    if args.fixture_secondary:
        args.capture = _secondary_fixture(work)
        args.require_secondary = True
    perf = time.perf_counter
    tick = perf()
    worker_report = work / "server-phases.json"
    log = (work / "server.log").open("w+")
    server = None if args.server_url else subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("saved_trajectory_browser_worker.py")),
         str(worker_report), str(args.capture),
         "--port", str(port), "--output-dir", str(work / "exports"), *launch_args],
        stdout=log, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    resources = _Resources(server.pid if server is not None else None)
    result = dict(
        capture=str(args.capture),
        host=dict(
            platform=platform.platform(),
            python=platform.python_version(),
            cpu_count=os.cpu_count(),
            available_memory_bytes=psutil.virtual_memory().available,
            versions={
                name: importlib.metadata.version(name)
                for name in ("pyvista", "vtk", "trame", "trame-vtk", "playwright", "psutil")
            },
        ),
        client="headless Chrome via Playwright; attached server"
        if args.server_url
        else "headless Chrome via Playwright on the same host (loopback)",
        server_managed=server is not None,
        steps={},
        failures=[],
        fixture="deterministic secondary inspection" if args.fixture_secondary else None,
        cold_process=True,
        probe_tick_monotonic=tick,
        cache_state="OS/font/driver caches retained; server and browser processes start fresh",
        viewport=[1600, 1000],
    )
    if args.capture_sha256:
        if not args.server_url or len(args.capture_sha256) != 64:
            parser.error("--capture-sha256 requires attach mode and a full SHA-256")
        int(args.capture_sha256, 16)
        result["capture_sha256"] = args.capture_sha256
        result["capture_hash_source"] = "verified remotely by the trial driver"
    else:
        with args.capture.open("rb") as stream:
            result["capture_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    frames = []  # (time, bytes, binary) of server->client websocket frames
    sent = []

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=args.channel, headless=True)
            result["browser_version"] = browser.version
            browser_cdp = browser.new_browser_cdp_session()
            processes = browser_cdp.send("SystemInfo.getProcessInfo")["processInfo"]
            resources.roots["browser"] = next(p["id"] for p in processes if p["type"] == "browser")
            page = browser.new_page(viewport={"width": 1600, "height": 1000}, accept_downloads=True)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "console",
                lambda message: (
                    errors.append(message.text) if "Unrecognized" in message.text else None
                ),
            )

            def on_socket(socket_):
                socket_.on(
                    "framereceived",
                    lambda payload: frames.append(
                        (perf(), len(payload) if payload else 0, isinstance(payload, bytes))
                    ),
                )
                socket_.on(
                    "framesent",
                    lambda payload: sent.append((perf(), len(payload) if payload else 0)),
                )

            page.on("websocket", on_socket)
            url = args.server_url or f"http://127.0.0.1:{port}"
            for _ in range(600):
                if server is not None and server.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(f"server exited: {log.read()[-2000:]}")
                try:
                    page.goto(url)
                    break
                except Exception:  # noqa: BLE001 -- server still loading the capture
                    time.sleep(0.2)
            page_open = perf()
            page.wait_for_selector("#selection-count", timeout=60000)
            # First delivered image: the remote view shows frames as a blob <img>.
            page.wait_for_function(
                "() => { const i = document.querySelector('.v-col-9 img');"
                " return i && i.src.startsWith('blob:') && i.complete && i.naturalWidth > 0; }",
                timeout=60000,
            )
            first_frame = perf()
            result["startup"] = dict(
                launch_to_page_s=page_open - tick,
                page_to_first_image_s=first_frame - page_open,
                launch_to_first_image_s=first_frame - tick,
                **resources.snapshot(),
            )
            if args.server_url:
                result["startup"]["probe_start_to_first_image_s"] = result["startup"].pop(
                    "launch_to_first_image_s"
                )
                result["startup"]["probe_start_to_page_s"] = result["startup"].pop(
                    "launch_to_page_s"
                )

            def text(selector):
                return page.inner_text(selector).replace("\n", " ")

            def shot(name):
                if args.screenshots:
                    args.screenshots.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(args.screenshots / f"{name}.png"))

            def cells():
                return int(text("#selection-count").split(";")[1].split()[0])

            def image_src():
                return page.eval_on_selector(
                    ".v-col-9 img", "i => i.complete && i.naturalWidth > 0 ? i.src : null"
                )

            def timed(name, action, settled, timeout=30.0):
                """Act; time the UI state and the next client-visible image."""
                before, old_image = len(frames), image_src()
                start = perf()
                action()
                action_s = perf() - start
                deadline = start + timeout
                while not settled():
                    if perf() > deadline:
                        result["failures"].append(f"{name}: UI did not settle")
                        return
                    page.wait_for_timeout(10)
                state_s = perf() - start
                while not (image_src() and image_src() != old_image):
                    if perf() > deadline:
                        result["failures"].append(f"{name}: no new image delivered")
                        return
                    page.wait_for_timeout(10)
                image_s = perf() - start
                page.wait_for_timeout(500)  # let trailing interactive frames finish
                received = [(size, binary) for at, size, binary in frames[before:] if at >= start]
                result["steps"][name] = dict(
                    input_action_s=action_s,
                    ui_settled_s=state_s,
                    first_new_image_s=image_s,
                    bytes_received=sum(size for size, _ in received),
                    binary_bytes_received=sum(size for size, binary in received if binary),
                    text_bytes_received=sum(size for size, binary in received if not binary),
                    bytes_sent=sum(size for at, size in sent if start <= at <= perf()),
                    frames_received=len(received),
                    displayed_cells=cells(),
                    **resources.snapshot(),
                )
                shot(name)

            def check(name, condition):
                if not condition:
                    result["failures"].append(name)

            full = cells()
            sliders = page.locator(".v-slider")

            canvas = page.locator(".v-col-9").bounding_box()

            def orbit():
                page.mouse.move(canvas["x"] + 500, canvas["y"] + 300)
                page.mouse.down()
                page.mouse.move(canvas["x"] + 620, canvas["y"] + 360, steps=6)
                page.mouse.up()

            for i in range(3):
                timed(f"camera_orbit_{i}", orbit, lambda: True)
            page.mouse.move(canvas["x"] + 600, canvas["y"] + 400)
            for i in range(3):
                timed(f"camera_zoom_{i}", lambda: page.mouse.wheel(0, -120), lambda: True)
            timed(
                "camera_initial_reset",
                lambda: page.get_by_text("Reset camera").click(),
                lambda: True,
            )

            def drag(index, fraction):
                thumb = sliders.nth(index).locator(".v-slider-thumb").bounding_box()
                track = sliders.nth(index).locator(".v-slider-track").bounding_box()
                page.mouse.move(thumb["x"] + thumb["width"] / 2, thumb["y"] + thumb["height"] / 2)
                page.mouse.down()
                page.mouse.move(
                    track["x"] + track["width"] * fraction,
                    track["y"] + track["height"] / 2,
                    steps=8,
                )
                page.mouse.up()

            timed("energy_half", lambda: drag(0, 0.5), lambda: cells() < full)
            timed("energy_zero", lambda: drag(0, -0.1), lambda: cells() == full)

            def keys(index, key, count):
                sliders.nth(index).locator(".v-slider-thumb").focus()
                for _ in range(count):
                    page.keyboard.press(key)

            timed("energy_keyboard", lambda: keys(0, "ArrowRight", 30), lambda: cells() < full)
            timed("energy_keyboard_home", lambda: keys(0, "Home", 1), lambda: cells() == full)
            timed("generation_zero", lambda: drag(1, -0.1), lambda: cells() < full)
            primaries = cells()
            check("generation 0 shows its value", text("#generation-value").endswith(": 0"))
            timed("generation_max", lambda: drag(1, 1.1), lambda: cells() == full)
            timed("time_small", lambda: drag(2, 0.01), lambda: cells() < primaries)
            timed("time_max", lambda: drag(2, 1.1), lambda: cells() == full)
            switch = page.locator(".v-switch input")
            timed("clip_on", switch.click, lambda: cells() < full)
            clipped = cells()
            timed("clip_move", lambda: drag(3, 0.8), lambda: cells() != clipped)
            timed("clip_off", switch.click, lambda: cells() == full)

            canvas = page.locator(".v-col-9").bounding_box()
            picked = None
            pick_start = perf()
            # Scan outward from the centre. Each click is also a remote camera
            # interaction that the server renders, so wait for its image before
            # the next click; rapid clicking would queue renders on large scenes.
            # By default the first track suffices; the deterministic fixture
            # requires a secondary so browser parent navigation is exercised.
            grid = sorted(
                ((i / 20, j / 20) for i in range(4, 17) for j in range(3, 17)),
                key=lambda point: (point[0] - 0.5) ** 2 + (point[1] - 0.45) ** 2,
            )
            for fx, fy in grid:
                old_image = image_src()
                page.mouse.click(
                    canvas["x"] + canvas["width"] * fx, canvas["y"] + canvas["height"] * fy
                )
                waited = perf()
                while image_src() == old_image and perf() - waited < 5:
                    page.wait_for_timeout(20)
                page.wait_for_timeout(200)
                if text("#pick-status").startswith("Picked"):
                    candidate = json.loads(page.inner_text("#picked-info"))
                    if not args.require_secondary or (
                        candidate["parent_id"] >= 0 and candidate["parent_available"]
                    ):
                        picked = candidate
                        break
            check("click picks a captured segment", picked is not None)
            result["pick"] = dict(
                search_s=perf() - pick_start,
                segment_id=picked and picked["segment_id"],
                track_id=picked and picked["track_id"],
                parent_id=picked and picked["parent_id"],
            )
            shot("pick")
            if picked:
                page.get_by_text("Inspect parent").click()
                page.wait_for_timeout(1000)
                status = text("#pick-status")
                result["pick"]["parent_status"] = status
                if picked["parent_id"] >= 0 and picked["parent_available"]:
                    parent = json.loads(page.inner_text("#picked-info"))
                    check("parent shows parent track", parent["track_id"] == picked["parent_id"])
                    # Hide the parent through the real slider, then restore:
                    # inspector identity must survive both updates.
                    parent_segment = parent["segment_id"]
                    timed("picked_filter", lambda: drag(0, 1.1), lambda: cells() < full)
                    hidden = json.loads(page.inner_text("#picked-info"))
                    check("filter retains parent segment", hidden["segment_id"] == parent_segment)
                    check("filter reports partial track", hidden["clipped"])
                    timed("picked_restore", lambda: drag(0, -0.1), lambda: cells() == full)
                    picked = json.loads(page.inner_text("#picked-info"))
                    check("restored parent is complete", not picked["clipped"])
                else:
                    check("parent action reports why", status and "Picked" not in status)
                shot("parent")

            selection = page.inner_text("#picked-info")
            before_camera = len(frames)
            page.mouse.move(canvas["x"] + 500, canvas["y"] + 300)
            page.mouse.down()
            page.mouse.move(canvas["x"] + 700, canvas["y"] + 380, steps=10)
            page.mouse.up()
            page.wait_for_timeout(1500)
            check("orbit delivers images", len(frames) > before_camera)
            check("orbit keeps selection", page.inner_text("#picked-info") == selection)
            result["orbit"] = dict(
                frames_received=len(frames) - before_camera,
                bytes_received=sum(size for _, size, _ in frames[before_camera:]),
            )
            shot("orbit")
            timed("reset_camera", lambda: page.get_by_text("Reset camera").click(), lambda: True)

            def choose(option):
                page.locator(".v-select .v-field").click()
                page.locator(".v-overlay--active .v-list-item", has_text=option).click()

            timed(
                "scale_instrument",
                lambda: choose("instrument"),
                lambda: "mm" in text("#clip-value"),
            )
            check(
                "instrument keeps selection",
                not picked
                or json.loads(page.inner_text("#picked-info"))["segment_id"]
                == picked["segment_id"],
            )
            timed(
                "scale_closeup",
                lambda: choose("closeup"),
                lambda: "angstrom" in text("#clip-value"),
            )

            downloads = {}
            for label, name, frames_expected in (
                ("Screenshot", "screenshot", 1),
                ("Orbit GIF", "orbit_gif", 30),
            ):
                start = perf()
                with page.expect_download(timeout=300000) as info:
                    page.get_by_text(label).click()
                target = work / info.value.suggested_filename
                info.value.save_as(target)
                with Image.open(target) as image:
                    frame_hashes = []
                    for index in range(getattr(image, "n_frames", 1)):
                        image.seek(index)
                        rgb = np.asarray(image.convert("RGB"))
                        check(f"{name} frame {index} contains rendered content", rgb.std() > 5)
                        frame_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
                    downloads[name] = dict(
                        seconds=perf() - start,
                        bytes=target.stat().st_size,
                        size=list(image.size),
                        frames=getattr(image, "n_frames", 1),
                        distinct_frames=len(set(frame_hashes)),
                    )
                check(f"{name} frames", downloads[name]["frames"] == frames_expected)
                server_outputs = (
                    list((work / "exports").glob(f"*{target.suffix}")) if server is not None else []
                )
                if server is not None:
                    check(f"{name} server copy exists", len(server_outputs) == 1)
                if len(server_outputs) == 1:
                    saved = server_outputs[0]
                    check(
                        f"{name} downloaded bytes match server copy",
                        saved.read_bytes() == target.read_bytes(),
                    )
                    meta = json.loads(saved.with_suffix(saved.suffix + ".json").read_text())
                    check(
                        f"{name} provenance image size",
                        meta["image_size"] == downloads[name]["size"],
                    )
                    check(f"{name} provenance scale", meta["scale"] == "closeup")
                    check(f"{name} provenance selection", meta["full_segments"] == full)
                page.wait_for_timeout(500)
                check(f"{name} status", text("#export-status").startswith("Export ready:"))
                downloads[name]["server_filename"] = text("#export-status").split()[2]
                with target.open("rb") as stream:
                    downloads[name]["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
                if frames_expected > 1:
                    check(
                        "GIF orbit changes rendered content", downloads[name]["distinct_frames"] > 1
                    )
            sidecars = sorted((work / "exports").glob("*.json"))
            if server is not None:
                check("server copies have provenance sidecars", len(sidecars) == 2)
            result["downloads"] = downloads
            page.get_by_text("Clear", exact=True).click()
            page.wait_for_function(
                "() => document.querySelector('#picked-info').textContent.trim() === ''"
            )
            check("clear removes inspector selection", not page.inner_text("#picked-info").strip())
            # A non-directory export path forces an actual server filesystem
            # failure; it must produce feedback without a bogus download.
            if server is not None:
                exports = work / "exports"
                exports.rename(work / "successful-exports")
                exports.write_text("blocked export directory")
                failed_downloads = []
                page.on("download", lambda download: failed_downloads.append(download))
                page.get_by_text("Screenshot", exact=True).click()
                page.wait_for_function(
                    "() => document.querySelector('#export-status').textContent.startsWith('Export failed:')"
                )
                page.wait_for_timeout(500)
                check("failed export does not download", not failed_downloads)
                result["export_failure_status"] = text("#export-status")
            else:
                result["external_verification_required"] = (
                    "Retrieve remote server copies/sidecars to verify download hashes and provenance; failure injection skipped on externally owned server."
                )
            result["browser_errors"] = errors
            check("no browser serialization errors", not errors)
            browser.close()
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        result["resources"] = resources.close()
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        if worker_report.exists():
            result["server_preparation"] = json.loads(worker_report.read_text())
        log.close()
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["failures"]:
        raise SystemExit(f"browser probe failures: {result['failures']}")
    print(f"browser probe passed; results in {args.output}")


if __name__ == "__main__":
    main()
