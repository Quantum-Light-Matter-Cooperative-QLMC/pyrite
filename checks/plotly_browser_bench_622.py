"""Browser render benchmark for Plotly spectrum traces: SVG versus WebGL (#622).

Run:
    pixi run python checks/plotly_browser_bench_622.py \
        --out scratch/plotly_browser_bench_622.json

Builds spectrum-shaped figures through the production Plotly spectrum trace
builder (``pyrite.plots.plotly.spectra._trace_series``) on synthetic line +
continuum data, then re-types every trace as ``Scatter`` (SVG) or
``Scattergl`` (WebGL). Each workload page loads the installed plotly.js and
times, in the browser, ``Plotly.newPlot`` to two animation frames (initial
render) and a 10 % x-range ``Plotly.relayout`` zoom to two animation frames.
Pages are reused only within one workload; every repeat purges the plot.

Pages are served from a loopback HTTP server (snap-packaged browsers cannot
read the host ``/tmp``). Browsers are driven through Selenium: Firefox via ``geckodriver`` and Chrome
or Chromium via selenium-manager, with the GPU blocklist ignored so headless
Chrome does not fall back to SwiftShader. Headless by default; ``--headful``
opens windows on the current display. The unmasked WebGL renderer string is
recorded so a software rasterizer is visible in the record.

Engineering evidence for the provisional WebGL point threshold; no physics or
numerical output is involved, so no validation-ledger claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import tempfile
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_DEFAULT_POINTS = (1000, 2500, 5000, 6000, 10000, 25000, 100000)
_DEFAULT_TRACES = (4, 12)
_COMPONENTS = ("total", "brem", "line", "coherent")

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<script src="plotly.min.js"></script></head>
<body style="margin:0"><div id="plot"></div>
<script>window.__FIG__ = __FIGURE_JSON__;</script></body></html>"""

_BENCH_JS = """
const [repeats, done] = [arguments[0], arguments[arguments.length - 1]];
const frames = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
(async () => {
  const fig = window.__FIG__;
  const div = document.getElementById("plot");
  const xs = fig.data[0].x;
  const lo = xs[0], hi = xs[xs.length - 1];
  const out = {render_ms: [], zoom_ms: [], webgl_renderer: null};
  try {
    const gl = document.createElement("canvas").getContext("webgl");
    const ext = gl && gl.getExtension("WEBGL_debug_renderer_info");
    out.webgl_renderer = gl ? (ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)
                                   : gl.getParameter(gl.RENDERER)) : "unavailable";
  } catch (e) { out.webgl_renderer = "error: " + e; }
  for (let i = 0; i < repeats + 1; i++) {
    Plotly.purge(div);
    await frames();
    let t0 = performance.now();
    await Plotly.newPlot(div, fig.data, fig.layout, {responsive: false});
    await frames();
    const render = performance.now() - t0;
    const mid = lo + 0.45 * (hi - lo);
    t0 = performance.now();
    await Plotly.relayout(div, {"xaxis.range": [mid, mid + 0.1 * (hi - lo)]});
    await frames();
    const zoom = performance.now() - t0;
    if (i > 0) { out.render_ms.push(render); out.zoom_ms.push(zoom); }  // i = 0 warms up
  }
  Plotly.purge(div);
  done(out);
})().catch(e => done({error: String(e)}));
"""


def _frame(points: int, traces: int):
    """Synthetic line-plus-continuum spectra on one shared grid per case."""
    import numpy as np
    import pandas as pd

    cases = max(1, traces // len(_COMPONENTS))
    energy = np.linspace(1000.0, 40000.0, points)
    rng = np.random.default_rng(622)
    rows = []
    for case in range(cases):
        brem = 1e3 * np.exp(-energy / (8000.0 + 1000.0 * case))
        line = np.zeros_like(energy)
        for centre in rng.uniform(3000.0, 35000.0, 12):
            line += 50.0 * np.exp(-0.5 * ((energy - centre) / 40.0) ** 2)
        coherent = 20.0 * (1.0 + np.sin(energy / 300.0 + case)) * np.exp(-energy / 20000.0)
        parts = {
            "total": brem + line + coherent,
            "brem": brem,
            "line": line,
            "coherent": coherent,
        }
        for component in _COMPONENTS:
            rows.append(
                pd.DataFrame(
                    {
                        "energy_eV": energy,
                        "intensity": parts[component],
                        "component": component,
                        "E0_keV": 30.0 + 10.0 * case,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _figure_json(points: int, traces: int, arm: str) -> str:
    import plotly.graph_objects as go

    from pyrite.plots.plotly.spectra import _trace_series

    fig = _trace_series(
        _frame(points, traces),
        hue="E0_keV",
        components=_COMPONENTS,
        width=720,
        height=360,
        title=f"{traces} x {points} points ({arm})",
        x_type="linear",
        y_type="linear",
        x_domain=None,
    )
    kind = go.Scattergl if arm == "webgl" else go.Scatter
    retyped = []
    for trace in fig.data:
        spec = trace.to_plotly_json()
        spec.pop("type", None)
        retyped.append(kind(**spec))
    fig = go.Figure(data=retyped, layout=fig.layout)
    return fig.to_json()


def _driver(browser: str, headful: bool):
    from selenium import webdriver

    if browser == "firefox":
        from selenium.webdriver.firefox.service import Service

        options = webdriver.FirefoxOptions()
        options.set_preference("webgl.sanitize-unmasked-renderer", False)
        if not headful:
            options.add_argument("-headless")
        gecko = shutil.which("geckodriver")
        return webdriver.Firefox(options=options, service=Service(gecko) if gecko else None)
    options = webdriver.ChromeOptions()
    # Without these, headless Chrome rasterizes WebGL in SwiftShader (software).
    options.add_argument("--ignore-gpu-blocklist")
    options.add_argument("--enable-gpu")
    if not headful:
        options.add_argument("--headless=new")
    binary = shutil.which("google-chrome") if browser == "chrome" else shutil.which("chromium")
    if binary:
        options.binary_location = binary
    return webdriver.Chrome(options=options)


def _median(values):
    return round(statistics.median(values), 2) if values else None


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102 - silence per-request logging
        pass


def run(args) -> dict:
    import plotly
    from plotly.offline import get_plotlyjs

    workdir = Path(tempfile.mkdtemp(prefix="pyrite-plotly-bench-"))
    (workdir / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    pages = {}
    for points in args.points:
        for traces in args.traces:
            for arm in ("svg", "webgl"):
                page = workdir / f"p{points}_t{traces}_{arm}.html"
                page.write_text(
                    _PAGE.replace("__FIGURE_JSON__", _figure_json(points, traces, arm)),
                    encoding="utf-8",
                )
                pages[(points, traces, arm)] = page

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=workdir))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/"
    record = {
        "issue": 622,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "plotly": plotly.__version__,
        "headful": args.headful,
        "repeats": args.repeats,
        "browsers": {},
    }
    for browser in args.browsers:
        driver = _driver(browser, args.headful)
        driver.set_script_timeout(args.timeout)
        rows = []
        try:
            caps = driver.capabilities
            meta = {"version": caps.get("browserVersion"), "webgl_renderer": None}
            for (points, traces, arm), page in pages.items():
                driver.get(base + page.name)
                result = driver.execute_async_script(_BENCH_JS, args.repeats)
                if "error" in result:
                    raise RuntimeError(f"{browser} {points}x{traces} {arm}: {result['error']}")
                meta["webgl_renderer"] = result["webgl_renderer"]
                row = {
                    "points_per_trace": points,
                    "traces": traces,
                    "arm": arm,
                    "render_ms_median": _median(result["render_ms"]),
                    "zoom_ms_median": _median(result["zoom_ms"]),
                    "render_ms": [round(v, 2) for v in result["render_ms"]],
                    "zoom_ms": [round(v, 2) for v in result["zoom_ms"]],
                }
                rows.append(row)
                print(
                    f"{browser:8s} {points:>7d} pts x {traces:>2d} {arm:5s} "
                    f"render {row['render_ms_median']:>9.1f} ms  zoom {row['zoom_ms_median']:>9.1f} ms",
                    flush=True,
                )
        finally:
            driver.quit()
        record["browsers"][browser] = {**meta, "rows": rows}
    server.shutdown()
    shutil.rmtree(workdir, ignore_errors=True)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--browsers", nargs="+", default=["firefox", "chrome"])
    parser.add_argument("--points", nargs="+", type=int, default=list(_DEFAULT_POINTS))
    parser.add_argument("--traces", nargs="+", type=int, default=list(_DEFAULT_TRACES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0, help="per-page script seconds")
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        import selenium  # noqa: F401
    except ImportError:
        print(
            "selenium missing: install the `browser` dependency group (`pixi install`)",
            file=sys.stderr,
        )
        return 2
    record = run(args)
    text = json.dumps(record, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
