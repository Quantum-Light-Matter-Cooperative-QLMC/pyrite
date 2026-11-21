"""Browser gate for the experimental Plotly Map cell selection (#622).

Run:
    uv run --with selenium==4.53.0 python checks/plotly_map_pilot_622.py \
        --grid 8 --out scratch/plotly_map_pilot_622_g8.json

Writes a synthetic azimuth x polar-tilt HOPG checkpoint (no transport: a
Gaussian line whose amplitude varies over the grid on a fixed bremsstrahlung
continuum, so some ``line_flux`` cells fall under the flux gate) into a fresh
``PYRITE_HOME``, launches ``analysis_app.py`` with ``marimo run --headless``,
and drives the Map view in each browser with real WebDriver pointer events
(Firefox via ``geckodriver``, Chrome via selenium-manager). Every step of the
pilot gate in the Plotly backend guide is recorded with what was observed:

* single click at a finite cell centre selects that geometry (and latency);
* a pan drag starting on another cell does not change the selection;
* clicking a flux-gated blank cell does not select it;
* changing map quantity or light/dark theme keeps a valid selection;
* changing beam energy or thickness clears the selection;
* Altair (stable) remains click-selectable after returning to it;
* keyboard: Tab focus plus Enter/Space on the map selects a cell or not;
* map relayout (zoom) latency.

Engineering evidence for a UI default decision; no numerical output, so no
validation-ledger claim. Exit 0 = ran (verdicts in the record), 2 = missing
optional ``selenium``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "src" / "pyrite" / "apps" / "analysis_app.py"

_DEEP_JS = """
window.__deep = function (sel) {
  const out = [], seen = new Set();
  const walk = (root) => {
    root.querySelectorAll(sel).forEach((e) => { if (!seen.has(e)) { seen.add(e); out.push(e); } });
    root.querySelectorAll("*").forEach((e) => { if (e.shadowRoot) walk(e.shadowRoot); });
  };
  walk(document);
  return out;
};
window.__visible = (e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
window.__mapGd = () => __deep(".js-plotly-plot").find(
  (g) => __visible(g) && (g.data || []).some((t) => t.name === "Select cell"));
"""


def build_fixture(workspace: Path, grid: int) -> int:
    """Write the synthetic map checkpoint under ``workspace``; return record count."""
    import numpy as np

    os.environ["PYRITE_HOME"] = str(workspace)
    from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
    from pyrite.checkpoints.persistence import checkpoint_path_for
    from pyrite.detectors import EnergyBins
    from pyrite.runs import run

    energy = np.linspace(1000.0, 30000.0, 2048)
    cases = build_cases(
        Sweep(
            material="hopg",
            thickness_ang=[10000.0, 20000.0],
            beam=BeamSpec(energy_keV=[30.0, 50.0]),
            tilt_deg=list(np.round(np.linspace(10.0, 40.0, grid), 4)),
            tilt_azim_deg=list(np.round(np.linspace(100.0, 170.0, grid), 4)),
            energy_bins=EnergyBins(cxr=energy, brem=energy),
        )
    )

    def payload(case):
        tilt, azimuth = case["tilt_deg"], case["tilt_azim_deg"]
        amplitude = (1.0 + np.cos(np.radians(4.0 * azimuth))) * np.exp(
            -(((tilt - 25.0) / 10.0) ** 2)
        ) + 0.005
        line = amplitude * 1e-4 * np.exp(-0.5 * ((energy - (5000.0 + 400.0 * tilt)) / 60.0) ** 2)
        brem = 1e-6 * np.exp(-energy / 8000.0)
        return {
            "E_grid": energy,
            "spec": line,
            "brem": brem,
            "E_grid_brem": energy,
            "brem_wide": brem,
            "eta": 0.05,
            "hit_frac": float(min(1.0, amplitude / 2.0)),
        }

    def runner(batch, callback, **_):
        for index, case in enumerate(batch):
            callback(index, case, payload(case))

    run.run_cases = runner
    with contextlib.redirect_stdout(io.StringIO()):
        assert run.run_sweep(cases, {}, checkpoint_path=checkpoint_path_for("hopg"), progress=False)
    return len(cases)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def marimo_app(workspace: Path):
    port = _free_port()
    env = {**os.environ, "PYRITE_HOME": str(workspace)}
    log = open(workspace / "marimo.log", "w")  # noqa: SIM115 - closed below
    proc = subprocess.Popen(
        [sys.executable, "-m", "marimo", "run", str(APP), "--headless", "--no-token"]
        + ["--port", str(port)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}/"
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            with contextlib.suppress(OSError):
                if urllib.request.urlopen(url, timeout=2).status == 200:
                    break
            time.sleep(0.5)
        else:
            raise RuntimeError("marimo did not start; see marimo.log")
        yield url
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        log.close()


def _driver(browser: str, headful: bool):
    from selenium import webdriver

    if browser == "firefox":
        from selenium.webdriver.firefox.service import Service

        options = webdriver.FirefoxOptions()
        if not headful:
            options.add_argument("-headless")
        options.add_argument("--width=1400")
        options.add_argument("--height=2400")
        gecko = shutil.which("geckodriver")
        return webdriver.Firefox(options=options, service=Service(gecko) if gecko else None)
    options = webdriver.ChromeOptions()
    if not headful:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,2400")
    binary = shutil.which("google-chrome") if browser == "chrome" else shutil.which("chromium")
    if binary:
        options.binary_location = binary
    return webdriver.Chrome(options=options)


class Page:
    """Thin helpers over marimo's shadow-DOM widgets."""

    def __init__(self, driver):
        self.d = driver

    def js(self, script, *args):
        self.d.execute_script(_DEEP_JS)
        return self.d.execute_script(script, *args)

    def wait(self, predicate, timeout=45.0, poll=0.1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                value = predicate()
                if value:
                    return value
            time.sleep(poll)
        return None

    def tab(self, label):
        el = self.wait(
            lambda: self.js(
                "return __deep('[role=tab]').find("
                "e => __visible(e) && e.innerText.trim() === arguments[0])",
                label,
            ),
            90,
        )
        if el is None:
            raise RuntimeError(f"no visible tab {label!r}")
        el.click()

    def select(self, option_text_prefix, value_text):
        """Choose ``value_text`` in the visible <select> owning an option with the prefix."""
        from selenium.webdriver.support.ui import Select

        el = self.js(
            "return __deep('select').find(s => __visible(s) && "
            "Array.from(s.options).some(o => o.text.startsWith(arguments[0])))",
            option_text_prefix,
        )
        if el is None:
            raise RuntimeError(f"no select with option {option_text_prefix!r}")
        index = self.d.execute_script(
            "return Array.from(arguments[0].options).findIndex(o => o.text.startsWith(arguments[1]))",
            el,
            value_text,
        )
        Select(el).select_by_index(index)

    def select_by_options(self, options, value_text):
        from selenium.webdriver.support.ui import Select

        el = self.js(
            "const want = JSON.parse(arguments[0]);"
            "const ok = (s) => want.every(w => Array.from(s.options).some(o => o.text.trim() === w))"
            " && s.options.length === want.length;"
            "const all = __deep('select').filter(ok);"
            "return all.find(__visible) || all[0];",
            json.dumps(options),
        )
        if el is None:
            raise RuntimeError(f"no select with options {options}")
        index = self.d.execute_script(
            "return Array.from(arguments[0].options).findIndex(o => o.text.startsWith(arguments[1]))",
            el,
            value_text,
        )
        Select(el).select_by_index(index)

    def heading(self):
        """Selected-cell heading text, '' for the placeholder, None when absent."""
        return self.js(
            "const s = __deep('strong').find(e => __visible(e) && e.innerText.trim() === 'Selected cell');"
            "if (s) return s.parentElement.innerText.trim();"
            "const p = __deep('em').find(e => __visible(e) && e.innerText.startsWith('Click a heatmap cell'));"
            "return p ? '' : null;"
        )

    def map_cells(self):
        """Grid axes and the finite (targetable) cells of the visible Plotly map."""
        return self.js(
            "const g = __mapGd(); if (!g) return null;"
            "const h = g.data[0], t = g.data[1];"
            "return {x: Array.from(h.x), y: Array.from(h.y),"
            " targets: Array.from(t.x).map((x, i) => [x, t.y[i]])};"
        )

    def cell_offset(self, x, y):
        """Offset from the map's centre to a data coordinate, in CSS pixels."""
        return self.js(
            "const g = __mapGd(), L = g._fullLayout, r = g.getBoundingClientRect();"
            "const px = L.xaxis._offset + L.xaxis.l2p(arguments[0]);"
            "const py = L.yaxis._offset + L.yaxis.l2p(arguments[1]);"
            "return [px - r.width / 2, py - r.height / 2];",
            x,
            y,
        )

    def map_element(self):
        return self.js("return __mapGd()")

    def click_cell(self, x, y):
        from selenium.webdriver import ActionChains

        dx, dy = self.cell_offset(x, y)
        gd = self.map_element()
        self.d.execute_script("arguments[0].scrollIntoView({block: 'center'})", gd)
        ActionChains(self.d).move_to_element_with_offset(gd, int(dx), int(dy)).pause(
            0.15
        ).click().perform()

    def drag_from_cell(self, x, y, by_px):
        from selenium.webdriver import ActionChains

        dx, dy = self.cell_offset(x, y)
        gd = self.map_element()
        ActionChains(self.d).move_to_element_with_offset(
            gd, int(dx), int(dy)
        ).click_and_hold().pause(0.1).move_by_offset(by_px // 2, 0).pause(0.05).move_by_offset(
            by_px - by_px // 2, 0
        ).pause(0.1).release().perform()


def _expected(x, y):
    return f"polar tilt {y:g} deg, azimuth {x:g} deg"


def _wait_heading(page, predicate, timeout=30.0):
    return page.wait(
        lambda: (lambda h: h if h is not None and predicate(h) else None)(page.heading()), timeout
    )


def _settle(page, seconds=2.5):
    """Let marimo finish a reactive rerun: heading/map stable for ``seconds``."""
    last, stable_since = object(), time.monotonic()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        current = (page.heading(), bool(page.map_element()))
        if current != last:
            last, stable_since = current, time.monotonic()
        elif time.monotonic() - stable_since >= seconds:
            return
        time.sleep(0.2)


def drive(page, url, grid):  # noqa: C901, PLR0915 - linear scripted scenario
    steps = {}

    def record(name, ok, **detail):
        verdict = None if ok is None else bool(ok)
        steps[name] = {"pass": verdict, **detail}
        label = "INFO" if verdict is None else ("PASS" if verdict else "FAIL")
        print(f"  {label} {name} {detail}", flush=True)

    page.d.get(url)
    page.wait(
        lambda: page.js("return __deep('[role=tab]').some(e => e.innerText.trim() === 'Map')"), 90
    )
    page.tab("Map")
    page.wait(
        lambda: page.js(
            "return __deep('select').some(s => Array.from(s.options).some(o => o.text === 'Plotly (experimental)'))"
        ),
        60,
    )
    page.select("Altair (stable)", "Plotly (experimental)")
    cells = page.wait(page.map_cells, 60)
    if not cells:
        record("plotly_map_rendered", False)
        return steps
    record(
        "plotly_map_rendered",
        True,
        grid=[len(cells["x"]), len(cells["y"])],
        targets=len(cells["targets"]),
    )
    _settle(page)

    # 1. single click at a finite cell centre
    targets = sorted(cells["targets"], key=lambda c: (abs(c[1] - 25.0), abs(c[0] - 135.0)))
    x0, y0 = targets[0]
    started = time.perf_counter()
    page.click_cell(x0, y0)
    got = _wait_heading(page, lambda h: _expected(x0, y0) in h)
    record(
        "click_selects_cell",
        got,
        cell=[x0, y0],
        latency_s=round(time.perf_counter() - started, 2),
        heading=page.heading(),
    )
    _settle(page)

    # 1b. click precision on several other cells (corners and interior)
    sample = [
        cells["targets"][i]
        for i in sorted(
            {
                0,
                len(cells["targets"]) - 1,
                len(cells["targets"]) // 3,
                2 * len(cells["targets"]) // 3,
            }
        )
    ]
    hits, latencies = 0, []
    for x, y in sample:
        started = time.perf_counter()
        page.click_cell(x, y)
        if _wait_heading(page, lambda h, x=x, y=y: _expected(x, y) in h, 20):
            hits += 1
            latencies.append(round(time.perf_counter() - started, 2))
        _settle(page, 1.5)
    record(
        "click_precision", hits == len(sample), hits=hits, tried=len(sample), latency_s=latencies
    )

    page.click_cell(x0, y0)
    _wait_heading(page, lambda h: _expected(x0, y0) in h)
    _settle(page)

    # 2. pan drag starting on another cell does not reselect
    other = next(c for c in targets[1:] if c[0] != x0 and c[1] != y0)
    before = page.heading()
    page.drag_from_cell(other[0], other[1], 80)
    _settle(page)
    after = page.heading()
    record(
        "pan_drag_keeps_selection",
        after == before and _expected(x0, y0) in (after or ""),
        heading=after,
    )
    # reset the view after panning
    page.js(
        "const g = __mapGd(); return Plotly.relayout(g, {'xaxis.autorange': true, 'yaxis.autorange': true})"
    )
    _settle(page, 1.0)

    # 3. map quantity change keeps a valid selection
    page.select("integrated coherent flux", "integrated flux under the dominant line")
    page.wait(
        lambda: page.map_cells() and page.map_cells()["targets"] != cells["targets"] or True, 20
    )
    _settle(page)
    gated_cells = page.map_cells()
    kept = page.heading()
    visible_after = [x0, y0] in gated_cells["targets"]
    record(
        "quantity_change_keeps_selection",
        (_expected(x0, y0) in (kept or "")) if visible_after else kept == "",
        heading=kept,
        cell_still_visible=visible_after,
    )

    # 4. a flux-gated blank cell is not selectable
    finite = {tuple(c) for c in gated_cells["targets"]}
    blanks = [(x, y) for x in gated_cells["x"] for y in gated_cells["y"] if (x, y) not in finite]
    if blanks:
        bx, by = blanks[len(blanks) // 2]
        before = page.heading()
        page.click_cell(bx, by)
        _settle(page)
        after = page.heading()
        record(
            "gated_cell_not_selectable",
            after == before,
            blank=[bx, by],
            blanks=len(blanks),
            before=before,
            after=after,
            selection_cleared=bool(before) and after == "",
        )
    else:
        record("gated_cell_not_selectable", False, note="fixture produced no gated cells")

    # restore quantity and selection
    page.select("integrated flux under the dominant line", "integrated coherent flux")
    _settle(page)
    if _expected(x0, y0) not in (page.heading() or ""):
        page.click_cell(x0, y0)
        _wait_heading(page, lambda h: _expected(x0, y0) in h)
        _settle(page)

    # 5. theme change keeps selection
    page.select_by_options(["System", "Light", "Dark"], "Dark")
    _settle(page, 3.0)
    record(
        "theme_change_keeps_selection",
        _expected(x0, y0) in (page.heading() or ""),
        heading=page.heading(),
    )
    page.select_by_options(["System", "Light", "Dark"], "Light")
    _settle(page, 3.0)

    # 6. energy and thickness changes clear the selection
    page.select_by_options(["30", "50"], "50")
    _settle(page, 3.0)
    record("energy_change_clears", page.heading() == "", heading=page.heading())
    page.select_by_options(["30", "50"], "30")
    _settle(page, 3.0)
    restored = page.heading()
    record(
        "energy_return_restores_selection",
        None,
        restored=_expected(x0, y0) in (restored or ""),
        heading=restored,
        note="stored choice is scope-bound: hidden, not discarded, while energy differs",
    )
    page.click_cell(x0, y0)
    _wait_heading(page, lambda h: _expected(x0, y0) in h)
    _settle(page)
    page.select_by_options(["1um", "2um"], "1um")
    _settle(page, 3.0)
    record("thickness_change_clears", page.heading() == "", heading=page.heading())

    # 7. keyboard: focus the map via Tab-reachable elements and press Enter/Space
    from selenium.webdriver.common.keys import Keys

    page.click_cell(x0, y0)  # reselect, then test whether keys move/choose another
    _wait_heading(page, lambda h: _expected(x0, y0) in h)
    _settle(page)
    focusable = page.js(
        "const g = __mapGd(); return Array.from(g.querySelectorAll('a,button,[tabindex],input,select'))"
        ".filter(e => e.tabIndex >= 0 && !e.closest('.modebar')).length"
    )
    gd = page.map_element()
    with contextlib.suppress(Exception):
        page.d.execute_script("arguments[0].setAttribute('tabindex','0'); arguments[0].focus()", gd)
        gd.send_keys(Keys.ARROW_RIGHT, Keys.ENTER, Keys.SPACE)
    _settle(page)
    record(
        "keyboard_cell_selection",
        False if focusable == 0 else None,
        focusable_non_modebar=focusable,
        heading=page.heading(),
        note="Plotly heatmap/scatter points expose no keyboard focus or selection",
    )

    # 8. zoom responsiveness on the map
    zoom_ms = page.js(
        "const g = __mapGd(), t = []; const xs = g.data[0].x, ys = g.data[0].y;"
        "const run = async () => { for (let i = 0; i < 5; i++) {"
        " const t0 = performance.now();"
        " await Plotly.relayout(g, {'xaxis.range': [xs[0], xs[Math.floor(xs.length / 2)]], 'yaxis.range': [ys[0], ys[Math.floor(ys.length / 2)]]});"
        " await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));"
        " t.push(performance.now() - t0);"
        " await Plotly.relayout(g, {'xaxis.autorange': true, 'yaxis.autorange': true}); }"
        " return t; }; return run();"
    )
    record("map_zoom_ms", True, zoom_ms=[round(v, 1) for v in zoom_ms])

    # 9. return to Altair: its own click selection still works
    page.select("Plotly (experimental)", "Altair (stable)")
    _settle(page, 3.0)
    vega = page.wait(
        lambda: page.js(
            "const v = __deep('.vega-embed').find(__visible); if (!v) return null;"
            "const marks = v.querySelectorAll('svg .mark-rect path, svg .mark-rect rect');"
            "return {svg: !!v.querySelector('svg.marks'), canvas: !!v.querySelector('canvas'), rects: marks.length};"
        ),
        45,
    )
    altair_ok = False
    if vega and vega["rects"]:
        el = page.js(
            "const v = __deep('.vega-embed').find(__visible);"
            "const m = v.querySelectorAll('svg .mark-rect path, svg .mark-rect rect');"
            "return m[Math.floor(m.length / 2)];"
        )
        from selenium.webdriver import ActionChains

        ActionChains(page.d).move_to_element(el).pause(0.15).click().perform()
        altair_ok = bool(_wait_heading(page, lambda h: h.startswith("Selected cell"), 30))
    elif vega and vega["canvas"]:
        from selenium.webdriver import ActionChains

        el = page.js("return __deep('.vega-embed').find(__visible).querySelector('canvas')")
        ActionChains(page.d).move_to_element(el).pause(0.15).click().perform()
        altair_ok = bool(_wait_heading(page, lambda h: h.startswith("Selected cell"), 30))
    record("altair_click_still_selects", altair_ok, vega=vega, heading=page.heading())
    return steps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--browsers", nargs="+", default=["firefox", "chrome"])
    parser.add_argument("--grid", type=int, default=8, help="azimuth and tilt points (>= 4)")
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--keep", type=Path, help="keep fixture/logs/screenshots in this directory")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.grid < 4:
        parser.error("--grid must be >= 4 (the Map heatmap needs a 4 x 4 sweep)")
    try:
        import selenium
    except ImportError:
        print("selenium missing: run with `uv run --with selenium==4.53.0 ...`", file=sys.stderr)
        return 2
    import marimo
    import plotly

    workspace = args.keep or Path(tempfile.mkdtemp(prefix="pyrite-map-pilot-"))
    workspace.mkdir(parents=True, exist_ok=True)
    n_records = build_fixture(workspace, args.grid)
    record = {
        "issue": 622,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid": args.grid,
        "records": n_records,
        "marimo": marimo.__version__,
        "plotly": plotly.__version__,
        "selenium": selenium.__version__,
        "headful": args.headful,
        "browsers": {},
    }
    with marimo_app(workspace) as url:
        for browser in args.browsers:
            print(f"{browser}:", flush=True)
            driver = _driver(browser, args.headful)
            try:
                page = Page(driver)
                steps = drive(page, url, args.grid)
                with contextlib.suppress(Exception):
                    driver.save_screenshot(str(workspace / f"{browser}_g{args.grid}.png"))
                record["browsers"][browser] = {
                    "version": driver.capabilities.get("browserVersion"),
                    "steps": steps,
                }
            finally:
                driver.quit()
    if not args.keep:
        shutil.rmtree(workspace, ignore_errors=True)
    text = json.dumps(record, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
