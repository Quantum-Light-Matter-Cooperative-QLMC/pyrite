"""Digitize Shi et al. 2024 Fig. 2b/d/f from the vector PDF (#1103).

The published curves are filled stroke outlines (red 0.753,0,0 = QM; black =
CL). Each outline is rasterized at 40 px/pt with a nonzero fill and
skeletonized; the medial axis of a constant-width stroke is its centreline,
so peaks and flanks carry no stroke-width bias. Per column the topmost
skeleton pixel is the curve value; empty columns (dotted strokes) are
linearly interpolated. Scale: tick-label spacing (offset-free); zero: the CL
curve's own flat skeleton. Peaks and areas are taken at full resolution; the
stored curves keep every 8th column (0.2 pt).

Writes ``tests/data/recoil/shi-2024-fig2.json``. Not part of any test run:
it needs the publisher PDF (not redistributed) and two extra packages:

  uv run --with pymupdf --with scikit-image python checks/digitize_shi_2024_fig2_1103.py PDF

Validation: shi-2024-fig2-intensity-attribution
"""

import hashlib
import json
import sys
from pathlib import Path

import fitz
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from skimage.morphology import skeletonize

PDF = sys.argv[1]
OUT = Path(__file__).resolve().parents[1] / "tests/data/recoil/shi-2024-fig2.json"
PX = 40.0  # px per pt
RED, BLACK = (0.75294, 0.0, 0.0), (0.0, 0.0, 0.0)
PANELS = {
    "b": dict(
        T_keV=5000.0,
        yr=(95, 155),
        seg=[
            (330, 398, (345.33, 3.6), (389.73, 4.0), (151.63 - 105.9) / 4, 1e-11, -2),
            (398, 466, (408.71, 5.4), (452.86, 5.8), (151.51 - 105.78) / 4, 1e-11, -3),
        ],
    ),
    "d": dict(
        T_keV=300.0,
        yr=(188, 254),
        seg=[
            (330, 392, (344.15, 2.8), (384.49, 3.2), (250.9 - 196.68) / 6, 1e-11, -2),
            (392, 466, (401.52, 4.2), (445.45, 4.6), (251.53 - 196.61) / 2, 1e-10, -3),
        ],
    ),
    "f": dict(
        T_keV=30.0,
        yr=(290, 349),
        seg=[
            (330, 379.0, (352.9, 1.2), (440.04, 2.0), (346.52 - 300.79) / 1, 1e-9, -2),
            (379.0, 466, (352.9, 1.2), (440.04, 2.0), (346.66 - 294.29) / 4, 1e-9, -3),
        ],
    ),
}


def to_path(drs):
    verts, codes = [], []
    for dr in drs:
        last = None
        for it in dr["items"]:
            if it[0] == "re":
                r = it[1]
                verts += [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]
                codes += [
                    MplPath.MOVETO,
                    MplPath.LINETO,
                    MplPath.LINETO,
                    MplPath.LINETO,
                    MplPath.CLOSEPOLY,
                ]
                last = None
                continue
            p0 = it[1]
            if last is None or abs(p0.x - last.x) > 1e-6 or abs(p0.y - last.y) > 1e-6:
                verts.append((p0.x, p0.y))
                codes.append(MplPath.MOVETO)
            if it[0] == "l":
                verts.append((it[2].x, it[2].y))
                codes.append(MplPath.LINETO)
                last = it[2]
            elif it[0] == "c":
                verts += [(it[2].x, it[2].y), (it[3].x, it[3].y), (it[4].x, it[4].y)]
                codes += [MplPath.CURVE4] * 3
                last = it[4]
    return MplPath(verts, codes)


def raster(path, x0, x1, y0, y1):
    w, h = (x1 - x0) * PX, (y1 - y0) * PX
    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # PDF y grows downward
    ax.add_patch(PathPatch(path, facecolor="k", edgecolor="none", lw=0, antialiased=False))
    cv = FigureCanvasAgg(fig)
    cv.draw()
    img = np.asarray(cv.buffer_rgba())[:, :, 0] < 128
    return img


page = fitz.open(PDF)[4]
drawings = page.get_drawings()


def close(c, ref):
    return c is not None and all(abs(a - b) < 2e-3 for a, b in zip(c, ref, strict=True))


curves = {}
for name, P in PANELS.items():
    y0, y1 = P["yr"]
    zero = {}
    for colour, label in ((BLACK, "CL"), (RED, "QM")):
        drs = [
            d
            for d in drawings
            if d["type"] == "f"
            and close(d.get("fill"), colour)
            and d["rect"].y0 >= y0
            and d["rect"].y1 <= y1
            and d["rect"].width > 50
            and len(d["items"]) > 20
        ]
        path = to_path(drs)
        for xlo, xhi, (xa, ea), (xb, eb), yscale, unit, nz in P["seg"]:
            img = raster(path, xlo, xhi, y0, y1)
            sk = skeletonize(img)
            cols = np.where(sk.any(axis=0))[0]
            top_px = np.array([np.argmax(sk[:, c]) for c in cols], float)
            ypt = y0 + (top_px + 0.5) / PX
            xpt = xlo + (cols + 0.5) / PX
            if label == "CL":
                zero[nz] = float(np.median(np.sort(ypt)[-max(10, ypt.size // 4) :]))
            xs = xlo + (np.arange(img.shape[1]) + 0.5) / PX
            yfull = np.interp(xs, xpt, ypt)
            val = np.clip(zero[nz] - yfull, 0, None) / yscale * unit
            E = ea + (xs - xa) * (eb - ea) / (xb - xa)
            m = (E > E[np.searchsorted(xs, xpt.min())]) & (xs <= xpt.max())
            E, val = E[m], val[m]
            dE = (E[1] - E[0]) * 1e3
            key = f"{P['T_keV']:g} keV {label} nz={nz}"
            curves[key] = dict(
                panel=name,
                T_keV=P["T_keV"],
                curve=label,
                nz=nz,
                area_per_sr=float(val.sum() * dE),
                peak=float(val.max()),
                peak_eV=float(E[val.argmax()] * 1e3),
                E_eV=(E[::8] * 1e3).round(2).tolist(),
                y=[float(f"{v:.4g}") for v in val[::8]],
            )
            print(
                f"{key:24s} peak {val.max():.3g} @ {E[val.argmax()] * 1e3:7.1f} eV  area {val.sum() * dE:.3g}  E {E[0] * 1e3:.0f}-{E[-1] * 1e3:.0f}"
            )
SOURCE = {
    "doi": "10.1038/s41467-024-52050-w",
    "citation": "Shi, Wong, Huang, Wong & Kaminer, Transverse recoil imprinted on free-electron radiation, Nat. Commun. 15, 7803 (2024), Fig. 2b/d/f",
    "zotero_item": "Q4I3JISU",
    "zotero_attachment": "4G9LUENI",
    "pdf_sha256": hashlib.sha256(Path(PDF).read_bytes()).hexdigest(),
    "verified": "2026-11-12",
    "method": "vector-path extraction; see checks/digitize_shi_2024_fig2_1103.py",
}
CONVENTIONS = {
    "geometry": "straight 30 nm Si film, beam along [001], photon in x-z plane at theta = 100 deg",
    "units": "E_eV photon energy [eV]; y, peak photons/(eV sr electron); area_per_sr photons/(sr electron) over the stored E range",
    "reflection": "paper label n_z with g = n 2 pi/a; PyRITE rows (h,k,|n_z|) are the negation",
    "curves": "CL = paper's classical Eq. (1); QM = paper's Eq. (3). n_z=-2 sits in the left half, n_z=-3 in the right; 30 keV shares one x axis, split at 1.45 keV",
    "axes": "n_z=-2 is read on the left y axis and n_z=-3 on the right axis (panels d, f carry different scales). The right-axis reading of d/f n_z=-3 is ambiguous: an independent model of the paper's own inputs matches the n_z=-3 QM heights read on the left axis (#1103). QM/CL area ratios within one n_z group are axis-independent.",
    "uncertainty": "peaks and positions reproduce manual reads within 3 %; areas depend on the stored E window and on dotted-stroke interpolation",
}
OUT.write_text(
    json.dumps({"source": SOURCE, "conventions": CONVENTIONS, "curves": curves}, indent=1) + "\n"
)
print(f"wrote {OUT}")
