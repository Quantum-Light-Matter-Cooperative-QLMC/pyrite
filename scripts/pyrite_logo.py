#!/usr/bin/env python3
"""Generate a physics-derived PyRITE SVG + PNG wordmark.

Run from a PyRITE checkout with the reference tables installed:

    PYRITE_MC_BACKEND=cpu uv run python scripts/pyrite_logo.py --ri-gap 0.10

Adjust --ri-gap to add space before the spectral I (logo units; 0 restores
the original spacing). Cached simulations are reused when only spacing changes.

Requires pyrite-mc, NumPy and Matplotlib (e.g. pyrite-mc[plots]).  The two
simulations are cached as NPZ so typography changes do not repeat transport.

The Y is a *selected* set of genuine electron tracks at normal incidence;
the lateral projection is independently magnified for visual readability.
The I is the normalized, background-subtracted shape of a genuine incoherent
PXR/CBS spectral peak. It is NOT an experimentally calibrated spectrum glyph.
The Y source is a uniformly sampled, collimated rectangular top-hat beam.
Because the Y uses a laterally homogeneous infinite slab at normal incidence,
translating each simulated history by its sampled entrance offset is EXACTLY
physically equivalent to launching it from that point; it doesn't change the
interaction RNG or scattering. The source positions are never Gaussian.
The upstream beam stem, glow, display colormap and typography are artistic.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

# Choose the backend before importing any PyRITE components.
os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from PIL import Image, ImageDraw, ImageOps

BG = "#0C1628"
INK = "#F5F5F0"
ACCENT = "#FFC858"
PALETTE = LinearSegmentedColormap.from_list(
    "pyrite_kinetic",
    [
        (0.00, "#4A58E8"),  # blue-violet: lower-energy tails
        (0.25, "#357EFF"),  # saturated blue
        (0.50, "#06C6EF"),  # electric cyan
        (0.74, "#31EAD2"),  # bright turquoise
        (0.89, "#8EF0C7"),  # pale teal, before the energetic gold
        (1.00, ACCENT),  # incident-electron energy: matches the stem
    ],
)
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "pyrite" / "_assets" / "branding"
CACHE_SCHEMA = 1  # Transport cache remains reusable: beam offsets are separate.


def rounded_image(image: Image.Image, corner_fraction: float) -> Image.Image:
    """Mask a raster logo to a gently rounded canvas, with antialiased edges.

    The source is not resimulated or redrawn: only the outside background
    corners become transparent. The social sharing card stays opaque navy.
    """
    if corner_fraction == 0:
        return image
    image = image.convert("RGBA")
    width, height = image.size
    scale = 4
    mask = Image.new("L", (width * scale, height * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, width * scale - 1, height * scale - 1),
        radius=round(corner_fraction * min(width, height) * scale),
        fill=255,
    )
    image.putalpha(mask.resize((width, height), Image.Resampling.LANCZOS))
    return image


def social_card_image(image: Image.Image, logo_width_fraction: float) -> Image.Image:
    """Center the wordmark inside GitHub's safe area on an opaque navy card."""
    content = ImageOps.contain(
        image,
        (round(1200 * logo_width_fraction), 400),
        Image.Resampling.LANCZOS,
    )
    card = Image.new("RGB", (1200, 630), BG)
    card.paste(content.convert("RGB"), ((1200 - content.width) // 2, (630 - content.height) // 2))
    return card


def round_svg_canvas(path: Path, corner_fraction: float) -> None:
    """Apply matching rounded bounds to Matplotlib's scalable SVG master."""
    if corner_fraction == 0:
        return
    svg = path.read_text(encoding="utf-8")
    viewbox = re.search(r'\bviewBox="([^"]+)"', svg)
    marker = '<g id="figure_1">'
    if viewbox is None or marker not in svg:
        raise ValueError("Unexpected Matplotlib SVG layout; cannot round logo canvas")
    x, y, width, height = map(float, viewbox.group(1).split())
    radius = corner_fraction * min(width, height)
    clip = (
        '<defs><clipPath id="pyrite-rounded-canvas">'
        f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
        f'rx="{radius:g}" ry="{radius:g}"/>'
        "</clipPath></defs>\n"
    )
    svg = svg.replace(
        marker,
        clip + '<g id="figure_1" clip-path="url(#pyrite-rounded-canvas)">',
        1,
    )
    path.write_text(svg, encoding="utf-8")


def _cache_key(tag: str, config: dict) -> str:
    record = json.dumps({"schema": CACHE_SCHEMA, "tag": tag, **config}, sort_keys=True)
    return hashlib.sha256(record.encode()).hexdigest()[:14]


def cached_arrays(path: Path, calculate, *, force: bool) -> dict[str, np.ndarray]:
    if path.is_file() and not force:
        print(f"Loading cached data: {path}")
        with np.load(path, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}
    print(f"Running PyRITE simulation for {path.name} ...")
    data = calculate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"Cached: {path}")
    return data


def simulate_y(
    *,
    material: str,
    energy_keV: float,
    thickness_ang: float,
    electrons: int,
    seed: int,
    max_steps: int,
) -> dict[str, np.ndarray]:
    """Run actual MC transport; return only plotting/provenance-relevant arrays.

    Coordinate convention: beam along +z, sample occupies 0 <= z <= thickness.
    The apparent beam entering upward in the logo is a *layout transformation*.
    """
    from pyrite.materials import CATALOG
    from pyrite.transport import simulate_trajectories

    composition = CATALOG.crystal(material).composition
    raw = simulate_trajectories(
        E0_keV=energy_keV,
        Ne=electrons,
        thickness_ang=thickness_ang,
        composition=composition,
        seed=seed,
        max_steps=max_steps,
        E_cut_keV=5.0,
        beam_dir=np.array([0.0, 0.0, 1.0]),
    )
    length = np.asarray(raw["L_ang"], float)
    direction = np.asarray(raw["v_hat"], float)
    center = np.asarray(raw["r_mid"], float)
    start = center - 0.5 * length[:, None] * direction
    end = center + 0.5 * length[:, None] * direction
    return {
        "start": start,
        "end": end,
        "energy_keV": np.asarray(raw["E_keV"], float),
        "electron_id": np.asarray(raw["elec_id"], int),
        "t_ang": np.asarray(raw["t_ang"], float),
        "initial_r_ang": np.asarray(raw["initial_r_ang"], float),
        "initial_v_hat": np.asarray(raw["initial_v_hat"], float),
    }


def simulate_i(*, energy_keV: float, thickness_ang: float, electrons: int) -> dict[str, np.ndarray]:
    """Run actual PXR + CBS on a fine explicit energy grid (no detector blur)."""
    import pyrite as pr
    from pyrite.beams import PointBunch, PointSpot
    from pyrite.config import CxrReflections, FixedSampling
    from pyrite.detectors import Detector, EnergyBins
    from pyrite.geometry import Footprint, Slab

    # Around the catalog HOPG basal reflection near 850-900 eV at 30 keV.
    grid = np.arange(700.0, 1100.01, 0.5)
    result = pr.simulate(
        pr.Beam(energy_keV=energy_keV, transverse=PointSpot(), longitudinal=PointBunch()),
        Slab(
            "hopg",
            thickness_ang=thickness_ang,
            tilt_deg=45.0,
            tilt_azim_deg=180.0,
            footprint=Footprint(5.0, 5.0),
            mosaic=False,
        ),
        Detector(observation_angle_deg=90.0),
        numerics=pr.Numerics(
            sampling=FixedSampling(cxr_electrons=electrons, brem_electrons=32),
            energy_bins=EnergyBins(cxr=grid, brem=np.array([700.0, 1100.0])),
            cxr_reflections=CxrReflections(n_families=1),
        ),
    )
    line = result.cxr.incoherent
    return {
        "energy_eV": np.asarray(line.photon_energy_eV, float),
        "density": np.asarray(line.density, float),
        "resolved_seed": np.array(int(result.case["seed"]), dtype=np.int64),
    }


def selected_y_segments(data: dict[str, np.ndarray], *, thickness_ang: float, n_per_arm: int = 28):
    ids = np.asarray(data["electron_id"], int)
    order = np.lexsort((data["t_ang"], ids))
    end_positions = np.r_[np.flatnonzero(np.diff(ids[order]) != 0), len(order) - 1]
    last_rows = order[end_positions]
    last_ids = ids[last_rows]
    exits = np.asarray(data["end"])[last_rows]
    transmitted = np.isclose(exits[:, 2], thickness_ang, atol=max(1.0, thickness_ang * 1e-5))
    dx = exits[:, 0] - np.asarray(data["initial_r_ang"])[last_ids, 0]
    valid_ids = last_ids[transmitted]
    dx = dx[transmitted]
    if len(valid_ids) < 10:
        raise RuntimeError(
            "Too few transmitted histories for Y: try a thinner target or more electrons"
        )

    selected: list[int] = []
    for sign in (-1, 1):
        mask = sign * dx > 0
        side_ids, magnitudes = valid_ids[mask], sign * dx[mask]
        if len(side_ids) < 3:
            raise RuntimeError(
                f"No usable {'left' if sign == -1 else 'right'} arm; try another seed"
            )
        side_ids = side_ids[np.argsort(magnitudes)]
        # Exclude near-axis tracks; retain a spread of true positive/negative tracks.
        side_ids = side_ids[int(0.35 * len(side_ids)) :]
        picks = np.linspace(0, len(side_ids) - 1, min(n_per_arm, len(side_ids)))
        selected.extend(side_ids[np.rint(picks).astype(int)].tolist())

    selected_ids = np.unique(np.asarray(selected, dtype=int))
    keep = np.isin(ids, selected_ids)
    # One shared scale, inferred from actual selected spatial extents.
    lateral = np.r_[data["start"][keep, 0], data["end"][keep, 0]]
    half_width_ang = max(float(np.quantile(np.abs(lateral), 0.975)), 1.0)
    return keep, selected_ids, half_width_ang


def uniform_beam_offsets(n_electrons: int, full_width_ang: float, seed: int) -> np.ndarray:
    """Uniform 2D rectangular top-hat at the entrance plane, in angstrom.

    This is an independent deterministic RNG stream, distinct from PyRITE's
    collision stream.  X and Y are each uniform on [-width/2, +width/2],
    with no angular divergence; a projected view thus has a flat X density.
    """
    if n_electrons <= 0 or not np.isfinite(full_width_ang) or full_width_ang <= 0:
        raise ValueError("Need a positive electron count and finite beam width")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xBEA7, 0x1050]))
    offsets = np.zeros((n_electrons, 3), dtype=float)
    offsets[:, :2] = rng.uniform(-0.5 * full_width_ang, 0.5 * full_width_ang, size=(n_electrons, 2))
    return offsets


def translate_y_histories(
    data: dict[str, np.ndarray], offsets: np.ndarray
) -> dict[str, np.ndarray]:
    """Install the top-hat source *physically* in every stored flight segment.

    For the laterally invariant, ungrooved, field-free infinite slab used in the
    logo, r(t) -> r(t) + r0 leaves all collision distances, directions, energy
    losses and exit outcomes unchanged. This is the exact uniform-source MC
    solution, not a visual spread of the paths or an artificial fork.
    """
    ids = np.asarray(data["electron_id"], dtype=int)
    entrance = np.asarray(data["initial_r_ang"], dtype=float)
    if offsets.shape != entrance.shape or np.any(ids < 0) or np.any(ids >= len(offsets)):
        raise ValueError("Source offsets must match the simulation electron IDs")
    transformed = dict(data)
    transformed["initial_r_ang"] = entrance + offsets
    transformed["start"] = np.asarray(data["start"], dtype=float) + offsets[ids]
    transformed["end"] = np.asarray(data["end"], dtype=float) + offsets[ids]
    return transformed


def dominant_line(line: dict[str, np.ndarray]):
    """Find the strongest positive central peak and a roughly local FWHM.

    This does not assert that the computed spectrum contains only one reflection.
    """
    energy = np.asarray(line["energy_eV"], float)
    density = np.asarray(line["density"], float)
    order = np.argsort(energy)
    energy, density = energy[order], density[order]
    density = np.where(np.isfinite(density), np.maximum(density, 0.0), 0.0)
    valid = (energy >= 750.0) & (energy <= 1000.0)
    if np.count_nonzero(valid) < 20:
        raise RuntimeError("CXR energy grid does not cover the expected HOPG line region")
    base = float(np.quantile(density[valid], 0.15))
    contrast = np.maximum(density - base, 0.0)
    candidates = np.flatnonzero(valid)
    peak = int(candidates[np.argmax(contrast[valid])])
    maximum = float(contrast[peak])
    if maximum <= 0 or peak in (0, len(energy) - 1):
        raise RuntimeError("No usable PXR/CBS peak; inspect the HOPG geometry and reference tables")
    half = maximum * 0.5
    left = peak
    right = peak
    while left > 0 and contrast[left] >= half:
        left -= 1
    while right < len(contrast) - 1 and contrast[right] >= half:
        right += 1
    if left == 0 or right == len(contrast) - 1:
        raise RuntimeError("Peak FWHM not resolved inside spectral window")
    x_left = np.interp(half, [contrast[left], contrast[left + 1]], [energy[left], energy[left + 1]])
    x_right = np.interp(
        half, [contrast[right], contrast[right - 1]], [energy[right], energy[right - 1]]
    )
    fwhm = float(x_right - x_left)
    if not (0 < fwhm < 150):
        raise RuntimeError(f"Unresolved / overly broad line (FWHM={fwhm:g} eV); adjust geometry")
    return energy, contrast / maximum, float(energy[peak]), fwhm


def draw_glyph(
    ax, letter: str, x: float, baseline: float, height: float, color: str, font_family: str
) -> float:
    # Explicit glyph outlines, not font-dependent SVG <text> placement.
    font = FontProperties(family=font_family, weight="bold")
    path = TextPath((0, 0), letter, size=1.0, prop=font)
    box = path.get_extents()
    scale = height / box.height
    transform = (
        Affine2D().translate(-box.x0, -box.y0).scale(scale).translate(x, baseline) + ax.transData
    )
    ax.add_patch(PathPatch(path, transform=transform, facecolor=color, edgecolor="none"))
    return box.width * scale


def draw_hbn_e(
    ax,
    *,
    x: float,
    baseline: float,
    cap: float,
    width: float,
    tilt_deg: float = 13.0,
    azimuth_deg: float = 10.0,
) -> dict[str, float | int | str]:
    """Three matching, shallowly viewed CIF-derived hBN monolayers.

    The bulk hBN CIF uses AA' stacking, exchanging B/N registry in alternating
    planes. Repeat ONE catalog-derived basal plane here instead: the logo's
    three bars should share the same visible honeycomb, not imply two different
    patterns. This visual AA registration is artistic, not the bulk structure.
    """
    from pyrite.materials import CATALOG
    from pyrite.plots.plotly.crystal_lattice import crystal_atom_sites, crystal_bonds

    spec = CATALOG.crystal("hbn")
    elements_full, coords_full = crystal_atom_sites(spec, n_a=7, n_b=3, n_c=1)
    mask = np.isclose(coords_full[:, 2], coords_full[:, 2].min())
    coords = coords_full[mask]
    elements = np.asarray(elements_full)[mask]
    bonds = crystal_bonds(elements.tolist(), coords)
    if not len(elements) or not bonds:
        raise RuntimeError("Could not construct the hBN basal sheet")

    # Rotate the real basal-plane coordinates in azimuth, then project a shallow
    # polar elevation. The modest depth exaggeration keeps bonds legible when
    # the logo is reduced to a narrow documentation header.
    phi = np.deg2rad(azimuth_deg)
    u = np.cos(phi) * coords[:, 0] + np.sin(phi) * coords[:, 1]
    depth = -np.sin(phi) * coords[:, 0] + np.cos(phi) * coords[:, 1]
    pad = 0.069  # leave room for the slightly enlarged boundary atoms
    scale = (width - 2 * pad) / np.ptp(u)
    px = x + pad + scale * (u - u.min())
    depth_offset = 2.2 * scale * np.sin(np.deg2rad(tilt_deg)) * (depth - depth.mean())
    row_spacing = (cap - baseline - 0.45) / 2.0  # 0.28) / 2.0
    row_heights = [cap - 0.24 - row * row_spacing for row in range(3)]

    segments = [
        [(px[a], py + depth_offset[a]), (px[b], py + depth_offset[b])]
        for py in row_heights
        for a, b in bonds
    ]
    ax.add_collection(
        LineCollection(segments, colors="#B6D6D8", linewidths=1.5, alpha=0.89, zorder=3)
    )

    # Identical B/N motifs on all three sheets; paint back-to-front within each.
    for py in row_heights:
        for idx in np.argsort(depth):
            element = elements[idx]
            radius = 0.050 if element == "B" else 0.056
            fill = "#1EDCF5" if element == "B" else "#FFD064"
            center = (px[idx], py + depth_offset[idx])
            ax.add_patch(
                Circle(
                    center,
                    radius,
                    facecolor=fill,
                    edgecolor="#233A51",
                    linewidth=0.45,
                    zorder=4,
                )
            )
            ax.add_patch(
                Circle(
                    (center[0] - 0.27 * radius, center[1] + 0.28 * radius),
                    0.30 * radius,
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.22,
                    zorder=5,
                )
            )

    return {
        "e_material": "hbn",
        "e_sheets": 3,
        "e_atoms": 3 * len(elements),
        "e_bonds": 3 * len(bonds),
        "e_tilt_deg": float(tilt_deg),
        "e_azimuth_deg": float(azimuth_deg),
    }


def draw_y(
    ax,
    y: dict[str, np.ndarray],
    *,
    x_center: float,
    y_entry: float,
    y_bottom: float,
    y_cap: float,
    thickness_ang: float,
    energy_keV: float,
    stem_width: float,
    color_gamma: float,
    beam_seed: int,
    beam_width_nm: float | None = None,
):
    # Resolve selection/scale from the *unshifted* MC transport. Selecting on
    # displacement relative to the launch (rather than final absolute X)
    # ensures left/right scattering selection is independent of beam position.
    keep, chosen, halfwidth = selected_y_segments(y, thickness_ang=thickness_ang)
    left, right = x_center - 0.83, x_center + 0.83
    width = (right - left) * 0.47

    # Make one consistent physical-to-display transform. By default, derive
    # the PHYSICAL uniform beam width so its sampled entry points span exactly
    # the requested displayed stem. Optionally override the physical width;
    # the drawn stem then adopts the corresponding physical extent.
    if beam_width_nm is None:
        beam_full_width_ang = stem_width * halfwidth / width
    else:
        beam_full_width_ang = 10.0 * beam_width_nm
        stem_width = beam_full_width_ang * width / halfwidth
    offsets = uniform_beam_offsets(len(y["initial_r_ang"]), beam_full_width_ang, beam_seed)
    uniform_y = translate_y_histories(y, offsets)
    start, end = uniform_y["start"][keep], uniform_y["end"][keep]

    def plot_x(x):
        return x_center + width * x / halfwidth

    def plot_z(z):
        return y_entry + (y_cap - y_entry) * z / thickness_ang

    segments = np.stack(
        [
            np.column_stack((plot_x(start[:, 0]), plot_z(start[:, 2]))),
            np.column_stack((plot_x(end[:, 0]), plot_z(end[:, 2]))),
        ],
        axis=1,
    )
    # Use the actual kinetic energy but increase the usable visual range.  A
    # percentile lower bound is appropriate for artwork with no physical
    # colorbar; the mapping remains monotonic, and E0 always maps to gold.
    selected_energy = np.asarray(y["energy_keV"][keep], dtype=float)
    energy_floor = min(0.97 * energy_keV, max(5.0, float(np.quantile(selected_energy, 0.10))))
    norm = PowerNorm(gamma=color_gamma, vmin=energy_floor, vmax=energy_keV, clip=True)
    colors = PALETTE(norm(selected_energy))

    bounds = Rectangle(
        (left, y_entry - 0.04),
        right - left,
        y_cap - y_entry + 0.08,
        transform=ax.transData,
        facecolor="none",
        edgecolor="none",
    )
    ax.add_patch(bounds)
    glow = LineCollection(segments, colors=colors, linewidths=3.5, alpha=0.07, capstyle="round")
    crisp = LineCollection(segments, colors=colors, linewidths=0.85, alpha=0.74, capstyle="round")
    for collection in (glow, crisp):
        collection.set_clip_path(bounds)
        ax.add_collection(collection)

    # The physical, flat-top incident beam occupies this EXACT width when
    # projected to the wordmark's units. The rectangle is a continuous beam
    # fill, not an increased linewidth on the individual particle trajectories.
    stem_color = PALETTE(1.0)  # exactly the high-energy trajectory color
    stem_height = y_entry - y_bottom
    for width_mult, opacity, layer in ((2.6, 0.035, 5), (1.5, 0.12, 6), (1.0, 0.96, 7)):
        ax.add_patch(
            Rectangle(
                (x_center - width_mult * stem_width / 2, y_bottom),
                width_mult * stem_width,
                stem_height,
                facecolor=stem_color,
                edgecolor="none",
                alpha=opacity,
                zorder=layer,
            )
        )
    # Continue the SAME sampled electrons straight back into vacuum: each
    # upstream ray meets its genuine Monte Carlo history at its own entry X.
    # Individual ray linewidths are unchanged (0.85), matching the material
    # segments; the filled background retains the legible solid-letter stem.
    entry_x = plot_x(uniform_y["initial_r_ang"][chosen, 0])
    beam_rays = np.stack(
        (
            np.column_stack((entry_x, np.full_like(entry_x, y_bottom))),
            np.column_stack((entry_x, np.full_like(entry_x, y_entry))),
        ),
        axis=1,
    )
    ax.add_collection(
        LineCollection(
            beam_rays,
            colors=[PALETTE(1.0)],
            linewidths=0.85,
            alpha=0.28,
            capstyle="butt",
            zorder=8,
        )
    )
    return {
        "trajectories_selected": int(len(chosen)),
        "y_lateral_scale_ang": halfwidth,
        "y_energy_color_floor_keV": energy_floor,
        "y_beam_full_width_ang": float(beam_full_width_ang),
        "y_beam_full_width_nm": float(beam_full_width_ang / 10.0),
        "y_stem_rendered_width": float(stem_width),
        "y_beam_sampled_x_min_ang": float(uniform_y["initial_r_ang"][chosen, 0].min()),
        "y_beam_sampled_x_max_ang": float(uniform_y["initial_r_ang"][chosen, 0].max()),
    }


def draw_i(
    ax,
    line: dict[str, np.ndarray],
    *,
    x_center: float,
    baseline: float,
    cap: float,
    width_scale: float,
):
    energy, contrast, peak, fwhm = dominant_line(line)
    # Keep the full spectral wings for now: only the explicit typographic caps
    # have been removed.  Widening the horizontal mapping thickens the I while
    # preserving the simulated intensity distribution and its natural tails.
    half_range = max(4.0 * fwhm, 10.0)
    mask = np.abs(energy - peak) <= half_range
    xx = x_center + width_scale * (energy[mask] - peak) / half_range
    yy = baseline + (cap - baseline) * np.power(contrast[mask], 0.78)
    # The 1-D source spectrum is the luminous vertical I stroke (height is
    # normalized intensity; horizontal placement is photon energy).
    ax.fill_between(xx, baseline, yy, color=ACCENT, alpha=0.86, lw=0, zorder=4)
    ax.plot(xx, yy, color="#F8DBA0", lw=6.5, alpha=0.10, zorder=5)
    ax.plot(xx, yy, color="#FFF1CD", lw=1.35, alpha=0.98, zorder=6)
    return {
        "line_peak_eV": peak,
        "line_fwhm_eV": fwhm,
        "line_density_peak": float(np.nanmax(line["density"])),
    }


def render(
    y_data,
    line_data,
    *,
    thickness_ang: float,
    energy_keV: float,
    out: Path,
    dpi: int,
    font_family: str,
    stem_width: float = 0.075,
    i_width: float = 0.45,
    ri_gap: float = 0.17,
    color_gamma: float = 1.85,
    beam_seed: int = 17,
    beam_width_nm: float | None = None,
    e_tilt_deg: float = -13.0,
    e_azimuth_deg: float = 0.0,
):
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["svg.hashsalt"] = "pyrite-brand-v1"
    plt.rcParams["path.simplify"] = False  # do not simplify the physical paths
    fig, ax = plt.subplots(figsize=(14.2, 3.1), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    baseline, cap = 0.81, 2.70
    letter_height, gap = cap - baseline, 0.19
    cursor = 0.45
    cursor += draw_glyph(ax, "P", cursor, baseline, letter_height, INK, font_family) + gap
    y_center = cursor + 0.85
    cursor += 1.70 + gap
    cursor += draw_glyph(ax, "R", cursor, baseline, letter_height, INK, font_family) + gap
    cursor += ri_gap  # Optical correction; shift I, T and E together to preserve I–T spacing.
    i_center = cursor + 0.34
    cursor += 0.68 + gap
    cursor += draw_glyph(ax, "T", cursor, baseline, letter_height, INK, font_family) + gap
    # Match the original E glyph's advance width, but render no glyph outline.
    font = FontProperties(family=font_family, weight="bold")
    e_bounds = TextPath((0, 0), "E", size=1.0, prop=font).get_extents()
    e_width = letter_height * e_bounds.width / e_bounds.height
    e_stats = draw_hbn_e(
        ax,
        x=cursor,
        baseline=baseline,
        cap=cap,
        width=e_width,
        tilt_deg=e_tilt_deg,
        azimuth_deg=e_azimuth_deg,
    )
    cursor += e_width

    stats = draw_y(
        ax,
        y_data,
        x_center=y_center,
        y_entry=1.59,
        y_bottom=baseline,
        y_cap=cap,
        thickness_ang=thickness_ang,
        energy_keV=energy_keV,
        stem_width=stem_width,
        color_gamma=color_gamma,
        beam_seed=beam_seed,
        beam_width_nm=beam_width_nm,
    )
    stats.update(
        draw_i(ax, line_data, x_center=i_center, baseline=baseline, cap=cap, width_scale=i_width)
    )
    stats.update(e_stats)
    ax.set_xlim(0.0, cursor + 0.40)
    ax.set_ylim(baseline - 0.16, cap + 0.16)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".svg", ".png"):
        path = out.with_suffix(ext)
        fig.savefig(
            path,
            format=ext.lstrip("."),
            facecolor=BG,
            transparent=False,
            dpi=dpi,
            metadata={"Date": None} if ext == ".svg" else None,
        )
        if ext == ".svg":
            # Matplotlib emits trailing spaces inside multiline path attributes.
            # Newlines already separate the coordinates; keep generated diffs clean.
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
        print(f"Saved {path}")
    plt.close(fig)
    return stats


def write_brand_assets(y_data, args, thickness_ang):
    """Derive compact web, app, social and favicon assets from the same artwork."""
    out = args.out.parent
    wordmark = args.out.with_suffix(".png")
    with Image.open(wordmark) as source:
        image = ImageOps.contain(source.convert("RGB"), (1200, 400), Image.Resampling.LANCZOS)
    rounded_image(image, args.corner_radius_fraction).save(wordmark, optimize=True)
    # GitHub's social-preview safe area needs horizontal breathing room. Only
    # this derivative is scaled down: the main wordmark remains full size.
    social_card_image(image, args.social_logo_width_fraction).save(
        out / "pyrite-social.png", optimize=True
    )
    round_svg_canvas(args.out.with_suffix(".svg"), args.corner_radius_fraction)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=120)
    fig.patch.set_facecolor(BG)
    draw_y(
        ax,
        y_data,
        x_center=0,
        y_entry=1.59,
        y_bottom=0.81,
        y_cap=2.70,
        thickness_ang=thickness_ang,
        energy_keV=args.energy_kev,
        stem_width=args.y_stem_width,
        color_gamma=args.y_color_gamma,
        beam_seed=args.seed,
        beam_width_nm=args.y_beam_width_nm,
    )
    ax.set(xlim=(-1.0, 1.0), ylim=(0.64, 2.87), aspect="equal")
    ax.axis("off")
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.98)
    icon = out / "pyrite-icon.png"
    fig.savefig(icon, facecolor=BG, dpi=120)
    plt.close(fig)
    with Image.open(icon) as source:
        image = source.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    image = rounded_image(image, args.corner_radius_fraction)
    image.save(icon, optimize=True)
    image.save(out / "pyrite-favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
    image.resize((180, 180), Image.Resampling.LANCZOS).save(out / "apple-touch-icon.png")
    small = image.resize((32, 32), Image.Resampling.LANCZOS)
    from io import BytesIO

    buffer = BytesIO()
    small.save(buffer, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    (out / "head.html").write_text(
        '<meta name="application-name" content="PyRITE">\n'
        f'<link rel="icon" type="image/png" href="{uri}">\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ASSETS / "pyrite-wordmark",
        help="Output stem for SVG, PNG and JSON",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "pyrite-output/branding/cache")
    parser.add_argument("--force", action="store_true", help="Rerun simulations ignoring caches")
    parser.add_argument("--y-material", default="hopg")
    parser.add_argument("--y-thickness-um", type=float, default=7.0)
    parser.add_argument("--i-thickness-nm", type=float, default=14.0)
    parser.add_argument("--energy-kev", type=float, default=30.0)
    parser.add_argument("--y-electrons", type=int, default=2**12)
    parser.add_argument(
        "--y-max-steps",
        type=int,
        default=2_000,
        help="Per-electron capacity; incomplete histories raise, never render",
    )
    parser.add_argument("--i-electrons", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--font-family",
        default="DejaVu Sans",
        help="Installed font for P, R, T and E-width matching",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--e-tilt-deg",
        type=float,
        default=17.0,
        help="Shallow viewing elevation of the three hBN E sheets in degrees",
    )
    parser.add_argument(
        "--e-azimuth-deg",
        type=float,
        default=-5.0,
        help="In-plane azimuth rotation of the hBN lattice (degrees)",
    )
    parser.add_argument(
        "--corner-radius-fraction",
        type=float,
        default=0.07,
        help="Rounded-canvas corner radius as a fraction of the shortest side",
    )
    parser.add_argument(
        "--social-logo-width-fraction",
        type=float,
        default=0.86,
        help="Fraction of the 1200 px social card width occupied by the logo (default: 0.86)",
    )
    parser.add_argument(
        "--y-stem-width",
        type=float,
        default=0.075,
        help="Requested Y stem width in logo units (also sets the default physical beam width)",
    )
    parser.add_argument(
        "--y-beam-width-nm",
        type=float,
        default=None,
        help="Optional PHYSICAL full beam width (nm); if set, source width determines the drawn stem width",
    )
    parser.add_argument(
        "--y-color-gamma",
        type=float,
        default=1.85,
        help="Contrast expansion for lower-energy trajectories",
    )
    parser.add_argument(
        "--i-width", type=float, default=0.45, help="Horizontal display scale of the spectral I"
    )
    parser.add_argument(
        "--ri-gap",
        type=float,
        default=0.10,
        help="Extra R-to-I spacing in logo units (default: 0.10; 0 restores original spacing)",
    )
    args = parser.parse_args()
    if not np.isfinite(args.ri_gap) or args.ri_gap < 0:
        parser.error("--ri-gap must be finite and nonnegative")
    if args.y_thickness_um <= 0 or args.i_thickness_nm <= 0 or args.energy_kev <= 5:
        parser.error("Thicknesses must be positive and beam energy >5 keV")
    if args.y_electrons < 20 or args.i_electrons < 20:
        parser.error("Electron budgets should each be >=20")
    if args.y_stem_width <= 0 or args.i_width <= 0 or args.y_color_gamma <= 0:
        parser.error("Logo dimensions and color gamma must be positive")
    if args.y_beam_width_nm is not None and (
        not np.isfinite(args.y_beam_width_nm) or args.y_beam_width_nm <= 0
    ):
        parser.error("--y-beam-width-nm must be finite and positive")
    if args.y_max_steps < 1:
        parser.error("--y-max-steps must be positive")
    if not 0.5 <= args.social_logo_width_fraction <= 1.0:
        parser.error("--social-logo-width-fraction must be in [0.5, 1.0]")
    y_thick_ang = args.y_thickness_um * 1e4
    i_thick_ang = args.i_thickness_nm * 10.0
    y_config = {
        "material": args.y_material,
        "energy_keV": args.energy_kev,
        "thickness_ang": y_thick_ang,
        "electrons": args.y_electrons,
        "seed": args.seed,
        "max_steps": args.y_max_steps,
    }
    i_config = {
        "energy_keV": args.energy_kev,
        "thickness_ang": i_thick_ang,
        "electrons": args.i_electrons,
    }
    cache = args.cache_dir
    y_data = cached_arrays(
        cache / f"Y-{_cache_key('Y', y_config)}.npz",
        lambda: simulate_y(**y_config),
        force=args.force,
    )
    i_data = cached_arrays(
        cache / f"I-{_cache_key('I', i_config)}.npz",
        lambda: simulate_i(**i_config),
        force=args.force,
    )
    stats = render(
        y_data,
        i_data,
        thickness_ang=y_thick_ang,
        energy_keV=args.energy_kev,
        out=args.out,
        dpi=args.dpi,
        font_family=args.font_family,
        stem_width=args.y_stem_width,
        i_width=args.i_width,
        ri_gap=args.ri_gap,
        color_gamma=args.y_color_gamma,
        beam_seed=args.seed,
        beam_width_nm=args.y_beam_width_nm,
        e_tilt_deg=args.e_tilt_deg,
        e_azimuth_deg=args.e_azimuth_deg,
    )
    write_brand_assets(y_data, args, y_thick_ang)
    report = {
        "generator": "scripts/pyrite_logo.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "y": y_config,
        "i": {
            **i_config,
            "material": "hopg",
            "tilt_deg": 45.0,
            "observation_angle_deg": 90.0,
            "mosaic": False,
            "resolved_seed": int(i_data["resolved_seed"]),
        },
        "rendering": {
            "font_family": args.font_family,
            "dpi": args.dpi,
            "background": BG,
            "selection": "transmitted histories, stratified by signed lateral exit offset",
            "lateral_axis": "magnified for typography",
            "spectral_baseline": "15th percentile (750-1000 eV)",
            "spectrum": "incoherent PXR/CBS, source density, no detector response",
            "y_stem_width_requested": args.y_stem_width,
            "y_beam_width_nm_requested": args.y_beam_width_nm,
            "beam_profile": "uniform rectangular top-hat in x and y, zero divergence",
            "uniform_source_method": (
                "Exact per-electron translation of PyRITE transport histories, "
                "valid for a laterally invariant homogeneous infinite slab at normal incidence"
            ),
            "beam_seed": args.seed,
            "y_color_gamma": args.y_color_gamma,
            "i_width": args.i_width,
            "ri_gap": args.ri_gap,
            "i_caps": False,
            "e_crystal": "hbn (three matching CIF-derived basal planes, artistic AA registry)",
            "e_tilt_deg": args.e_tilt_deg,
            "e_azimuth_deg": args.e_azimuth_deg,
            "corner_radius_fraction": args.corner_radius_fraction,
            "social_logo_width_fraction": args.social_logo_width_fraction,
            "social_card_size_px": [1200, 630],
        },
        **stats,
    }
    from pyrite import __version__

    report["pyrite_version"] = __version__
    report["cache_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            cache / f"Y-{_cache_key('Y', y_config)}.npz",
            cache / f"I-{_cache_key('I', i_config)}.npz",
        )
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Peak %.2f eV; FWHM %.2f eV; selected %d tracks"  # noqa: UP031
        % (stats["line_peak_eV"], stats["line_fwhm_eV"], stats["trajectories_selected"])
    )


if __name__ == "__main__":
    main()
