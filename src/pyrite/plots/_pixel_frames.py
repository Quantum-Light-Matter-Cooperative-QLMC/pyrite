"""Renderer-neutral tidy frames for pixel-detector observations.

Detector images are block-reduced to a bounded display grid, so a 512 by 512
chip draws at most ``max_cells`` squared marks while each cell still reports
the exact pixel range it covers. Selected-pixel spectra and histograms are
built from one pixel's factors only; nothing here materializes a
pixel-by-energy cube. No plotting-library imports.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..instrument import AcquisitionBatch
    from ..observations import StoredObservation
    from ..results.model import PixelMetadata

ImageKind = Literal["total", "window", "transmission", "coverage"]
CountMode = Literal["expected", "realized"]

#: Image kinds, their display titles, and how display blocks reduce pixels.
IMAGE_KINDS: dict[str, tuple[str, str]] = {
    "total": ("Registered counts (all reporting bins)", "sum"),
    "window": ("Registered counts in energy window", "sum"),
    "transmission": ("Primary filter transmission", "mean"),
    "coverage": ("Filters crossed by the centre ray", "max"),
}


@dataclass(frozen=True)
class ObservationImage:
    """One full-resolution detector image and how to display it.

    Parameters
    ----------
    values
        ``(ny, nx)`` image.
    kind
        One of :data:`IMAGE_KINDS`.
    title, value_title
        Chart title and colour-legend title.
    reduce
        ``"sum"``, ``"mean"``, or ``"max"`` for display-block reduction.
    """

    values: np.ndarray
    kind: str
    title: str
    value_title: str
    reduce: str


def counting_observation(observation: StoredObservation, counts: CountMode) -> StoredObservation:
    """Return the observation scored as expected counts or its realization.

    ``"realized"`` is valid only for a Poisson acquisition. ``"expected"`` on a
    Poisson acquisition rescores the same true factors in expected mode, so an
    expectation is never replaced by a draw or vice versa.
    """
    acquisition = observation.acquisition
    if counts == "realized":
        if acquisition.mode != "poisson":
            raise ValueError("this observation has no Poisson realization")
        return observation
    if counts != "expected":
        raise ValueError("counts must be 'expected' or 'realized'")
    if acquisition.mode == "expected":
        return observation
    return observation.rescore(acquisition=replace(acquisition, mode="expected", seed=None))


def observation_image(
    observation: StoredObservation,
    kind: ImageKind,
    *,
    energy_range_eV: tuple[float, float] | None = None,
    pixel_chunk: int = 4096,
) -> ObservationImage:
    """Compute one detector image in bounded pixel chunks.

    ``energy_range_eV`` must match reporting edges for ``"window"``. The
    transmission image uses the continuum-grid node nearest its midpoint (or
    the reporting range's midpoint when it is ``None``) and names that node's
    energy in its title.
    """
    title, reduce = IMAGE_KINDS[kind]
    spatial = observation.spatial
    counts_title = (
        "expected counts" if observation.acquisition.mode == "expected" else "realized counts"
    )
    if kind == "total":
        values = observation.acquisition_image(pixel_chunk=pixel_chunk)
        return ObservationImage(values, kind, title, counts_title, reduce)
    if kind == "window":
        if energy_range_eV is None:
            raise ValueError("a window image needs energy_range_eV")
        values = observation.acquisition_image(energy_range_eV, pixel_chunk=pixel_chunk)
        low, high = energy_range_eV
        return ObservationImage(
            values, kind, f"{title} [{low:g}, {high:g}) eV", counts_title, reduce
        )
    if kind == "transmission":
        # The continuum grid spans the reporting range; the line grid may not.
        edges = observation.acquisition.measured_edges_eV
        low, high = (edges[0], edges[-1]) if energy_range_eV is None else energy_range_eV
        node = spatial.energy_node(0.5 * (float(low) + float(high)), component="background")
        values = spatial.transmission_image(node, component="background")
        return ObservationImage(values, kind, f"{title} at {node:g} eV", "T", reduce)
    if kind == "coverage":
        values = np.count_nonzero(spatial.filter_coverage(), axis=-1)
        return ObservationImage(values, kind, title, "filters", reduce)
    raise ValueError(f"kind must be one of {', '.join(IMAGE_KINDS)}")


def _block_reduce(values: np.ndarray, block: int, reduce: str) -> np.ndarray:
    ny, nx = values.shape
    starts_y = np.arange(0, ny, block)
    starts_x = np.arange(0, nx, block)
    if reduce == "sum":
        return np.add.reduceat(np.add.reduceat(values, starts_y, axis=0), starts_x, axis=1)
    if reduce == "max":
        return np.maximum.reduceat(np.maximum.reduceat(values, starts_y, axis=0), starts_x, axis=1)
    if reduce == "mean":
        sums = np.add.reduceat(np.add.reduceat(values, starts_y, axis=0), starts_x, axis=1)
        sizes_y = np.diff(np.append(starts_y, ny))
        sizes_x = np.diff(np.append(starts_x, nx))
        return sums / np.outer(sizes_y, sizes_x)
    raise ValueError("reduce must be 'sum', 'mean', or 'max'")


def detector_image_frame(image: ObservationImage, *, max_cells: int = 128) -> pd.DataFrame:
    """Block-reduce an image to at most ``max_cells`` cells per axis.

    Columns: ``row0``/``row1`` and ``column0``/``column1`` (inclusive pixel
    range of the cell), ``row``/``column`` (cell origin, for plotting),
    ``block`` (pixels per cell side), and ``value`` reduced by
    ``image.reduce``. A ``block`` of 1 is full resolution.
    """
    values = np.asarray(image.values, dtype=float)
    ny, nx = values.shape
    block = max(1, -(-max(ny, nx) // int(max_cells)))
    reduced = _block_reduce(values, block, image.reduce)
    rows = np.arange(0, ny, block)
    columns = np.arange(0, nx, block)
    row_grid, column_grid = np.meshgrid(rows, columns, indexing="ij")
    return pd.DataFrame(
        {
            "row": row_grid.ravel(),
            "column": column_grid.ravel(),
            "row0": row_grid.ravel(),
            "row1": np.minimum(row_grid.ravel() + block, ny) - 1,
            "column0": column_grid.ravel(),
            "column1": np.minimum(column_grid.ravel() + block, nx) - 1,
            "block": block,
            "value": reduced.ravel(),
        }
    )


def pixel_spectrum_frame(
    observation: StoredObservation, row: int, column: int, *, components: tuple[str, ...]
) -> pd.DataFrame:
    """True accepted spectra of one pixel, long form by component.

    ``value`` is photons per incident electron per eV, pixel solid angle and
    filter transmission included, before detector response.
    """
    frames = []
    for component in components:
        energy, spectra = observation.spatial.spectra(pixels=[(row, column)], component=component)
        frames.append(
            pd.DataFrame({"energy_eV": energy, "value": spectra[0], "component": component})
        )
    return pd.concat(frames, ignore_index=True)


def histogram_frame(batch: AcquisitionBatch) -> tuple[pd.DataFrame, dict[str, float]]:
    """One-pixel reporting-bin histogram and its out-of-histogram accounting.

    Returns the ``low_eV``/``high_eV``/``counts`` frame for the registered
    half-open bins and a mapping of underflow, overflow, and below-cut counts
    in the same count form (expected or realized).
    """
    if batch.coordinates.shape[0] != 1:
        raise ValueError("histogram_frame takes a one-pixel batch")
    realized = batch.realized is not None
    edges = batch.measured_edges_eV
    frame = pd.DataFrame(
        {"low_eV": edges[:-1], "high_eV": edges[1:], "counts": batch.counts[0].astype(float)}
    )
    suffix = "realized" if realized else "expected"
    accounting = {
        name: float(getattr(batch, f"{name}_{suffix}")[0])
        for name in ("underflow", "overflow", "below_cut")
    }
    return frame, accounting


def pixel_metadata_rows(metadata: PixelMetadata) -> list[dict[str, str]]:
    """Two-column ``quantity``/``value`` rows describing one selected pixel."""
    if metadata.coordinates.shape[0] != 1:
        raise ValueError("pixel_metadata_rows takes a one-pixel selection")

    def vector(values) -> str:
        return "(" + ", ".join(f"{value:.6g}" for value in values) + ")"

    row, column = metadata.coordinates[0]
    rows = [
        ("row, column", f"{row}, {column}"),
        ("local x, y [mm]", vector(metadata.local_position_mm[0])),
        ("lab centre [mm]", vector(metadata.center_mm[0])),
        ("direction (lab)", vector(metadata.direction_lab[0])),
        ("polar, azimuth [deg]", f"{metadata.polar_deg[0]:.6g}, {metadata.azimuth_deg[0]:.6g}"),
        ("distance [mm]", f"{metadata.distance_mm[0]:.6g}"),
        ("solid angle [sr]", f"{metadata.solid_angle_sr[0]:.6g}"),
        ("angular tile", str(metadata.tile_index[0])),
    ]
    if metadata.tile_direction_lab is not None:
        rows.append(("tile direction (lab)", vector(metadata.tile_direction_lab[0])))
    rows.extend(
        (f"filter {index + 1} path [mm]", f"{path:.6g}")
        for index, path in enumerate(metadata.path_length_mm[0])
    )
    return [{"quantity": name, "value": value} for name, value in rows]


__all__ = [
    "IMAGE_KINDS",
    "ObservationImage",
    "counting_observation",
    "detector_image_frame",
    "histogram_frame",
    "observation_image",
    "pixel_metadata_rows",
    "pixel_spectrum_frame",
]
