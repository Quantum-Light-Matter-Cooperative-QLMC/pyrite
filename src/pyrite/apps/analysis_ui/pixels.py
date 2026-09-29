"""Pixel-detector observation controls and state for the pixel app.

Discovery, loading, control construction, and image resolution live here so
the marimo cells only wire widgets. Nothing here renders; see
:mod:`.views.pixels`.
"""

from collections import OrderedDict
from dataclasses import dataclass

from pyrite.observations import (
    ObservationInventory,
    ObservationStoreError,
    StoredObservation,
    observation_inventory,
)
from pyrite.plots._pixel_frames import (
    IMAGE_KINDS,
    ObservationImage,
    counting_observation,
    observation_image,
)

#: Resolved images kept per process; a 512 by 512 image is about 2 MiB.
_IMAGE_CACHE_SIZE = 8
_IMAGE_CACHE: OrderedDict[tuple, tuple] = OrderedDict()

_KIND_LABELS = {
    "Total counts": "total",
    "Energy window": "window",
    "Filter transmission": "transmission",
    "Filter coverage": "coverage",
}


@dataclass(frozen=True)
class PixelObservationState:
    """The selected stored observation, or why none is shown."""

    inventory: ObservationInventory | None
    observation: StoredObservation | None
    message: str | None = None


def discover_observations(stem: str | None) -> ObservationInventory | None:
    """Inventory of the loaded checkpoint's stored observations."""
    return None if stem is None else observation_inventory(stem)


def make_observation_selector(mo, inventory: ObservationInventory | None):
    """Dropdown over stored observations, or ``None`` when there are none."""
    if inventory is None or not inventory.entries:
        return None
    options = {entry.label: entry.observation_digest for entry in inventory.entries}
    return mo.ui.dropdown(options, value=next(iter(options)), label="Observation")


def load_selected_observation(
    inventory: ObservationInventory | None, selector
) -> PixelObservationState:
    """Reopen the selected observation without transport, or explain why not."""
    if inventory is None:
        return PixelObservationState(None, None, "No checkpoint is loaded.")
    if selector is None or selector.value is None:
        detail = (
            "Configure a counting physical detector with `pyrite profile physical-detector "
            "set` and run the profile with `pyrite run` to store one."
        )
        return PixelObservationState(
            inventory, None, f"No stored pixel observation for `{inventory.stem}`. {detail}"
        )
    try:
        return PixelObservationState(inventory, inventory.load(selector.value))
    except (ObservationStoreError, ValueError, TypeError) as exc:
        return PixelObservationState(inventory, None, f"Observation could not be opened: {exc}")


def make_pixel_image_controls(mo, observation: StoredObservation):
    """Image kind, count form, reporting window, and colour scale."""
    edges = [float(edge) for edge in observation.acquisition.measured_edges_eV]
    counts = {"Expected": "expected"}
    if observation.acquisition.mode == "poisson":
        counts = {"Realized (Poisson)": "realized", **counts}
    lows = {f"{edge:g} eV": edge for edge in edges[:-1]}
    highs = {f"{edge:g} eV": edge for edge in edges[1:]}
    return mo.ui.dictionary(
        {
            "kind": mo.ui.radio(_KIND_LABELS, value="Total counts", label="Image", inline=True),
            "counts": mo.ui.radio(counts, value=next(iter(counts)), label="Counts", inline=True),
            "low": mo.ui.dropdown(lows, value=next(iter(lows)), label="window from"),
            "high": mo.ui.dropdown(highs, value=list(highs)[-1], label="window to"),
            "scale": mo.ui.radio(
                {"Linear": "linear", "Log": "log"}, value="Linear", label="Scale", inline=True
            ),
        }
    )


def make_pixel_selection_controls(mo, observation: StoredObservation):
    """Row/column inputs, defaulting to the centre pixel.

    A fixed default keeps static HTML export deterministic: export renders the
    centre pixel, and interactive sessions change it here.
    """
    ny, nx = observation.spatial.shape
    return mo.ui.dictionary(
        {
            "row": mo.ui.number(start=0, stop=ny - 1, step=1, value=ny // 2, label="row"),
            "column": mo.ui.number(start=0, stop=nx - 1, step=1, value=nx // 2, label="column"),
        }
    )


def resolve_pixel_image(
    observation: StoredObservation, values
) -> tuple[StoredObservation, ObservationImage, tuple[str, ...]]:
    """Scored observation, its requested image, and user-facing warnings.

    An inverted reporting window falls back to the full reporting range with
    a warning instead of failing. Results are memoized on the observation
    digest and image settings, so the app can resolve lazily inside its tab
    and a pixel-selection change never rescores the image.
    """
    key = (
        observation.digest,
        values["kind"],
        values["counts"],
        float(values["low"]),
        float(values["high"]),
    )
    cached = _IMAGE_CACHE.get(key)
    if cached is not None:
        _IMAGE_CACHE.move_to_end(key)
        return cached
    resolved = _resolve_pixel_image(observation, values)
    _IMAGE_CACHE[key] = resolved
    while len(_IMAGE_CACHE) > _IMAGE_CACHE_SIZE:
        _IMAGE_CACHE.popitem(last=False)
    return resolved


def _resolve_pixel_image(
    observation: StoredObservation, values
) -> tuple[StoredObservation, ObservationImage, tuple[str, ...]]:
    warnings: list[str] = []
    scored = counting_observation(observation, values["counts"])
    kind = values["kind"]
    low, high = float(values["low"]), float(values["high"])
    window = (low, high)
    if low >= high:
        edges = observation.acquisition.measured_edges_eV
        window = (float(edges[0]), float(edges[-1]))
        warnings.append(
            f"Window start {low:g} eV is not below its end {high:g} eV; "
            "showing the full reporting range."
        )
    if kind not in IMAGE_KINDS:
        raise ValueError(f"unknown image kind {kind!r}")
    image = observation_image(scored, kind, energy_range_eV=window)
    if kind in {"total", "window"} and not image.values.any():
        warnings.append("Every pixel registers zero counts for this selection.")
    return scored, image, tuple(warnings)


def spectrum_components(observation: StoredObservation) -> tuple[str, ...]:
    """True-spectrum components to show for the observation's emission."""
    line = "coherent" if observation.emission == "coherent" else "line"
    components = [line]
    if observation.spatial.characteristic_line is not None:
        components.append("characteristic")
    components.append("background")
    return tuple(components)


__all__ = [
    "PixelObservationState",
    "discover_observations",
    "load_selected_observation",
    "make_observation_selector",
    "make_pixel_image_controls",
    "make_pixel_selection_controls",
    "resolve_pixel_image",
    "spectrum_components",
]
