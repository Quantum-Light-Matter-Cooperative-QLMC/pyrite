from pyrite.plots._pixel_frames import (
    detector_image_frame,
    histogram_frame,
    pixel_metadata_rows,
    pixel_spectrum_frame,
)
from pyrite.plots.altair.pixels import (
    detector_image_chart,
    histogram_chart,
    pixel_spectrum_chart,
)

from ..pixels import spectrum_components
from .common import themed_chart

_SEMANTICS_NOTES = {
    "clustered-photon-event-at-incident-ray-pixel": (
        "Counts are clustered photon events attributed to the incident-ray pixel; raw "
        "neighbouring-pixel triggers are not modelled. The response is uniform and uncalibrated."
    ),
    "ideal-energy-preserving-photon-event": (
        "The ideal photon counter is a unit-efficiency, energy-preserving reference response, "
        "not calibrated hardware."
    ),
}


def _filter_summary(filters) -> str:
    if not filters:
        return "none"
    return "; ".join(
        f"{index + 1}: {plate['thickness_mm']:g} mm" for index, plate in enumerate(filters)
    )


def _status_rows(observation, scored) -> list[dict[str, str]]:
    acquisition = scored.acquisition
    edges = acquisition.measured_edges_eV
    widths = {round(float(high - low), 9) for low, high in zip(edges[:-1], edges[1:], strict=True)}
    response = observation.identity.payload["response"]
    true_spatial = observation.identity.payload["true_spatial"]
    if acquisition.mode == "poisson":
        counts = f"Poisson realization, seed {acquisition.seed} (not an expectation)"
    else:
        counts = "deterministic expected counts"
    ny, nx = observation.spatial.shape
    return [
        {"quantity": "count semantics", "value": response["event_semantics"]},
        {"quantity": "response", "value": response["type"].rsplit(".", 1)[-1]},
        {"quantity": "counts", "value": counts},
        {"quantity": "exposure [s]", "value": f"{acquisition.exposure_s:g}"},
        {
            "quantity": "beam",
            "value": f"{scored.rep_rate_hz:g} Hz x {scored.bunch_charge_pc:g} pC",
        },
        {"quantity": "hit threshold [eV]", "value": f"{acquisition.hit_threshold_eV:g}"},
        {
            "quantity": "reporting bins",
            "value": (
                f"{len(edges) - 1} half-open bins over [{edges[0]:g}, {edges[-1]:g}) eV, "
                + (f"width {next(iter(widths)):g} eV" if len(widths) == 1 else "nonuniform widths")
            ),
        },
        {"quantity": "pixels", "value": f"{ny} x {nx}"},
        {"quantity": "filters", "value": _filter_summary(true_spatial["filters"])},
        {
            "quantity": "angular tiles",
            "value": " x ".join(map(str, true_spatial["scorer"].get("angular_shape", ())))
            + f" ({true_spatial['scorer'].get('reconstruction', 'nearest_tile').replace('_', '-')}"
            + " reconstruction)",
        },
        {"quantity": "observation digest", "value": observation.digest},
    ]


def render_pixel_detector(
    mo,
    *,
    state,
    selector,
    image_controls,
    selection_controls,
    resolved,
    theme,
):
    """Detector images, pixel selection, and one pixel's spectra/histogram.

    ``resolved`` is ``(scored_observation, image, warnings)`` from
    :func:`..pixels.resolve_pixel_image`, computed in its own cell so a
    pixel-selection change does not rescore the whole image.
    """
    header = [selector] if selector is not None else []
    if state.inventory is not None and state.inventory.problems:
        header.append(
            mo.callout(
                mo.md("**Unreadable observation files:** " + "; ".join(state.inventory.problems)),
                kind="warn",
            )
        )
    if state.observation is None:
        return mo.vstack([*header, mo.callout(mo.md(state.message or ""), kind="info")])

    observation = state.observation
    scored, image, warnings = resolved
    values = image_controls.value
    row = int(selection_controls.value["row"])
    column = int(selection_controls.value["column"])
    scale = values["scale"]

    image_chart = themed_chart(
        detector_image_chart(
            detector_image_frame(image),
            title=image.title,
            value_title=image.value_title,
            scale_type=scale if image.kind in {"total", "window"} else "linear",
            selected=(row, column),
        ),
        theme,
    )
    metadata = observation.spatial.pixel_metadata(pixels=[(row, column)])
    spectrum = themed_chart(
        pixel_spectrum_chart(
            pixel_spectrum_frame(
                observation, row, column, components=spectrum_components(observation)
            ),
            title=f"True accepted spectra, pixel ({row}, {column}), before response",
            y_type=scale,
        ),
        theme,
    )
    histogram, accounting = histogram_frame(scored.acquire(pixels=[(row, column)]))
    count_title = "expected counts" if scored.acquisition.mode == "expected" else "realized counts"
    histogram_view = themed_chart(
        histogram_chart(
            histogram,
            title=f"Measured histogram, pixel ({row}, {column})",
            count_title=count_title,
            y_type=scale,
        ),
        theme,
    )
    accounting_text = ", ".join(
        f"{name.replace('_', ' ')} {value:.4g}" for name, value in accounting.items()
    )
    parts = [
        *header,
        mo.hstack(
            [image_controls["kind"], image_controls["counts"], image_controls["scale"]],
            wrap=True,
        ),
        mo.hstack([image_controls["low"], image_controls["high"]], wrap=True),
        *(mo.callout(mo.md(warning), kind="warn") for warning in warnings),
        mo.hstack(
            [
                image_chart,
                mo.vstack(
                    [
                        mo.hstack(
                            [selection_controls["row"], selection_controls["column"]],
                            wrap=True,
                        ),
                        mo.ui.table(pixel_metadata_rows(metadata), selection=None),
                    ]
                ),
            ],
            wrap=True,
            align="start",
        ),
        spectrum,
        histogram_view,
        mo.md(
            f"Outside the registered histogram ({count_title}): {accounting_text}. "
            + _SEMANTICS_NOTES.get(observation.identity.payload["response"]["event_semantics"], "")
        ),
        mo.accordion(
            {
                "Acquisition and identity": mo.ui.table(
                    _status_rows(observation, scored), selection=None
                )
            }
        ),
    ]
    return mo.vstack(parts)


__all__ = ["render_pixel_detector"]
