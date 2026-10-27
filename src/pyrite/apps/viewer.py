"""Initial-material resolution for the marimo 3D visualization app
(``src/pyrite/apps/trace_app.py``).

Marimo apps in this environment can't be driven live (no browser/kernel access;
see the repo's marimo-live-driving notes), so the "which material does the
dropdown start on" logic lives here as a pure, unit-testable helper
(:func:`initial_material`) rather than inline in the notebook. The notebook just
calls it in a cell that runs before the dropdown cell.

The launcher that resolves and transports the material into the marimo
subprocess is :mod:`pyrite.cli.commands.app_viewer`; this module stays free of
CLI imports so the notebook can load it on its own. The persisted default the
launcher and the notebook share lives lower still, in
:mod:`pyrite._app_defaults`, so neither package has to import the other.
"""

from typing import Any

from .._app_defaults import get_viewer_default, set_viewer_default
from .._env import env_value


def get_default_material():
    """The persisted default material, or None if never set / empty."""
    return get_viewer_default()


def set_default_material(material):
    """Persist ``material`` as the default for future no-argument runs."""
    set_viewer_default(material)


def initial_material(cli_args, persisted_default):
    """Resolve the dropdown's initial value. Pure function -- called both from
    the notebook cell (with ``mo.cli_args()``) and directly from tests.

    Precedence: cli-arg material -> ``PYRITE_VIEWER_INITIAL`` env var (fallback
    transport, in case ``mo.cli_args()`` doesn't reach ``marimo edit``) ->
    ``persisted_default`` -> ``"hopg"``.
    """
    cli_material = cli_args.get("material")
    if cli_material:
        return str(cli_material)
    env_material = env_value("PYRITE_VIEWER_INITIAL")
    if env_material:
        return env_material
    if persisted_default:
        return persisted_default
    return "hopg"


def load_saved_capture(
    path: str,
    *,
    first: int = 50,
    max_segments: int = 100_000,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load bounded whole histories for the trace app without running transport."""
    from ..montecarlo.trajectory_selection import select_trajectories
    from ..plots.trajectory_capture import saved_trajectory_data

    selection = select_trajectories(path, first=first)
    case, data = saved_trajectory_data(selection, max_segments=max_segments)
    return selection, case, data


def saved_capture_panel(
    saved_capture: Any,
    error: str | None,
    *,
    mo: Any,
    theme: str,
    camera: dict,
) -> Any:
    """Render the trace app saved-source panel from already loaded rows."""
    from ..plots.altair.trajectories import penetration_survival_chart, trajectory_chart
    from ..plots.plotly.trajectories import trajectory_volume_figure_from_data
    from ._design import apply_altair_theme, apply_plotly_theme

    if saved_capture is None:
        return mo.md(error or "Enter an artifact path and open a saved capture.")
    _selection, _saved_case, _saved_data = saved_capture
    _saved_volume = trajectory_volume_figure_from_data(_saved_case, _saved_data)
    apply_plotly_theme(_saved_volume, theme)
    _saved_volume.update_scenes(camera=camera)
    return mo.vstack(
        [
            mo.md(
                f"**Saved capture:** {_selection.histories} histories, {_selection.segments:,} segments. "
                "Fractions describe this selection."
            ),
            mo.ui.plotly(_saved_volume),
            mo.hstack(
                [
                    mo.ui.altair_chart(
                        apply_altair_theme(
                            trajectory_chart(_saved_case, data=_saved_data, width=420),
                            theme,
                        )
                    ),
                    mo.ui.altair_chart(
                        apply_altair_theme(penetration_survival_chart(_selection, width=420), theme)
                    ),
                ],
                wrap=True,
            ),
        ]
    )


def render_trace_animation(
    mo,
    render_key,
    render_path,
    case,
    data,
    *,
    max_generation,
    min_energy_keV,
    realistic,
    beam_fwhm_mm,
    n_frames,
    camera,
    logarithmic_color,
):
    """Render an uncached trace movie and return the notebook's status pair."""
    from ..plots.plotly.render import prune_render_cache, render_reveal_animation
    from ..plots.plotly.trajectories import visible_trajectory_data

    if render_path.exists():
        return render_key, None
    try:
        with mo.status.progress_bar(
            total=n_frames,
            title="Rendering animation",
            subtitle=f"{n_frames} frames, ~3 s each offscreen",
        ) as bar:

            def progress_cb(_k, _n):
                bar.update()

            render_reveal_animation(
                case,
                visible_trajectory_data(
                    data, max_generation=max_generation, min_energy_keV=min_energy_keV
                ),
                render_path,
                realistic=realistic,
                beam_fwhm_mm=beam_fwhm_mm,
                n_frames=n_frames,
                fps=12,
                camera=camera,
                logarithmic_color=logarithmic_color,
                progress_cb=progress_cb,
            )
        prune_render_cache()
    except (ImportError, RuntimeError) as exc:
        return render_key, str(exc)
    return render_key, None
