"""Initial-material resolution for the marimo 3D visualization app
(``src/pyrite/apps/trace_app.py``).

Marimo apps in this environment can't be driven live (no browser/kernel access;
see the repo's marimo-live-driving notes), so the "which material does the
dropdown start on" logic lives here as a pure, unit-testable helper
(:func:`initial_material`) rather than inline in the notebook. The notebook just
calls it in a cell that runs before the dropdown cell.

The launcher that resolves and transports the material into the marimo
subprocess is :mod:`pyrite.cli.commands.app_viewer`; this module stays free of
CLI imports so the notebook can load it on its own.
"""

from .._env import env_value
from ..paths import atomic_write_text, state_dir

_DEFAULT_FILE = state_dir() / "viewer-default"


def get_default_material():
    """The persisted default material, or None if never set / empty."""
    try:
        text = _DEFAULT_FILE.read_text().strip()
    except FileNotFoundError:
        return None
    return text or None


def set_default_material(material):
    """Persist ``material`` as the default for future no-argument runs."""
    atomic_write_text(_DEFAULT_FILE, material)


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
