"""Persisted "which material does the app start on" state for the marimo apps.

Both sides of each app read this state, so it lives below both of them: the
launchers (:mod:`pyrite.cli.commands.app_analysis`,
:mod:`pyrite.cli.commands.app_viewer`) write it from ``-d/--save-default`` and
read it back to resolve a no-argument launch, while the notebooks
(``analysis_app.py``, ``trace_app.py``, via :mod:`pyrite.apps.analyze` and
:mod:`pyrite.apps.viewer`) read it to seed their dropdown.

Keeping it here rather than in ``apps/`` is what lets ``cli`` stop importing
``apps``; that was the last edge holding ``apps`` inside the driver import
cycle. Each app owns one one-line file under :func:`pyrite.paths.state_dir`.

The paths are module globals rather than constants closed over by the
accessors so tests can retarget one app's state file with
``monkeypatch.setattr(_app_defaults, "ANALYSIS_DEFAULT_FILE", tmp_path / ...)``.
"""

from .paths import atomic_write_text, state_dir

ANALYSIS_DEFAULT_FILE = state_dir() / "analysis-default"
VIEWER_DEFAULT_FILE = state_dir() / "viewer-default"


def _read(path):
    """The material recorded in ``path``, or None if absent / blank."""
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        return None
    return text or None


def get_analysis_default():
    """The persisted analysis-app material, or None if never set / empty."""
    return _read(ANALYSIS_DEFAULT_FILE)


def set_analysis_default(material):
    """Persist ``material`` as the analysis app's default."""
    atomic_write_text(ANALYSIS_DEFAULT_FILE, material)


def get_viewer_default():
    """The persisted viewer-app material, or None if never set / empty."""
    return _read(VIEWER_DEFAULT_FILE)


def set_viewer_default(material):
    """Persist ``material`` as the viewer app's default."""
    atomic_write_text(VIEWER_DEFAULT_FILE, material)
