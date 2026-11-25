"""Install hints for optional runtime extras (``pyrite-mc[plots]``, ``pyrite-mc[apps]``).

Availability is probed with :func:`importlib.util.find_spec`, which never imports
the dependency, so callers can check before paying an import or spawning marimo.
"""

from importlib.util import find_spec

_PLOTS_MODULES = (
    "matplotlib",
    "plotly",
    "datashader",
    "kaleido",
    "imageio",
    "imageio_ffmpeg",
)
EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "plots": _PLOTS_MODULES,
    "apps": ("marimo", "anywidget", *_PLOTS_MODULES),
}


#: Extras a development checkout gets from a dedicated pixi environment; every
#: other extra is already in the ``default`` environment.
_EXTRA_ENVIRONMENTS = {"nvidia": "nvidia", "amd": "amd", "intel": "intel"}


def checkout_environment(extra: str) -> str:
    """The pixi environment of a development checkout that provides ``extra``."""
    return _EXTRA_ENVIRONMENTS.get(extra, "default")


class MissingExtraError(ImportError):
    """An optional dependency of a ``pyrite-mc`` extra is not installed."""

    def __init__(self, extra: str, feature: str, missing: tuple[str, ...]):
        self.extra = extra
        self.feature = feature
        self.missing = missing
        super().__init__(
            f"{feature} requires the optional '{extra}' extra (missing: {', '.join(missing)}). "
            f'Install with `pip install "pyrite-mc[{extra}]"` or, in a checkout, '
            f"`pixi install -e {checkout_environment(extra)}`."
        )


def missing_modules(extra: str, modules: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Top-level modules of ``extra`` (or just ``modules``) that are not importable."""
    return tuple(name for name in modules or EXTRA_MODULES[extra] if find_spec(name) is None)


def require_extra(extra: str, *modules: str, feature: str) -> None:
    """Raise :class:`MissingExtraError` if ``modules`` (default: all of ``extra``) are absent."""
    missing = missing_modules(extra, modules)
    if missing:
        raise MissingExtraError(extra, feature, missing)
