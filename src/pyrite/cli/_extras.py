"""CLI-facing install hints for the optional ``apps``/``plots``/``tui`` extras."""

import click

from .._extras import MissingExtraError, require_extra


def require_cli_extra(extra: str, *modules: str, feature: str) -> None:
    """Exit with a usage-style error naming the extra when its dependencies are absent."""
    try:
        require_extra(extra, *modules, feature=feature)
    except MissingExtraError as exc:
        raise click.ClickException(str(exc)) from exc
