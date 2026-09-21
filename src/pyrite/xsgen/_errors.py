"""Error taxonomy for the external-code table generator.

Four distinct failure modes, because the remedy differs for each and D9
forbids papering over any of them with a surrogate model: the code tree is
missing, the Fortran compiler is missing, the build failed, or the program ran
and failed. Every message must name what to install or configure -- a bare
``FileNotFoundError`` from deep inside :mod:`subprocess` does not.
"""

from __future__ import annotations


class XsgenError(RuntimeError):
    """Base for every external-code generation failure."""


class SourceUnavailableError(XsgenError):
    """No usable source tree for an external code.

    Raised when neither a vendored tree, a configured path, nor the
    conventional sibling checkout resolves. The message names every location
    tried, the config key that overrides them, and the upstream deposit.
    """


class ToolchainUnavailableError(XsgenError):
    """No Fortran compiler.

    gfortran is an optional *runtime* dependency (D9), not a Python one, so
    this is a normal condition on a fresh machine and the message carries the
    install hint rather than a traceback.
    """


class BuildError(XsgenError):
    """The Fortran compiler ran and failed.

    Carries the exact command so the user can reproduce the failure outside
    PyRITE.
    """


class RunError(XsgenError):
    """An external program ran and failed, or produced no usable output."""


class TableNotFoundError(XsgenError):
    """No stored table for a request key, in either resolution tier."""


__all__ = [
    "BuildError",
    "RunError",
    "SourceUnavailableError",
    "TableNotFoundError",
    "ToolchainUnavailableError",
    "XsgenError",
]
