"""Generate, store, and resolve tables produced by external Fortran codes.

ELSEPA (elastic DCS), SBETHE (collisional stopping and inelastic cross
sections), and BremsLib (bremsstrahlung SDCS and shape function) each own
their physics, but all three need the same plumbing: locate a code tree, build
it, run it in an isolated working directory, parse fixed-name outputs, key and
store the result, and resolve it at load. That plumbing lives here once.

Driver tier: this package imports ``materials``, and nothing in the physics
core imports it. Transport and spectrum read tables through
:func:`pyrite.xsgen.store.resolve` only, and never learn whether the bytes
came from the wheel or from the user's own generated table.

Design: ``agentdocs/specs/2026-09-21-external-fortran-code-integration.md``.
"""

from ._errors import (
    BuildError,
    DataFetchError,
    RunError,
    SourceUnavailableError,
    TableNotFoundError,
    ToolchainUnavailableError,
    XsgenError,
)
from .store import (
    ElementTarget,
    MaterialTarget,
    StoredTable,
    TableRequest,
    material_identity,
    require,
    resolve,
)

__all__ = [
    "BuildError",
    "DataFetchError",
    "ElementTarget",
    "MaterialTarget",
    "RunError",
    "SourceUnavailableError",
    "StoredTable",
    "TableNotFoundError",
    "TableRequest",
    "ToolchainUnavailableError",
    "XsgenError",
    "material_identity",
    "require",
    "resolve",
]
