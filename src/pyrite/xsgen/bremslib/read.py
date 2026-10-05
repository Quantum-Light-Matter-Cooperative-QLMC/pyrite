"""Read the precomputed BremsLib v2.0 library in place.

No runner (D7). BremsLib ships 86300 ASCII files holding scaled double- and
single-differential bremsstrahlung cross sections, and ``Brems.exe`` only
recomputes what those files already contain, so PyRITE reads the library and
never builds or ports its GPL-3 Fortran.

Three file shapes, per the upstream manual (``BremsLib_v2.0.pdf``):

``SDCS/SDCS_<Z>.txt``
    One file per element. A header line, then one line per incident energy
    ``T1``: the energy in MeV, followed by a triplet -- scaled SDCS in mb,
    its relative computational uncertainty, and the point-to-finite-nucleus
    ratio -- for each of the 13 grid values of ``k/T1``.
``SDCS/DDCS_int/DDCS_int_<Z>.txt``
    Same layout, holding the vendor's own angular integral of the scaled
    DDCS. Its middle entry is the integral's *relative deviation* from the
    SDCS rather than a computational uncertainty.
``DDCS/DDCS_<Z>_<T1>_<k>.txt``
    One file per (Z, T1, k) node: photon emission angle in degrees, scaled
    DDCS in mb/sr, its relative uncertainty, and -- only when the reduced
    wavelength to nuclear radius ratio is below 40 -- the point-to-finite
    ratio. Where that fourth column is present the second holds the
    finite-nucleus DDCS; where it is absent the two do not differ
    significantly.

Both cross sections carry the upstream ``k / Z**2`` scaling factor, which is
preserved here: it cancels in the shape function, and unscaling belongs to
whichever consumer wants an absolute cross section.

Validation: bremslib-library-reference
"""

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .._errors import SourceUnavailableError

#: Relative name of the per-element SDCS file, under the library root.
SDCS_DIR = "SDCS"

#: Relative name of the per-node DDCS directory, under the library root.
DDCS_DIR = "DDCS"

#: Relative name of the vendor's angular-integral directory.
DDCS_INT_DIR = "SDCS/DDCS_int"

#: Directory-name prefix of the library data folder inside an unpacked
#: deposit. The patch version is part of the name (``BremsLib_v2.0.8``) while
#: the manual calls the folder ``BremsLib_v2.0``, so it is matched by prefix.
LIBRARY_DIR_PREFIX = "BremsLib_v2.0"

#: Highest incident energy the library covers at every ``k/T1`` node. Above
#: it only ``k = 0`` exists and the SDCS files carry zeros for the missing
#: entries, so a table that reached higher would ship holes an interpolating
#: consumer could not distinguish from a vanishing cross section.
COMPLETE_T1_MAX_MEV = 30.0

#: Number of ``k/T1`` grid values in every SDCS and DDCS_int line.
RATIO_COUNT = 13

_NODE_NAME = re.compile(r"^DDCS_(?P<z>\d+)_(?P<t1>[0-9.]+E[+-]\d+)_(?P<k>[0-9.]+E[+-]\d+)\.txt$")

_LABEL_VALUE = re.compile(r"=\s*([0-9.E+-]+)\s*\]")

_DIGEST_CHUNK = 1 << 20


def library_root(root: Path) -> Path:
    """Return the library data directory inside a resolved BremsLib tree.

    Accepts either an unpacked deposit -- whose root holds
    ``BremsLib_v2.0.<patch>/`` beside the two code folders -- or a path
    pointed straight at the library directory, which is what a user who kept
    only the data ends up with.

    Raises
    ------
    SourceUnavailableError
        If no directory below ``root`` holds an ``SDCS/SDCS_<Z>.txt``.
    """
    candidates = [root, *sorted(root.glob(f"{LIBRARY_DIR_PREFIX}*"))]
    for candidate in candidates:
        if next(candidate.glob(f"{SDCS_DIR}/SDCS_*.txt"), None) is not None:
            return candidate
    raise SourceUnavailableError(
        f"{root} holds no BremsLib library data: no "
        f"{SDCS_DIR}/SDCS_<Z>.txt in it or in any {LIBRARY_DIR_PREFIX}* directory below it"
    )


def sdcs_path(library: Path, z: int) -> Path:
    """Return the SDCS file for element ``z``."""
    return library / SDCS_DIR / f"SDCS_{int(z)}.txt"


def ddcs_integral_path(library: Path, z: int) -> Path:
    """Return the vendor angular-integral file for element ``z``."""
    return library / DDCS_INT_DIR / f"DDCS_int_{int(z)}.txt"


def ddcs_filename(z: int, t1_MeV: float, k_MeV: float) -> str:
    """Return the DDCS file name for one node.

    Reproduces the upstream naming: the incident energy to two significant
    digits and the photon energy to four, both in MeV in scientific notation.
    The rounding is lossy -- at ``T1 = 30 MeV`` the top node's
    ``k = 29.997 MeV`` is named ``3.000E+01`` -- which is why
    :func:`node_index` maps a node to its ``k/T1`` column by nearest grid
    value rather than by the energy in its name.
    """
    return f"DDCS_{int(z)}_{t1_MeV:.1E}_{k_MeV:.3E}.txt"


@dataclass(frozen=True)
class RatioTable:
    """One SDCS-shaped file: values on the ``(T1, k/T1)`` grid.

    Parameters
    ----------
    t1_MeV
        Incident electron kinetic energies, ascending.
    k_over_t1
        The 13 nominal ``k/T1`` grid values from the column headers. Nominal
        because the topmost is spelled ``1.0`` there while the data behind it
        stand slightly below it -- 0.99, ``1 - 50 eV / T1``, or 0.9999,
        depending on ``T1`` -- exactly as the manual describes.
    value
        Scaled cross section, or angular integral, per ``(T1, k/T1)``.
    middle
        The second entry of each triplet: a relative computational
        uncertainty in an SDCS file, a relative deviation from the SDCS in a
        ``DDCS_int`` file.
    point_finite
        Ratio of the point-nucleus to the finite-nucleus result. Exactly 1
        where the point-nucleus approximation was used.
    """

    t1_MeV: np.ndarray
    k_over_t1: np.ndarray
    value: np.ndarray
    middle: np.ndarray
    point_finite: np.ndarray

    def row(self, t1_MeV: float) -> int:
        """Return the row index whose energy matches ``t1_MeV``.

        Raises
        ------
        KeyError
            If no row matches. The library writes its grid energies to full
            double precision (``1.00000000000000008E-05``) while file names
            carry two digits, so the comparison is relative rather than
            exact.
        """
        matches = np.flatnonzero(np.isclose(self.t1_MeV, t1_MeV, rtol=1e-9, atol=0.0))
        if matches.size != 1:
            raise KeyError(f"no unique T1 = {t1_MeV!r} MeV row in the BremsLib table")
        return int(matches[0])


def parse_ratio_table(data: str | bytes) -> RatioTable:
    """Parse one SDCS-shaped file: ``SDCS_<Z>.txt`` or ``DDCS_int_<Z>.txt``.

    Raises
    ------
    ValueError
        If the header does not carry 13 labelled ``k/T1`` columns, a row does
        not hold the matching 40 numbers, the energies are not ascending, or
        a value is not finite.
    """
    text = data.decode("ascii") if isinstance(data, bytes) else data
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("BremsLib table has no data rows")
    labels = lines[0].split()[1::3]
    ratios = []
    for label in labels:
        match = _LABEL_VALUE.search(label)
        if match is None:
            raise ValueError(f"BremsLib column header is not a labelled ratio: {label!r}")
        ratios.append(float(match.group(1)))
    if len(ratios) != RATIO_COUNT:
        raise ValueError(f"BremsLib table has {len(ratios)} k/T1 columns, expected {RATIO_COUNT}")

    width = 1 + 3 * RATIO_COUNT
    rows: list[list[float]] = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != width:
            raise ValueError(f"BremsLib data row has {len(fields)} columns, expected {width}")
        try:
            rows.append([float(field) for field in fields])
        except ValueError as exc:
            raise ValueError(f"BremsLib data row is not numeric: {line!r}") from exc
    values = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("BremsLib table contains non-finite values")
    t1 = values[:, 0]
    if np.any(np.diff(t1) <= 0.0) or t1[0] <= 0.0:
        raise ValueError("BremsLib T1 grid must be positive and ascending")
    return RatioTable(
        t1_MeV=t1,
        k_over_t1=np.asarray(ratios, dtype=np.float64),
        value=values[:, 1::3],
        middle=values[:, 2::3],
        point_finite=values[:, 3::3],
    )


@dataclass(frozen=True)
class DdcsPanel:
    """One ``(Z, T1, k)`` node: the DDCS over the photon emission angle.

    Parameters
    ----------
    theta_deg
        Emission angles, ascending from 0 to 180 degrees. The grid depends on
        ``T1`` -- 181, 221 or 441 points -- which is why a table over many
        nodes needs an index rather than one rectangular block.
    ddcs_mb_sr
        Scaled DDCS, finite-nucleus where the library distinguishes the two.
    rel_err
        Relative computational uncertainty. Signed upstream; the sign carries
        no information PyRITE uses, so it is preserved rather than
        interpreted.
    point_finite
        Point-to-finite-nucleus ratio, or all ones where the file omits the
        column because the two do not differ significantly.
    finite_nucleus
        Whether that fourth column was present.
    """

    theta_deg: np.ndarray
    ddcs_mb_sr: np.ndarray
    rel_err: np.ndarray
    point_finite: np.ndarray
    finite_nucleus: bool


def parse_ddcs(data: str | bytes) -> DdcsPanel:
    """Parse one ``DDCS_<Z>_<T1>_<k>.txt`` node file.

    Raises
    ------
    ValueError
        If a row does not hold three or four numbers, the angular grid is not
        ascending within 0 to 180 degrees, or a cross section is negative.
    """
    text = data.decode("ascii") if isinstance(data, bytes) else data
    lines = [line for line in text.splitlines() if line.strip()]
    rows: list[list[float]] = []
    width = 0
    for line in lines[1:]:
        fields = line.split()
        if len(fields) not in (3, 4):
            raise ValueError(f"BremsLib DDCS row has {len(fields)} columns, expected 3 or 4")
        if width and len(fields) != width:
            raise ValueError("BremsLib DDCS file mixes three- and four-column rows")
        width = len(fields)
        try:
            rows.append([float(field) for field in fields])
        except ValueError as exc:
            raise ValueError(f"BremsLib DDCS row is not numeric: {line!r}") from exc
    if len(rows) < 3:
        raise ValueError("BremsLib DDCS file holds fewer than three angular rows")
    values = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("BremsLib DDCS file contains non-finite values")
    theta = values[:, 0]
    if np.any(np.diff(theta) <= 0.0) or theta[0] != 0.0 or theta[-1] != 180.0:
        raise ValueError("BremsLib DDCS theta grid must ascend from 0 to 180 degrees")
    ddcs = values[:, 1]
    if np.any(ddcs < 0.0):
        raise ValueError("BremsLib DDCS values must be non-negative")
    finite = width == 4
    return DdcsPanel(
        theta_deg=theta,
        ddcs_mb_sr=ddcs,
        rel_err=values[:, 2],
        point_finite=values[:, 3] if finite else np.ones_like(ddcs),
        finite_nucleus=finite,
    )


@dataclass(frozen=True)
class NodeFile:
    """One DDCS node file on disk, with the energies its name encodes."""

    path: Path
    t1_MeV: float
    k_MeV: float


def iter_node_files(library: Path, z: int) -> list[NodeFile]:
    """List the DDCS node files for element ``z``, ascending in ``(T1, k)``.

    The directory is the authority on which nodes exist: which ``k/T1``
    values are populated depends on ``T1``, and above
    :data:`COMPLETE_T1_MAX_MEV` only ``k = 0`` is present.

    Raises
    ------
    SourceUnavailableError
        If the library holds no node file for ``z``.
    """
    directory = library / DDCS_DIR
    prefix = f"DDCS_{int(z)}_"
    nodes: list[NodeFile] = []
    for path in directory.glob(f"{prefix}*.txt"):
        match = _NODE_NAME.match(path.name)
        if match is None or int(match.group("z")) != int(z):
            continue
        nodes.append(
            NodeFile(path=path, t1_MeV=float(match.group("t1")), k_MeV=float(match.group("k")))
        )
    if not nodes:
        raise SourceUnavailableError(
            f"BremsLib library at {library} holds no DDCS files for Z={int(z)} "
            f"(looked for {prefix}*.txt in {DDCS_DIR}/)"
        )
    return sorted(nodes, key=lambda node: (node.t1_MeV, node.k_MeV))


def node_index(k_over_t1: Sequence[float] | np.ndarray, ratio: float) -> int:
    """Return the ``k/T1`` column a node's energy ratio belongs to.

    Nearest grid value, not equality: the topmost node stands slightly below
    its nominal ratio of 1 by an amount that depends on ``T1``, and
    :func:`ddcs_filename`'s four-significant-digit rounding perturbs the
    ratio recovered from a file name further.

    Raises
    ------
    ValueError
        If the nearest grid value is further than half the smallest gap in
        the grid, which means the ratio is not a grid node at all.
    """
    grid = np.asarray(k_over_t1, dtype=np.float64)
    distance = np.abs(grid - float(ratio))
    index = int(np.argmin(distance))
    tolerance = 0.5 * float(np.min(np.diff(grid)))
    if distance[index] > tolerance:
        raise ValueError(
            f"k/T1 = {ratio!r} is not a BremsLib grid node; "
            f"nearest is {grid[index]!r} at a distance of {distance[index]!r}"
        )
    return index


def file_digest(paths: Iterable[Path]) -> str:
    """Return a SHA-256 over the contents of ``paths``, name-tagged.

    Each file contributes its name and its bytes, so an edit to any of them
    moves the digest while the iteration order does not.
    """
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(_DIGEST_CHUNK):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def library_version(library: Path, z: int, nodes: Sequence[NodeFile]) -> str:
    """Return the code version recorded for a table read from this library.

    BremsLib has no buildable source, so a table's ``code_version`` is a
    digest of the library it was read from rather than of Fortran sources.
    It covers the library directory name -- which carries the deposit's patch
    version -- the whole of the element's SDCS and ``DDCS_int`` files, and the
    name and byte size of every node file read.

    The node files' *contents* are deliberately not hashed: that is a 20 MB
    read per element, paid again on every cache hit, and an edit to a node
    file that matters changes its angular integral and so changes the
    ``DDCS_int`` file that is hashed. What this would miss is an edit that
    redistributed a node's DDCS over angle while preserving both its integral
    and its byte size. Upstream reissues whole deposits, so that is a
    theoretical gap rather than a practical one, and the stored table's
    ``arrays_sha256`` still records what was read.

    Raises
    ------
    SourceUnavailableError
        If the element's SDCS or ``DDCS_int`` file is missing.
    """
    digest = hashlib.sha256()
    digest.update(library.name.encode("utf-8"))
    digest.update(b"\0")
    for path in (sdcs_path(library, z), ddcs_integral_path(library, z)):
        try:
            digest.update(file_digest([path]).encode("ascii"))
        except OSError as exc:
            raise SourceUnavailableError(
                f"BremsLib library at {library} is missing {path.name}: {exc}"
            ) from exc
        digest.update(b"\0")
    for node in sorted(nodes, key=lambda item: item.path.name):
        digest.update(node.path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(node.path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "COMPLETE_T1_MAX_MEV",
    "DDCS_DIR",
    "DDCS_INT_DIR",
    "LIBRARY_DIR_PREFIX",
    "RATIO_COUNT",
    "SDCS_DIR",
    "DdcsPanel",
    "NodeFile",
    "RatioTable",
    "ddcs_filename",
    "ddcs_integral_path",
    "file_digest",
    "iter_node_files",
    "library_root",
    "library_version",
    "node_index",
    "parse_ddcs",
    "parse_ratio_table",
    "sdcs_path",
]
