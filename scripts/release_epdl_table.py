"""Derive PyRITE's compact EPDL2025 photon cross-section table.

Maintainer-only. Reads the SHA-256-pinned upstream EPICS2025 photo-atomic tape
(``EPDL2025.ALL``, about 86 MB) and writes the small table that
:mod:`pyrite.materials.photon_cross_sections` reads at runtime. Publish it as a
PyRITE release asset and pin its URL in ``pyrite.datasets.EPDL``; users
install it with ``pyrite tables fetch epdl``.

For every element Z=1--100 the table keeps the five MF=23 integrated cross
sections whose sum is the narrow-beam total:

* MT=522 photoionization (all subshells),
* MT=502 coherent scattering,
* MT=504 incoherent scattering,
* MT=517 pair production in the nuclear field,
* MT=515 pair production in the electron field (triplet).

MT=501 (total) and MT=516 (pair total) are sums of these and are not stored.
The upstream sections are pre-linearized (ENDF interpolation law 2, lin-lin)
on dense grids. The only modification is knot thinning: within each run of
strictly increasing energies -- photoionization edges are repeated energies,
and a zero-valued threshold point starts a pair-production section -- knots
are dropped greedily while lin-lin interpolation through the kept knots, the
upstream law, still reproduces every dropped upstream node to
``--tolerance`` (relative). Zero
values and edge pairs are kept verbatim. Energies stay float64; cross sections
are stored as float32 (relative rounding ~6e-8, far below the tolerance).

The output is a deterministic ``.npz``: fixed member order and zip timestamps,
so a rerun from the same tape and tolerance reproduces the same bytes. Pin the
printed SHA-256 in ``photon_cross_sections.EPDL_TABLE_SHA256`` and
``pyrite.datasets.EPDL``.

Usage::

    UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python scripts/release_epdl_table.py \\
        --source /path/to/EPDL2025.ALL --write
"""

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

EPDL_URL = "https://nuclear.llnl.gov/EPICS/ENDF2025/EPDL2025.ALL"
EPDL_SHA256 = "59bbd8c559685dda0bf0de2762bc43126f599cd154d635940f17b6a59c1c43fd"
CHANNEL_MTS = (522, 502, 504, 517, 515)
DEFAULT_TOLERANCE = 5.0e-4
Z_MAX = 100
OUTPUT = Path(__file__).resolve().parents[1] / "build" / "xsgen-release" / "epdl2025_mf23.npz"


def _chord_ok(e: np.ndarray, s: np.ndarray, i: int, j: int, tolerance: float) -> bool:
    """Whether the lin-lin chord ``i -> j`` reproduces every node between them."""
    if j == i + 1:
        return True
    t = (e[i + 1 : j] - e[i]) / (e[j] - e[i])
    predicted = s[i] + t * (s[j] - s[i])
    return bool(np.all(np.abs(predicted - s[i + 1 : j]) <= tolerance * s[i + 1 : j]))


def thin_run(energy: np.ndarray, sigma: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices of a greedy knot subset for one strictly increasing positive run.

    Both endpoints are always kept. From each kept knot the next one is the
    farthest node whose lin-lin chord stays within ``tolerance`` (relative) of
    every skipped node (galloping, then bisection).
    """
    n = energy.size
    if n <= 2:
        return np.arange(n)
    keep = [0]
    i = 0
    while i < n - 1:
        good, step = i + 1, 1
        bad = n
        while True:
            j = min(i + step, n - 1)
            if _chord_ok(energy, sigma, i, j, tolerance):
                good = j
                if j == n - 1:
                    break
                step *= 2
            else:
                bad = j
                break
        while bad - good > 1:
            mid = (good + bad) // 2
            if _chord_ok(energy, sigma, i, mid, tolerance):
                good = mid
            else:
                bad = mid
        keep.append(good)
        i = good
    return np.asarray(keep)


def thin_section(
    energy: np.ndarray, sigma: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Thin one lin-lin MF=23 section, keeping edges and zero values verbatim."""
    energy = np.asarray(energy, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if energy.ndim != 1 or energy.shape != sigma.shape or energy.size < 2:
        raise ValueError("section must be paired one-dimensional arrays")
    if np.any(np.diff(energy) < 0.0) or np.any(sigma < 0.0):
        raise ValueError("section energies must be non-decreasing and values non-negative")
    # A run breaks at a repeated energy (edge) and around every zero value.
    breaks = np.zeros(energy.size, dtype=bool)
    breaks[1:] = np.diff(energy) == 0.0
    zero = sigma == 0.0
    breaks |= zero
    breaks[1:] |= zero[:-1]
    starts = np.flatnonzero(breaks | (np.arange(energy.size) == 0))
    keep: list[int] = []
    for start, stop in zip(starts, [*starts[1:], energy.size], strict=True):
        if zero[start] or stop - start <= 2:
            keep.extend(range(start, stop))
            continue
        keep.extend(start + thin_run(energy[start:stop], sigma[start:stop], tolerance))
    idx = np.asarray(keep)
    return energy[idx], sigma[idx]


def build_table(source: Path, tolerance: float) -> dict[str, np.ndarray]:
    """Read the upstream tape and return the arrays of the packaged table."""
    from endf_parserpy import EndfFile

    z_col, mt_col, offsets = [], [], [0]
    energies, sigmas = [], []
    seen = set()
    with EndfFile(source, on_error="raise") as tape:
        for position in range(len(tape)):
            material = tape[position]
            z = round(float(material.za) / 1000.0)
            if z in seen or not 1 <= z <= Z_MAX:
                raise ValueError(f"unexpected or duplicate material Z={z}")
            seen.add(z)
            for mt in CHANNEL_MTS:
                section = material[23, mt]
                if np.any(np.rint(np.asarray(section["INT"], dtype=float)) != 2):
                    raise ValueError(f"Z={z} MT={mt}: expected lin-lin (law 2) only")
                energy, sigma = thin_section(
                    np.asarray(section["Eint"], dtype=float),
                    np.asarray(section["sigma"], dtype=float),
                    tolerance,
                )
                z_col.append(z)
                mt_col.append(mt)
                energies.append(energy)
                sigmas.append(sigma)
                offsets.append(offsets[-1] + energy.size)
    if seen != set(range(1, Z_MAX + 1)):
        raise ValueError(f"tape does not cover Z=1..{Z_MAX}")
    return {
        "z": np.asarray(z_col, dtype=np.int16),
        "mt": np.asarray(mt_col, dtype=np.int16),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "energy_eV": np.concatenate(energies).astype(np.float64),
        "sigma_barn": np.concatenate(sigmas).astype(np.float32),
        "tolerance": np.asarray(tolerance, dtype=np.float64),
    }


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Write ``arrays`` as a byte-reproducible compressed ``.npz``; return its SHA-256."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arrays[name]))
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, member.getvalue(), compresslevel=9)
    data = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="local EPDL2025.ALL (else download)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--write", action="store_true", help="write the table")
    args = parser.parse_args(argv)

    source = args.source
    if source is None:
        source = Path("EPDL2025.ALL")
        urllib.request.urlretrieve(EPDL_URL, source)  # noqa: S310 - pinned below
    actual = _sha256(source)
    if actual != EPDL_SHA256:
        raise SystemExit(f"{source}: SHA-256 {actual} != pinned {EPDL_SHA256}")

    arrays = build_table(source, args.tolerance)
    print(f"tables: {arrays['z'].size}, knots: {arrays['energy_eV'].size}")
    if args.write:
        sha = write_deterministic_npz(args.output, arrays)
        print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
        print(f"sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
