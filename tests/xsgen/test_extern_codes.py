"""End-to-end anchors against the real external codes.

Everything else in this package drives a fake binary. These tests compile and
run the genuine article, which CI cannot do -- no Fortran compiler, and the
code trees are obtained by the user -- so they carry the ``extern_codes``
marker and are skipped unless ``PYRITE_EXTERN_CODES_TESTS=1``, following the
``online`` marker's convention. The ELSEPA anchor takes about 80 s.

The anchor is the vendor's own published test-run output, which is the
cheapest strong check available: it exercises source resolution, the source
digest, the build, scratch isolation, the symlinked database, and fixed-name
output collection in one pass, against numbers PyRITE did not produce.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from pyrite.xsgen._errors import SourceUnavailableError
from pyrite.xsgen._run import run_program
from pyrite.xsgen.sources import missing_data_dirs, resolve_source, source_digest
from pyrite.xsgen.toolchain import build, find_toolchain

pytestmark = pytest.mark.extern_codes

_OPT_IN = os.environ.get("PYRITE_EXTERN_CODES_TESTS") == "1"


def _elsepa():
    if not _OPT_IN:
        pytest.skip("set PYRITE_EXTERN_CODES_TESTS=1 to compile and run the external codes")
    if shutil.which("gfortran") is None and not os.environ.get("PYRITE_XSGEN_FC"):
        pytest.skip("no Fortran compiler")
    try:
        source = resolve_source("elsepa")
    except SourceUnavailableError as exc:
        pytest.skip(f"no ELSEPA tree: {exc}")
    if missing_data_dirs(source):
        pytest.skip("ELSEPA database/ is absent")
    return source


def _numeric_rows(text: str) -> np.ndarray:
    """Return the numeric table from an ELSEPA ``dcs_*.dat`` file."""
    rows = [
        [float(field) for field in line.split()]
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return np.asarray(rows, dtype=float)


def test_elsepa_reproduces_its_own_published_test_run():
    """Generator regression anchor for ELSEPA.

    Tolerance: the vendor ships six significant digits, and a different
    compiler's libm moves the last one. The observed disagreement is confined
    to the interference column, whose entries are ~1e-9 against a ~1e-12 to
    1e-15 DCS scale, at 1 part in 5.6e5. ``rtol=1e-5`` is one last-digit unit
    at six significant figures -- tight enough that a real change in the
    physics or a misparsed column fails, loose enough to survive a compiler
    swap. It is *not* fitted to the observed error: a byte-exact comparison
    would be an assertion about gfortran, not about ELSEPA.
    """
    source = _elsepa()
    binary = build(source, "elscata")
    deck = (source.root / "elscata.in").read_text(encoding="latin-1")

    result = run_program(
        binary,
        data_dirs=source.data_dirs,
        stdin_text=deck,
        outputs=("dcs_1p000e03.dat",),
        timeout=900,
    )

    reference_path = source.root / "test-run-output" / "dcs_1p000e03.dat"
    produced = _numeric_rows(result.text("dcs_1p000e03.dat"))
    reference = _numeric_rows(reference_path.read_text(encoding="latin-1"))

    assert produced.shape == reference.shape
    np.testing.assert_allclose(produced, reference, rtol=1.0e-5, atol=0.0)


def _free_atom_deck(z: int, energy: str = "1.00E2") -> str:
    """Return a minimal free-atom ELSEPA deck for element ``z``.

    Written out in full rather than patched from ``elscata.in``: rewriting the
    shipped deck by string surgery silently left ``IZ`` at its template value,
    so a run meant for silver computed gold and the test it fed compared two
    identical results. An explicit deck cannot fail that way. Absorption and
    polarization are off and only one energy is requested, which puts each run
    at about 0.1 s.
    """
    fields = [
        ("IZ", z),
        ("MNUCL", 3),
        ("NELEC", z),
        ("MELEC", 4),
        ("MUFFIN", 0),
        ("IELEC", -1),
        ("MEXCH", 1),
        ("MCPOL", 0),
        ("MABS", 0),
        ("IHEF", 2),
        ("EV", energy),
    ]
    return "".join(f"{name:<8}{value}\n" for name, value in fields)


def test_elsepa_runs_concurrently_without_crossing_outputs():
    """Two real runs at once, which is what the scratch directory is for.

    ELSEPA names its output by energy, not by target, so two runs at the same
    energy for different elements write the same file name. Sharing a
    directory would let one overwrite the other while both still exit 0 --
    a silent wrong answer.
    """
    from concurrent.futures import ThreadPoolExecutor

    source = _elsepa()
    binary = build(source, "elscata")

    def run(z: int):
        return run_program(
            binary,
            data_dirs=source.data_dirs,
            stdin_text=_free_atom_deck(z),
            outputs=("dcs_1p000e02.dat",),
            timeout=300,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        gold, silver = pool.map(run, [79, 47])

    assert gold.workdir != silver.workdir
    gold_rows = _numeric_rows(gold.text("dcs_1p000e02.dat"))
    silver_rows = _numeric_rows(silver.text("dcs_1p000e02.dat"))
    assert gold_rows.shape == silver_rows.shape

    # Column 2 is the DCS. Gold and silver must disagree across it; equality
    # would mean one run read the other's file. Column 0 (the angle grid) is
    # shared by construction, so comparing it would prove nothing.
    #
    # ``atol=0.0`` is load-bearing. The DCS is of order 1e-15 cm^2/sr, far
    # below ``np.allclose``'s default ``atol`` of 1e-8, so the default makes
    # *any* two cross sections compare equal and the assertion vacuous --
    # it passed against two runs of the same element before this was fixed.
    assert not np.allclose(gold_rows[:, 2], silver_rows[:, 2], rtol=1.0e-6, atol=0.0)
    np.testing.assert_array_equal(gold_rows[:, 0], silver_rows[:, 0])


def test_the_source_digest_identifies_the_real_tree():
    source = _elsepa()
    digest = source_digest(source)
    assert len(digest) == 64
    assert digest == source_digest(resolve_source("elsepa", source.root))


def test_the_real_compiler_reports_a_version_for_the_manifest():
    if not _OPT_IN:
        pytest.skip("set PYRITE_EXTERN_CODES_TESTS=1 to compile and run the external codes")
    if shutil.which("gfortran") is None and not os.environ.get("PYRITE_XSGEN_FC"):
        pytest.skip("no Fortran compiler")
    toolchain = find_toolchain()
    assert toolchain.version
    assert toolchain.compiler.is_file()
