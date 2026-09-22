"""End-to-end anchors against the real external codes.

Everything else in this package drives a fake binary. These tests compile and
run the genuine article, which CI cannot do because it has no Fortran compiler,
so they carry the ``extern_codes``
marker and are skipped unless ``PYRITE_EXTERN_CODES_TESTS=1``, following the
``online`` marker's convention. The ELSEPA anchor takes about 80 s.

BremsLib is the exception that still belongs here: nothing is compiled,
because PyRITE reads its precomputed library (D7), but that 810 MB library is
not redistributed either, so CI cannot see it any more than it can see a
Fortran compiler.

The anchor is the vendor's own published test-run output, which is the
cheapest strong check available: it exercises source resolution, the source
digest, the build, scratch isolation, the symlinked database, and fixed-name
output collection in one pass, against numbers PyRITE did not produce.
"""

import os
import shutil

import numpy as np
import pytest

from pyrite.xsgen._errors import SourceUnavailableError
from pyrite.xsgen._run import run_program
from pyrite.xsgen.bremslib import (
    angular_integral,
    build_table,
    ddcs_filename,
    iter_node_files,
    library_root,
    panel_of,
    parse_ddcs,
    parse_ratio_table,
    shape_function,
)
from pyrite.xsgen.bremslib import generate_element as generate_bremslib
from pyrite.xsgen.bremslib.read import ddcs_integral_path
from pyrite.xsgen.bremslib.release import build_release, load_release_index, release_arrays
from pyrite.xsgen.elsepa import generate_element
from pyrite.xsgen.sbethe import (
    SbetheDeck,
    generate_material,
    parse_integrated,
    parse_oscillator,
    parse_stopping,
)
from pyrite.xsgen.sources import (
    missing_data_dirs,
    resolve_source,
    source_digest,
    vendored_root,
)
from pyrite.xsgen.toolchain import build, find_toolchain

pytestmark = pytest.mark.extern_codes

_OPT_IN = os.environ.get("PYRITE_EXTERN_CODES_TESTS") == "1"


def _elsepa():
    if not _OPT_IN:
        pytest.skip("set PYRITE_EXTERN_CODES_TESTS=1 to compile and run the external codes")
    if shutil.which("gfortran") is None and not os.environ.get("PYRITE_XSGEN_FC"):
        pytest.skip("no Fortran compiler")
    try:
        source = resolve_source("elsepa", vendored_root("elsepa"))
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


def _sbethe():
    if not _OPT_IN:
        pytest.skip("set PYRITE_EXTERN_CODES_TESTS=1 to compile and run the external codes")
    if shutil.which("gfortran") is None and not os.environ.get("PYRITE_XSGEN_FC"):
        pytest.skip("no Fortran compiler")
    try:
        source = resolve_source("sbethe", vendored_root("sbethe"))
    except SourceUnavailableError as exc:
        pytest.skip(f"no SBETHE tree: {exc}")
    if missing_data_dirs(source):
        pytest.skip("SBETHE sdbase/ is not installed; run `pyrite tables fetch sbethe`")
    return source


def _run_sbethe_water():
    """Drive one real SBETHE run for liquid water and parse its outputs."""
    source = _sbethe()
    binary = build(source, "sbethe", toolchain=find_toolchain())
    deck = SbetheDeck(
        name="water",
        composition={1: 2, 8: 1},
        density_g_cm3=1.0,
        mean_excitation_eV=75.0,
    )
    run = run_program(
        binary,
        data_dirs=source.data_dirs,
        stdin_text=deck.render(),
        outputs=("stp.dat", "asymptotic.dat", "OOS.dat"),
    )
    return (
        parse_stopping(run.outputs["stp.dat"]),
        parse_integrated(run.outputs["asymptotic.dat"]),
        parse_oscillator(run.outputs["OOS.dat"]),
    )


def test_the_rendered_deck_drives_a_complete_real_sbethe_run():
    """The deck is the whole contract with a prompt-driven program.

    A wrong answer order does not fail loudly: the program reads the next
    answer for a different question and produces a table for a material nobody
    asked for. So the anchor checks that the material SBETHE echoed back is the
    one the deck described.
    """
    stopping, integrated, oscillator = _run_sbethe_water()

    # Echoed back by the program from the answers it consumed.
    assert stopping.electrons_per_molecule == pytest.approx(10.0)
    assert stopping.molecular_weight_g_mol == pytest.approx(18.0153, rel=1e-4)
    assert stopping.density_g_cm3 == pytest.approx(1.0)
    assert stopping.mean_excitation_eV == pytest.approx(75.0)
    assert oscillator.electrons_per_molecule == pytest.approx(10.0)

    # stp.dat starts at the corrected-Bethe floor; asymptotic.dat reaches below it.
    assert stopping.energy_ev[0] == pytest.approx(1.0e3)
    assert integrated.energy_ev[0] < stopping.energy_ev[0]


def test_the_real_collision_stopping_power_of_water_reproduces_estar():
    """Anchor the parsed column against published values PyRITE did not produce.

    ICRU 37 / NIST ESTAR collision stopping power for liquid water at I=75 eV:
    113.0 MeV cm^2/g at 1 keV and 2.398 MeV cm^2/g at 1 GeV. A 2% tolerance
    covers SBETHE's shell and density-effect corrections differing in detail
    from ESTAR's, which is a physics comparison #90 owns; what is anchored
    here is that the right column, in the right units, reached the caller.
    """
    stopping, _, _ = _run_sbethe_water()

    at_1kev = float(np.interp(1.0e3, stopping.energy_ev, stopping.stopping_mev_cm2_per_g))
    at_1gev = float(np.interp(1.0e9, stopping.energy_ev, stopping.stopping_mev_cm2_per_g))

    assert at_1kev == pytest.approx(113.0, rel=0.02)
    assert at_1gev == pytest.approx(2.398, rel=0.02)

    # The collision stopping power has a minimum near 1 MeV; it is not monotone.
    minimum_ev = float(stopping.energy_ev[np.argmin(stopping.stopping_mev_cm2_per_g)])
    assert 3.0e5 < minimum_ev < 3.0e6


def test_two_materials_do_not_share_a_scratch_directory():
    """Every run writes the same fixed output names into its own directory."""
    source = _sbethe()
    binary = build(source, "sbethe", toolchain=find_toolchain())
    results = []
    for name, composition, density, excitation in (
        ("water", {1: 2, 8: 1}, 1.0, 75.0),
        ("alumina", {13: 2, 8: 3}, 3.97, 145.2),
    ):
        deck = SbetheDeck(
            name=name,
            composition=composition,
            density_g_cm3=density,
            mean_excitation_eV=excitation,
        )
        run = run_program(
            binary,
            data_dirs=source.data_dirs,
            stdin_text=deck.render(),
            outputs=("stp.dat",),
        )
        results.append(parse_stopping(run.outputs["stp.dat"]))

    water, alumina = results
    assert water.molecular_weight_g_mol != alumina.molecular_weight_g_mol
    assert not np.allclose(
        water.stopping_mev_cm2_per_g,
        alumina.stopping_mev_cm2_per_g,
        rtol=1.0e-6,
        atol=0.0,
    )


def test_the_generate_path_runs_elsepa_from_its_own_rendered_deck(isolated_dirs, monkeypatch):
    """Exercise the deck the generator actually writes, not a hand-written one.

    The other anchors feed ``run_program`` a deck spelled in the test, so they
    never covered ``ElsepaDeck.render`` reaching ``elscata``. Two defects hid
    behind that gap: the energy field overflowed the ``A12`` value column, and
    the parser required an absorption line that MABS 0 does not write.
    """
    source = _elsepa()
    monkeypatch.setattr("pyrite.xsgen.store.data_dir", lambda: isolated_dirs["data"] / "packaged")

    result = generate_element(79, [1.0e3], source_path=source.root)

    arrays = result.table.arrays()
    assert result.generated is True
    assert arrays["dcs_cm2_sr"].shape[0] == 1
    assert float(arrays["energy_eV"][0]) == pytest.approx(1.0e3)
    assert float(arrays["total_elastic_cm2"][0]) > 0.0
    # MABS 0 is the deck default, so there is no absorption.
    assert float(arrays["absorption_cm2"][0]) == 0.0


def test_the_generate_path_runs_sbethe_from_its_own_rendered_deck(monkeypatch, tmp_path):
    """The same end-to-end cover for the material path.

    Only the table store is redirected here, not ``user_data_dir`` as a whole:
    SBETHE's ``sdbase/`` is *fetched* into that same directory, so isolating
    it would hide the real reference data and skip this anchor.
    """
    _sbethe()
    monkeypatch.setattr("pyrite.xsgen.store.user_table_dir", lambda: tmp_path / "user")
    monkeypatch.setattr("pyrite.xsgen.store.packaged_table_dir", lambda: tmp_path / "packaged")

    result = generate_material("water", {1: 2, 8: 1}, density_g_cm3=1.0, mean_excitation_eV=75.0)

    arrays = result.table.arrays()
    assert result.generated is True
    assert float(arrays["mean_excitation_eV"]) == pytest.approx(75.0)
    assert arrays["stopping_MeV_cm2_per_g"].shape == arrays["stopping_energy_eV"].shape


def _bremslib():
    """Return the real BremsLib library directory, or skip.

    Nothing is built here, so the compiler check the other two guards make
    does not apply: what this anchor needs is the library itself.
    """
    if not _OPT_IN:
        pytest.skip("set PYRITE_EXTERN_CODES_TESTS=1 to read the external code libraries")
    try:
        source = resolve_source("bremslib")
    except SourceUnavailableError as exc:
        pytest.skip(f"no BremsLib checkout: {exc}")
    try:
        return library_root(source.root)
    except SourceUnavailableError as exc:
        pytest.skip(f"no BremsLib library data: {exc}")


@pytest.mark.parametrize(
    ("z", "t1_MeV", "column"),
    [(1, 1.0e-03, 0), (13, 1.0e-01, 5), (79, 1.0e01, 9), (79, 3.0e01, 0), (100, 2.0e00, 12)],
)
def test_our_angular_integral_reproduces_the_published_one(z, t1_MeV, column):
    """Generator regression anchor for BremsLib.

    Upstream publishes the angular integral of every DDCS node in
    ``DDCS_int_<Z>.txt``, computed from the same tabulated grid with a
    10th-order Newton-Cotes formula. Reproducing it is the strongest cheap
    check available: it exercises source resolution, the library layout, both
    file parsers, the ragged angular grid, and the integration rule against
    numbers PyRITE did not produce.

    Tolerance: the observed disagreement across these nodes is at most 2e-5
    relative, and the rule difference -- Simpson against 10th-order
    Newton-Cotes -- is its whole content. ``rtol=1e-4`` leaves room for a
    node whose forward peak is less well resolved while still failing on a
    mis-scaled column or a misread grid, both of which move the integral by
    orders of magnitude.

    Validation: bremslib-library-reference
    """
    library = _bremslib()
    vendor = parse_ratio_table(ddcs_integral_path(library, z).read_text(encoding="ascii"))
    ratio = float(vendor.k_over_t1[column])
    # The top node stands slightly below its nominal unit ratio, and the file
    # name rounds the photon energy to four digits, so it is found as the
    # library's own highest node at this energy rather than by name. Sorting
    # by the parsed energy matters: node names do not sort by photon energy,
    # because a two-digit exponent puts ``8.000E-01`` after ``2.000E+00``.
    at_energy = [
        node
        for node in iter_node_files(library, z)
        if np.isclose(node.t1_MeV, t1_MeV, rtol=1e-6, atol=0.0)
    ]
    assert at_energy, f"no BremsLib DDCS nodes for Z={z} at T1={t1_MeV} MeV"
    if ratio == 1.0:
        node_path = max(at_energy, key=lambda node: node.k_MeV).path
    else:
        node_path = library / "DDCS" / ddcs_filename(z, t1_MeV, ratio * t1_MeV)
    panel = parse_ddcs(node_path.read_text(encoding="ascii"))

    ours = angular_integral(panel.theta_deg, panel.ddcs_mb_sr)

    published = float(vendor.value[vendor.row(t1_MeV), column])
    assert ours == pytest.approx(published, rel=1.0e-4)
    # The shape function is the DDCS over exactly that integral, so it
    # integrates to one under the same rule.
    shape = shape_function(panel.theta_deg, panel.ddcs_mb_sr)
    assert angular_integral(panel.theta_deg, shape) == pytest.approx(1.0, rel=1e-12)


def test_the_generate_path_reads_the_real_library_then_reuses_the_table(monkeypatch, tmp_path):
    """End-to-end cover for the read-only generator.

    Bounded to 1 keV so the anchor reads a few hundred of the library's 86300
    files rather than all 863 belonging to the element.
    """
    library = _bremslib()
    monkeypatch.setattr("pyrite.xsgen.store.user_table_dir", lambda: tmp_path / "user")
    monkeypatch.setattr("pyrite.xsgen.store.packaged_table_dir", lambda: tmp_path / "packaged")

    first = generate_bremslib(79, t1_max_MeV=1.0e-03, source_path=library)
    second = generate_bremslib(79, t1_max_MeV=1.0e-03, source_path=library)

    assert first.generated and not second.generated
    arrays = first.table.arrays()
    energies = arrays["t1_MeV"]
    assert energies.max() <= 1.0e-03
    assert arrays["node_t1_index"].size == 13 * energies.size
    # Every node's own integral, against the published one, across the table.
    ours = arrays["node_angular_integral_mb"]
    assert np.allclose(ours, arrays["node_vendor_integral_mb"], rtol=1.0e-4)
    # And the flat angular index recovers a node whose grid is the library's.
    panel = panel_of(arrays, 0)
    assert panel.theta_deg.size == 181
    assert first.table.manifest["compiler"] is None


def test_the_pinned_release_matches_the_library_it_claims(tmp_path, monkeypatch):
    """Stale-release anchor: the shipped index against a real checkout.

    Rebuilding one released element from the local library must reproduce the
    pinned key. The key covers a digest of every library file the table was
    read from, so a mismatch means the checkout and the release come from
    different deposits -- exactly the staleness the release manifest's
    deposit record exists to expose -- or the release transform changed
    without a new ``VARIANT``.

    Bounded to Z=6, the lightest released element, which still reads its full
    30 MeV table: that is what the release holds.
    """
    library = _bremslib()
    index = load_release_index()
    if index is None:
        pytest.skip("this build pins no BremsLib table release")
    entry = index.entry(6)
    assert entry is not None, "the release must cover carbon, a catalogue element"
    monkeypatch.setattr("pyrite.xsgen.store.user_table_dir", lambda: tmp_path / "user")

    _, rebuilt = build_release(tmp_path / "out", source_path=library, elements=(6,))

    assert rebuilt.tables[0].key == entry.key


def test_the_float32_release_keeps_the_shape_function(tmp_path):
    """The release transform against the exact table, node by node.

    The shape function is what a consumer samples, so it is what the
    precision reduction must preserve. Float32 carries about seven
    significant digits; ``rtol=1e-6`` is an order of magnitude above its
    rounding and far below any physical effect, and a wrong cast -- float16,
    or the integral renormalized against the narrowed DDCS -- fails it.
    """
    library = _bremslib()
    exact = build_table(library, 6, t1_max_MeV=1.0)
    released = release_arrays(exact)

    for node in range(0, exact["node_offset"].size - 1, 97):
        want = panel_of(exact, node)
        got = panel_of(released, node)
        integral = exact["node_angular_integral_mb"][node]
        np.testing.assert_allclose(got.ddcs_mb_sr / integral, want.ddcs_mb_sr / integral, rtol=1e-6)
