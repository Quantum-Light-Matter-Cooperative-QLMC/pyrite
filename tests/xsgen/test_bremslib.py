"""Contracts for reading and converting the BremsLib library.

No BremsLib here: its 810 MB library is not redistributed, so the fixtures
below write a miniature library in the upstream file format -- header line,
one triplet per ``k/T1`` column, ragged angular grids, the three- and
four-column DDCS variants -- and the anchors against the real thing live in
``test_extern_codes.py``.
"""

from pathlib import Path

import numpy as np
import pytest

from pyrite.xsgen._errors import SourceUnavailableError
from pyrite.xsgen.bremslib import (
    angular_integral,
    build_table,
    ddcs_filename,
    generate_element,
    iter_node_files,
    library_root,
    library_version,
    panel_of,
    parse_ddcs,
    parse_ratio_table,
    shape_function,
)
from pyrite.xsgen.bremslib.read import RATIO_COUNT, node_index

#: The upstream ``k/T1`` grid. The last value is nominal: the data behind it
#: stand slightly below 1 by a ``T1``-dependent amount.
RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.0)

#: Two incident energies, straddling the 1 MeV boundary at which the upstream
#: angular grid changes length, so every fixture is ragged.
ENERGIES = (1.0e-03, 2.0e00)

#: Angular grid lengths per energy, shortened from the real 181 and 221 but
#: keeping an even number of intervals, which composite Simpson needs.
GRID_POINTS = {1.0e-03: 5, 2.0e00: 9}


def _ratio_file(kind: str, energies, value) -> str:
    """Render an SDCS-shaped file: header, then one line per energy."""
    labels = ["T1(MeV)"]
    for ratio in RATIOS:
        labels += [f"{kind}[Ep/T1={ratio:.7f}]", f"RelErr[Ep/T1={ratio:.3f}]", "pnt/fin"]
    lines = ["\t".join(labels)]
    for energy in energies:
        row = [f"{energy:.17E}"]
        for column, _ in enumerate(RATIOS):
            row += [f"{value(energy, column):.17E}", "1.0E-06", "1.00000000000000"]
        lines.append("  ".join(row))
    return "\n".join(lines) + "\n"


def _ddcs_file(points: int, *, level: float, columns: int = 4) -> str:
    """Render one DDCS node file with ``points`` angular samples."""
    theta = np.linspace(0.0, 180.0, points)
    lines = [" theta\t DDCS_scaled\t relErr\t point/finite"[: None if columns == 4 else 30]]
    for angle in theta:
        row = [f"{angle:6.2f}", f"{level:.8e}", "3.915e-06"]
        if columns == 4:
            row.append("0.99983983")
        lines.append("\t".join(row))
    return "\n".join(lines) + "\n"


def _write_library(root: Path, z: int = 79, *, level: float = 2.0) -> Path:
    """Write a miniature but structurally complete library under ``root``.

    The ``DDCS_int`` file deliberately carries a *shorter* energy grid than
    the SDCS file, as the real library does: upstream publishes integrals up
    to 100 MeV and cross sections up to 300 MeV, on grids that do not line up
    row for row.
    """
    library = root / "BremsLib_v2.0.8"
    (library / "SDCS" / "DDCS_int").mkdir(parents=True)
    (library / "DDCS").mkdir(parents=True)

    def sdcs_value(energy: float, column: int) -> float:
        return level * 4.0 * np.pi * (1.0 + column)

    # One extra, higher energy the SDCS carries and the integrals do not.
    sdcs_energies = (*ENERGIES, 5.0e01)
    (library / "SDCS" / f"SDCS_{z}.txt").write_text(
        _ratio_file("CS", sdcs_energies, sdcs_value), encoding="ascii"
    )
    (library / "SDCS" / "DDCS_int" / f"DDCS_int_{z}.txt").write_text(
        _ratio_file("DDCS_int", ENERGIES, sdcs_value), encoding="ascii"
    )
    for energy in ENERGIES:
        for column, ratio in enumerate(RATIOS):
            name = ddcs_filename(z, energy, ratio * energy)
            (library / "DDCS" / name).write_text(
                _ddcs_file(
                    GRID_POINTS[energy],
                    level=level * (1.0 + column),
                    columns=4 if energy > 1.0 else 3,
                ),
                encoding="ascii",
            )
    return library


@pytest.fixture
def library(tmp_path) -> Path:
    return _write_library(tmp_path / "BremsLib_v2.0.8")


# --- file formats ---------------------------------------------------------


def test_ratio_table_reads_the_thirteen_labelled_columns(library):
    table = parse_ratio_table((library / "SDCS" / "SDCS_79.txt").read_text(encoding="ascii"))

    assert table.k_over_t1.tolist() == list(RATIOS)
    assert table.value.shape == (3, RATIO_COUNT)
    assert table.row(2.0) == 1
    assert table.value[0, 0] == pytest.approx(2.0 * 4.0 * np.pi)


def test_ratio_table_rejects_a_row_of_the_wrong_width(library):
    text = (library / "SDCS" / "SDCS_79.txt").read_text(encoding="ascii")
    truncated = text.replace("  1.00000000000000", "", 1)

    with pytest.raises(ValueError, match="columns, expected 40"):
        parse_ratio_table(truncated)


def test_ratio_table_rejects_an_unlabelled_header(library):
    text = (library / "SDCS" / "SDCS_79.txt").read_text(encoding="ascii")

    with pytest.raises(ValueError, match="labelled ratio"):
        parse_ratio_table(text.replace("CS[Ep/T1=0.0000000]", "CS", 1))


def test_ratio_table_rejects_a_descending_energy_grid():
    rows = _ratio_file("CS", (2.0, 1.0), lambda energy, column: 1.0)

    with pytest.raises(ValueError, match="ascending"):
        parse_ratio_table(rows)


def test_ddcs_panel_reads_both_column_counts(library):
    four = parse_ddcs((library / "DDCS" / "DDCS_79_2.0E+00_0.000E+00.txt").read_text())
    three = parse_ddcs((library / "DDCS" / "DDCS_79_1.0E-03_0.000E+00.txt").read_text())

    assert four.finite_nucleus and four.point_finite[0] == pytest.approx(0.99983983)
    # No fourth column means the point-nucleus result does not differ
    # significantly, which is a ratio of one rather than a missing value.
    assert not three.finite_nucleus
    assert three.point_finite.tolist() == [1.0] * three.theta_deg.size


def test_ddcs_panel_requires_a_grid_spanning_the_full_range():
    text = _ddcs_file(5, level=1.0).replace("180.00", "170.00")

    with pytest.raises(ValueError, match="0 to 180"):
        parse_ddcs(text)


def test_ddcs_panel_rejects_mixed_row_widths():
    text = _ddcs_file(5, level=1.0)
    lines = text.splitlines()
    lines[2] = "\t".join(lines[2].split("\t")[:3])

    with pytest.raises(ValueError, match="three- and four-column"):
        parse_ddcs("\n".join(lines))


def test_ddcs_filename_reproduces_the_upstream_naming():
    assert ddcs_filename(79, 1.0, 0.5) == "DDCS_79_1.0E+00_5.000E-01.txt"
    assert ddcs_filename(7, 1.2e-05, 0.0) == "DDCS_7_1.2E-05_0.000E+00.txt"
    # Four significant digits round the top node's photon energy onto the
    # incident energy: 0.9999 * 30 MeV is named as though the ratio were 1.
    assert ddcs_filename(79, 30.0, 29.997) == "DDCS_79_3.0E+01_3.000E+01.txt"


def test_node_index_snaps_the_top_node_onto_the_nominal_unit_ratio():
    assert node_index(RATIOS, 0.9999) == RATIO_COUNT - 1
    assert node_index(RATIOS, 0.99) == RATIO_COUNT - 1
    assert node_index(RATIOS, 0.95) == 10
    assert node_index(RATIOS, 0.0) == 0


def test_node_index_rejects_a_ratio_that_is_not_a_grid_node():
    with pytest.raises(ValueError, match="not a BremsLib grid node"):
        node_index(RATIOS, 0.42)


# --- library layout ------------------------------------------------------


def test_library_root_accepts_a_deposit_or_the_data_directory_itself(library):
    assert library_root(library.parent) == library
    assert library_root(library) == library


def test_library_root_names_what_it_looked_for(tmp_path):
    with pytest.raises(SourceUnavailableError, match="SDCS/SDCS_<Z>.txt"):
        library_root(tmp_path)


def test_node_listing_is_sorted_and_element_exact(library):
    (library / "DDCS" / "DDCS_7_1.0E-03_0.000E+00.txt").write_text("", encoding="ascii")

    nodes = iter_node_files(library, 79)

    assert len(nodes) == len(ENERGIES) * RATIO_COUNT
    assert [node.t1_MeV for node in nodes] == sorted(node.t1_MeV for node in nodes)
    assert all("DDCS_79_" in node.path.name for node in nodes)


def test_node_listing_fails_for_an_element_the_library_lacks(library):
    with pytest.raises(SourceUnavailableError, match="no DDCS files for Z=13"):
        iter_node_files(library, 13)


# --- the derived quantities ----------------------------------------------


def test_shape_function_integrates_to_one_on_the_native_grid(library):
    node = library / "DDCS" / ddcs_filename(79, 2.0, 0.5 * 2.0)
    panel = parse_ddcs(node.read_text())

    shape = shape_function(panel.theta_deg, panel.ddcs_mb_sr)

    assert angular_integral(panel.theta_deg, shape) == pytest.approx(1.0, rel=1e-12)


def test_angular_integral_refuses_a_node_with_no_cross_section():
    theta = np.linspace(0.0, 180.0, 5)

    with pytest.raises(ValueError, match="no positive finite angular integral"):
        angular_integral(theta, np.zeros_like(theta))


def test_angular_integral_recovers_an_isotropic_solid_angle(library):
    """A flat DDCS integrates to ``4*pi`` times its level, up to the rule's error."""
    panel = parse_ddcs((library / "DDCS" / "DDCS_79_2.0E+00_0.000E+00.txt").read_text())

    integral = angular_integral(panel.theta_deg, panel.ddcs_mb_sr)

    assert integral == pytest.approx(4.0 * np.pi * 2.0, rel=2e-3)


# --- the stored table ----------------------------------------------------


def test_build_table_indexes_ragged_angular_grids(library):
    arrays = build_table(library, 79)

    assert arrays["t1_MeV"].tolist() == list(ENERGIES)
    assert arrays["node_t1_index"].size == len(ENERGIES) * RATIO_COUNT
    # Ascending in (T1, k/T1), and each node's slice is its own grid length.
    assert arrays["node_t1_index"][:RATIO_COUNT].tolist() == [0] * RATIO_COUNT
    assert arrays["node_k_index"][:RATIO_COUNT].tolist() == list(range(RATIO_COUNT))
    lengths = np.diff(arrays["node_offset"])
    assert lengths[0] == GRID_POINTS[ENERGIES[0]]
    assert lengths[-1] == GRID_POINTS[ENERGIES[1]]
    assert int(arrays["node_offset"][-1]) == arrays["theta_deg"].size

    first = panel_of(arrays, 0)
    assert first.theta_deg.tolist() == [0.0, 45.0, 90.0, 135.0, 180.0]
    assert not first.finite_nucleus


def test_build_table_reads_the_vendor_integral_off_its_own_energy_grid(library):
    """The two files' energy grids do not line up, so the row must be looked up.

    Reusing the SDCS row index would read the integral for a different
    energy, which is undetectable in the stored numbers.
    """
    arrays = build_table(library, 79)

    ours = arrays["node_angular_integral_mb"]
    vendor = arrays["node_vendor_integral_mb"]
    sdcs = arrays["sdcs_mb"][arrays["node_t1_index"], arrays["node_k_index"]]
    assert np.allclose(vendor, sdcs, rtol=0.0, atol=0.0)
    assert np.allclose(ours, vendor, rtol=3e-3)


def test_build_table_bounds_itself_with_a_maximum_energy(library):
    arrays = build_table(library, 79, t1_max_MeV=1.0)

    assert arrays["t1_MeV"].tolist() == [ENERGIES[0]]
    assert arrays["node_t1_index"].size == RATIO_COUNT


def test_build_table_refuses_a_hole_in_the_node_grid(library):
    (library / "DDCS" / ddcs_filename(79, 2.0, 0.9 * 2.0)).unlink()

    with pytest.raises(ValueError, match="missing 1 of 26 DDCS nodes"):
        build_table(library, 79)


def test_build_table_refuses_an_energy_range_holding_nothing(library):
    with pytest.raises(ValueError, match="no Z=79 nodes at or below"):
        build_table(library, 79, t1_max_MeV=1e-09)


# --- generation, keying, and provenance ----------------------------------


def test_generate_stores_then_reuses_without_rereading(library, isolated_dirs, packaged_dir):
    first = generate_element(79, source_path=library)
    second = generate_element(79, source_path=library)

    assert first.generated and not second.generated
    assert second.table.key == first.table.key
    arrays = second.table.arrays()
    assert arrays["node_t1_index"].size == len(ENERGIES) * RATIO_COUNT


def test_generated_manifest_records_the_deposit_and_claims_no_compiler(
    library, isolated_dirs, packaged_dir
):
    manifest = generate_element(79, source_path=library).table.manifest

    assert manifest["code"] == "bremslib"
    assert manifest["quantity"] == "bremsstrahlung_sdcs_ddcs"
    assert manifest["target"] == {"kind": "element", "z": 79}
    # The deposit DOI is what a stale shipped table is detected against, and
    # the directory name says which unpacked copy was read.
    assert "10.17632/6zfsc9xsz8.9" in manifest["upstream"]
    assert "BremsLib_v2.0.8" in manifest["upstream"]
    # Nothing was compiled, so claiming a compiler would be false provenance.
    assert manifest["compiler"] is None


def test_the_energy_bound_is_part_of_the_key(library, isolated_dirs, packaged_dir):
    whole = generate_element(79, source_path=library)
    bounded = generate_element(79, t1_max_MeV=1.0, source_path=library)

    assert bounded.generated
    assert bounded.table.key != whole.table.key


def test_a_changed_library_is_a_changed_table(library, isolated_dirs, packaged_dir):
    before = generate_element(79, source_path=library)

    sdcs = library / "SDCS" / "SDCS_79.txt"
    sdcs.write_text(sdcs.read_text().replace("1.0E-06", "2.0E-06"), encoding="ascii")
    after = generate_element(79, source_path=library)

    assert after.generated
    assert after.table.key != before.table.key


def test_library_version_follows_a_node_file_changing_size(library):
    nodes = iter_node_files(library, 79)
    before = library_version(library, 79, nodes)

    node = nodes[0].path
    node.write_text(node.read_text() + "\n", encoding="ascii")

    assert library_version(library, 79, nodes) != before


def test_generate_rejects_an_element_outside_the_library(library, isolated_dirs):
    with pytest.raises(ValueError, match="Z = 1 to 100"):
        generate_element(101, source_path=library)


def test_generate_rejects_a_non_positive_energy_bound(library, isolated_dirs):
    with pytest.raises(ValueError, match="t1_max_MeV must be positive"):
        generate_element(79, t1_max_MeV=0.0, source_path=library)
