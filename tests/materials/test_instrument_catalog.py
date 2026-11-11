"""Internal instrument catalog objects: curves, gratings, ALEX-s detectors (#1080).

Every curve and grating here is synthetic; no vendor data belongs in tests.
"""

import pytest

from pyrite import _catalog_layout as layout
from pyrite.materials import MaterialConfigError, load_material_catalog
from pyrite.materials._curves import CurveError, parse_curve

_CSV = """\
# source = "synthetic test curve"
# revision = "A"
# digitization = "synthetic, hand-typed"
# confidential = true
# quantity = "quantum_efficiency"
wavelength [nm], value [percent]
10.0, 20.0
20.0, 50.0
40.0, 80.0
"""

_GRATING = """\
label = "Synthetic flat-field grating"
groove_density_per_mm = 1200
order = -1
order_convention = "sum"
coating = "Au"
blank_mm = [30.0, 50.0]
efficiency_curve = "geff"

[mount]
r_mm = 350.0
alpha_deg = 85.0
band_nm = [5.0, 20.0]
r_prime_mm = [470.0, 460.0]
beta_deg = [-80.0, -75.0]
beta_H_deg = -78.0
r_H_prime_mm = 465.0
focal_plane_length_mm = "2.5 cm"
"""

_GRATING_CURVE = """\
source = "synthetic grating efficiency"
digitization = "analytic toy"
confidential = false
quantity = "efficiency"
abscissa = "energy"
abscissa_unit = "keV"
x = [0.05, 0.1, 0.2]
y = [0.1, 0.2, 0.15]
"""

_DETECTOR = """\
distance_mm = 500.0
shape = [2048, 512]
pitch_mm = [0.0135, 0.0135]

[response]
kind = "alexs"
sensor_format = [2048, 512]
sensor_code = "SYNTH-BI"
readout_rate_hz = "1 MHz"
gain_mode = "high"
qe_curve = "qe"
"""


@pytest.fixture
def layer(_isolate_user_catalog):
    load_material_catalog.cache_clear()
    root = _isolate_user_catalog
    for table in ("curves", "gratings", "detectors"):
        (root / table).mkdir(exist_ok=True)
    (root / "curves" / "qe.csv").write_text(_CSV)
    (root / "curves" / "geff.toml").write_text(_GRATING_CURVE)
    (root / "gratings" / "synthetic-ff.toml").write_text(_GRATING)
    (root / "detectors" / "lab-ccd.toml").write_text(_DETECTOR)
    yield root
    load_material_catalog.cache_clear()


def _errors(**kwargs) -> str:
    load_material_catalog.cache_clear()
    with pytest.raises(MaterialConfigError) as excinfo:
        load_material_catalog(**kwargs)
    return str(excinfo.value)


def test_user_layer_loads_lists_and_resolves_curves(layer):
    catalog = load_material_catalog()

    grating = catalog.gratings["synthetic-ff"]
    assert grating["order"] == -1 and grating["order_convention"] == "sum"
    assert grating["mount"]["focal_plane_length_mm"] == pytest.approx(25.0)
    assert grating["mount"]["beta_deg"] == (-80.0, -75.0)
    response = catalog.detectors["lab-ccd"]["response"]
    assert response["kind"] == "alexs"
    assert response["readout_rate_hz"] == pytest.approx(1.0e6)
    assert response["sensor_format"] == (2048, 512)
    assert set(catalog.curves) == {"qe", "geff"}

    qe = catalog.curves["qe"]
    assert qe.confidential and qe.revision == "A" and qe.abscissa == "wavelength"
    # Wavelength rows convert to ascending energy; percent converts to fractions.
    assert qe.value == pytest.approx((0.8, 0.5, 0.2))
    assert qe.energy_eV[0] == pytest.approx(1239.84198 / 40.0, rel=1e-6)
    assert all(b > a for a, b in zip(qe.energy_eV, qe.energy_eV[1:], strict=False))
    assert catalog.curves["geff"].energy_eV == pytest.approx((50.0, 100.0, 200.0))

    bundled = layout.bundled_catalog()
    assert layout.object_sources(bundled, "gratings") == {"synthetic-ff": "user"}
    assert layout.object_sources(bundled, "detectors")["lab-ccd"] == "user"
    assert "synthetic-ff" in layout.object_keys(bundled, "gratings")


def test_complete_directory_catalog_reports_catalog_source(tmp_path, layer):
    import shutil

    external = tmp_path / "external"
    shutil.copytree(layout.bundled_catalog(), external)
    for name in ("curves", "gratings"):
        shutil.copytree(layer / name, external / name)

    catalog = load_material_catalog(external)

    assert set(catalog.curves) == {"geff"}
    assert layout.object_sources(external, "gratings") == {"synthetic-ff": "catalog"}


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("wavelength [nm]", "wavelength [kg]", "not a length unit"),
        ("wavelength [nm]", "momentum [nm]", "abscissa must be one of"),
        ("20.0, 50.0", "5.0, 50.0", "strictly increasing"),
        ("20.0, 50.0", "20.0, 150.0", "must lie in [0, 1]"),
        ("value [percent]", "value [nm]", "not dimensionless"),
        ("20.0, 50.0", "20.0, nan", "must be finite"),
        ("# confidential = true\n", "", "confidential must be true or false"),
        ('# source = "synthetic test curve"\n', "", "source must be a nonempty string"),
    ],
)
def test_bad_curves_fail_with_clear_errors(layer, old, new, message):
    (layer / "curves" / "qe.csv").write_text(_CSV.replace(old, new))

    errors = _errors()

    assert "detectors.lab-ccd.response.qe_curve" in errors
    assert message in errors


def test_curve_paths_stay_inside_curves(layer, tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_text(_CSV)
    (layer / "curves" / "qe.csv").unlink()
    (layer / "curves" / "qe.csv").symlink_to(outside)

    assert "unsafe symlink" in _errors()


def test_curve_name_cannot_escape_or_be_ambiguous(layer):
    (layer / "detectors" / "lab-ccd.toml").write_text(
        _DETECTOR.replace('qe_curve = "qe"', 'qe_curve = "../escape"')
    )
    assert "curve name must use letters" in _errors()

    (layer / "detectors" / "lab-ccd.toml").write_text(_DETECTOR)
    (layer / "curves" / "qe.toml").write_text(_GRATING_CURVE)
    assert "defined by both qe.csv and qe.toml" in _errors()


def test_unknown_curve_reference_names_the_search_directory(layer):
    (layer / "curves" / "geff.toml").unlink()

    errors = _errors()

    assert "gratings.synthetic-ff.efficiency_curve: unknown curve 'geff'" in errors
    assert str(layer / "curves") in errors


def test_toml_curve_validates_lengths(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(_GRATING_CURVE.replace("y = [0.1, 0.2, 0.15]", "y = [0.1, 0.2]"))

    with pytest.raises(CurveError, match="equal lengths"):
        parse_curve("bad", path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("order = -1", "order = 0", "gratings.synthetic-ff.order: must be a nonzero integer"),
        ('order_convention = "sum"', 'order_convention = "left"', "must be one of"),
        ("band_nm = [5.0, 20.0]\n", "", "requires band_nm (values at the band ends)"),
        ("r_mm = 350.0", "r_mm = -1", "r_mm: must be a finite positive number"),
        ("alpha_deg = 85.0", 'alpha_deg = "2 rad"', "alpha_deg: must lie strictly between"),
        ("coating", "coatng", "gratings.synthetic-ff.coatng: unknown key"),
        ("groove_density_per_mm = 1200", 'groove_density_per_mm = "1200 mm"', "1/mm"),
    ],
)
def test_bad_gratings_fail_with_clear_errors(layer, old, new, message):
    (layer / "gratings" / "synthetic-ff.toml").write_text(_GRATING.replace(old, new))

    assert message in _errors()


def test_alexs_response_requires_its_fields(layer):
    (layer / "detectors" / "lab-ccd.toml").write_text(
        _DETECTOR.replace('gain_mode = "high"\n', "bias_v = 100\n")
    )

    errors = _errors()

    assert "detectors.lab-ccd.response.gain_mode: missing required key" in errors
    assert "detectors.lab-ccd.response.bias_v: unknown key" in errors


def test_alexs_lowering_is_not_yet_supported(layer):
    from pyrite.campaign.observation import physical_detector_from_config

    detector = load_material_catalog().detectors["lab-ccd"]

    with pytest.raises(NotImplementedError, match="not yet supported"):
        physical_detector_from_config(detector)


def test_bundled_catalog_declares_no_gratings_or_curves():
    bundled = layout.bundled_catalog()

    assert not (bundled / "gratings").exists()
    assert not (bundled / "curves").exists()
    catalog = load_material_catalog(bundled)
    assert not catalog.gratings and not catalog.curves
