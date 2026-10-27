"""Production admission and original-pins replay for charged filter re-entry."""

from copy import deepcopy
from dataclasses import replace

import pytest

from pyrite.montecarlo.runner import photon_aperture as photon
from pyrite.montecarlo.transport.photon_histories import PhotonAperture
from pyrite.montecarlo.transport.photon_navigation import PhotonBox
from tests.montecarlo.test_coupled_radiative_runner import (
    synthetic_si_tables as synthetic_si_tables,
)
from tests.montecarlo.test_photon_coupling import silicon_stage as silicon_stage
from tests.montecarlo.test_photon_production import selected as selected


@pytest.fixture
def filter_case(selected):
    base = replace(
        selected.base,
        elastic_model="elsepa",
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
    )
    config = replace(selected.config, charged_feedback=True, electron_cutoff_eV=5000.0)
    box = PhotonBox(
        (0.0, 0.0, 5e5),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        (1e5, 1e5, 1e4),
        base.composition,
    )
    aperture = PhotonAperture((0.0, 0.0, 1e6), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2e5, 2e5)
    return base, config, box, aperture


def _context(filter_case, **kw):
    base, config, box, aperture = filter_case
    return photon.prepare_source(
        base,
        config,
        aperture,
        [box],
        base.crystal_width_mm * 1e7,
        base.crystal_height_mm * 1e7,
        "per-electron",
        box_materials=["silicon"],
        **kw,
    )


def test_production_admits_filter_reentry_and_pins_every_body(filter_case):
    context = _context(filter_case)
    stage = context["stage"]
    assert len(stage.regions) == 2
    assert stage.backend == "cpu"
    feedback = context["feedback_identity"]
    assert feedback["model"] == "cpu-charged-scene-feedback-v1"
    assert feedback["box_materials"] == ["silicon"]
    assert feedback["tables"] and feedback["positron_tables"]
    assert feedback["max_crossings"] == stage.max_crossings
    assert [r.stage.base.transport["inelastic_materials"] for r in stage.regions] == [
        ["silicon"],
        ["silicon"],
    ]


def test_reentry_record_reconstructs_materials_and_refuses_changed_pins(filter_case, monkeypatch):
    base, _, _, _ = filter_case
    context = _context(filter_case)
    record = photon.photon_case_record(base, "0" * 64, context)
    case = replace(base, photon_transport=record)
    restored = photon.prepare_photon_case(case, "per-electron")
    assert restored["feedback_identity"] == context["feedback_identity"]
    corrupt = deepcopy(record)
    corrupt["identity"]["charged_feedback"]["tables"]["new-table"] = "1" * 64
    with pytest.raises(ValueError, match="identity or data pins changed"):
        photon.prepare_photon_case(replace(base, photon_transport=corrupt), "per-electron")


def test_filter_composition_must_match_its_charged_material_key(filter_case):
    base, config, box, aperture = filter_case
    with pytest.raises(ValueError, match="composition"):
        photon.prepare_source(
            base,
            config,
            aperture,
            [replace(box, composition=(("Cu", 0.08),))],
            base.crystal_width_mm * 1e7,
            base.crystal_height_mm * 1e7,
            "per-electron",
            box_materials=["silicon"],
        )


def test_feedback_filter_without_shell_material_refused_before_primary(filter_case):
    base, config, box, aperture = filter_case
    with pytest.raises(ValueError, match="catalog material"):
        photon.prepare_source(
            base,
            config,
            aperture,
            [box],
            base.crystal_width_mm * 1e7,
            base.crystal_height_mm * 1e7,
            "per-electron",
        )


def test_catalog_filter_without_shell_data_refuses_before_resource_loading(filter_case):
    from pyrite.materials import CATALOG
    from pyrite.materials._composition import resolve_composition

    base, config, box, aperture = filter_case
    copper = replace(box, composition=resolve_composition("copper", catalog=CATALOG))
    with pytest.raises(ValueError, match="no conduction-band shell data"):
        photon.prepare_source(
            base,
            config,
            aperture,
            [copper],
            base.crystal_width_mm * 1e7,
            base.crystal_height_mm * 1e7,
            "per-electron",
            box_materials=["copper"],
        )
