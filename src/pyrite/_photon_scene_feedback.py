"""Production preparation of body-local charged scene feedback and table pins."""

from types import SimpleNamespace

CPU_SCENE_MODEL = "cpu-charged-scene-feedback-v1"
CUDA_SCENE_MODEL = "cuda-charged-scene-feedback-v1"


def prepare_feedback_scene(
    case, layers, boxes, box_materials, width, height, numerics, photon_cutoff, electron_cutoff
):
    """Resolve a charged material/table bundle for every photon-scene body.

    Catalog keys must resolve to the exact box composition. Source transport
    remains target-only; these additional resources belong to the enabled
    observation, including both projectile table families. No material means
    no charged stage in an explicit vacuum layer. Limits and energy
    substitution are those of ``scene_feedback_stage``.
    Validation: photon-charged-reentry
    """
    from ._photon_support import _feedback_table_markers
    from .montecarlo.runner.case_tables import (
        _case_bremslib_tables,
        _case_elastic_kwargs,
        _case_stopping_tables,
        case_table_markers,
    )
    from .montecarlo.transport.photon_reentry import scene_feedback_stage

    materials, compositions = _scene_materials(case, layers, boxes, box_materials)
    stopping, elastic, brems, tables, positron = [], [], {}, {}, {}
    for key, composition in zip(materials, compositions, strict=True):
        if not composition:
            stopping.append(None)
            elastic.append(None)
            continue
        body = dict(
            case, crystal=key, composition=composition, abs_layers=None, layer_radiators=None
        )
        stopping.extend(_case_stopping_tables(body))
        elastic.extend(_case_elastic_kwargs(body)["elastic_tables"])
        body_brems = _case_bremslib_tables(body)
        if set(element for element, _ in composition) - body_brems.keys():
            raise ValueError(
                "charged scene feedback requires BremsLib tables for every body element"
            )
        brems.update(body_brems)
        tables.update(case_table_markers(body))
        positron.update(
            _feedback_table_markers(
                SimpleNamespace(
                    base=SimpleNamespace(
                        transport={
                            "inelastic_materials": [key],
                            "elastic_model": "elsepa",
                            "layers": [(0.0, 1.0, composition)],
                        }
                    )
                )
            )
        )
    backend = "cuda" if numerics.transport_core == "cuda" else "cpu"
    stage = scene_feedback_stage(
        layers,
        boxes=boxes,
        photon_cutoff_eV=photon_cutoff,
        width_ang=width,
        height_ang=height,
        backend=backend,
        stopping_tables=stopping,
        elastic_model="elsepa",
        elastic_tables=elastic,
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=float(case["inelastic_cutoff_eV"]),
        inelastic_materials=materials,
        secondary_threshold_eV=electron_cutoff,
        E_cut_keV=electron_cutoff * 1e-3,
        transport_core=numerics.transport_core,
        energy_model=case.get("energy_model", "frozen"),
        straggling=bool(case.get("straggling", False)),
        max_dE_frac=case.get("max_dE_frac", 0.0),
        atomic_electron_deflection=case.get("atomic_electron_deflection"),
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=case["radiative_cutoff_eV"],
        bremslib_tables=brems,
    )
    return stage, {
        "model": CUDA_SCENE_MODEL if backend == "cuda" else CPU_SCENE_MODEL,
        "box_materials": list(box_materials),
        "tables": tables,
        "positron_tables": positron,
        "max_crossings": stage.max_crossings,
        "max_relaxation_rounds": [region.stage.max_relaxation_rounds for region in stage.regions],
        "navigation": "analytic-binary64-device" if backend == "cuda" else "analytic-binary64-host",
        "charged_transport_core": "cuda" if backend == "cuda" else numerics.transport_core,
    }


def _scene_materials(case, layers, boxes, box_materials):
    """Validate exact catalog ownership before loading any body-local resources."""
    from .materials import CATALOG
    from .materials._composition import resolve_composition
    from .montecarlo.runner.case_tables import case_shell_materials
    from .montecarlo.transport.shell_transport import validate_shell_cutoff

    if case.get("elastic_model") != "elsepa":
        raise ValueError("charged scene feedback requires ELSEPA elastic tables")
    if (
        box_materials is None
        or len(box_materials) != len(boxes)
        or any(not isinstance(key, str) for key in box_materials)
    ):
        raise ValueError("charged filters require one catalog material key per box")
    for key, box in zip(box_materials, boxes, strict=True):
        composition = resolve_composition(key, catalog=CATALOG)
        if tuple(map(tuple, composition)) != tuple(map(tuple, box.composition)):
            raise ValueError("charged filter material key does not match its photon composition")
    materials = [*case_shell_materials(case), *box_materials]
    compositions = [layer[2] for layer in layers] + [box.composition for box in boxes]
    if len(materials) != len(compositions):
        raise ValueError("charged target materials must cover every target layer")
    active = [key for key, composition in zip(materials, compositions, strict=True) if composition]
    if any(key is None for key in active):
        raise ValueError(
            "charged scene feedback requires a catalog material for every material body"
        )
    validate_shell_cutoff(active, float(case["inelastic_cutoff_eV"]))
    return materials, compositions
