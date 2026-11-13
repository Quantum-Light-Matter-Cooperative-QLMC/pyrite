"""Private submitter/host comparison through the existing scan identity owners.

No transport, checkpoint I/O, plugin shipping, or independent cache hash schema.
The compact record compares the existing resolved dataset identity/table union.
"""

import json
from argparse import Namespace


def _admit_axial_cases(settings, sweep) -> None:
    """Lower an axial run's cases so local per-case admission refuses before remote work.

    Scan resolution refuses only an unverified axial selector; each fibre case
    is admitted against the repository lookup when its cases are built (#624).
    Other runs are not lowered here.
    """
    if settings.channeling_model != "axial":
        return
    from ..api import build_configured_cases

    build_configured_cases(sweep, settings)


def resolve_inventory(request: dict) -> dict:
    """Resolve every material/detector using the scan's catalog and table paths."""
    from ..materials import CATALOG
    from ..runs.scan import _resolved_run

    inventory = {}
    for material in request["materials"]:
        for detector_id in CATALOG.profile_detector_set(request["catalog_profile"]):
            args = Namespace(
                allow_physics_fallback=request.get("allow_physics_fallback", False),
                quick=request["quick"],
                catalog_profile=request["catalog_profile"],
                detector_id=detector_id,
                high_energy_floor_map=request.get("high_energy_floor_map", {}),
            )
            settings, sweep, identity, stem = _resolved_run(args, material)
            _admit_axial_cases(settings, sweep)
            # The existing dataset digest includes model generations, selected
            # parameters and the union of consumed tables. Do not lower every
            # case or substitute this union for per-case CAS/source keys.
            inventory[f"{material}/{detector_id}"] = {
                "stem": stem,
                "dataset_digest": identity["parameter_sha256"],
                "table_markers": identity["resolved_parameters"].get("xsgen_tables", {}),
            }
    return inventory


def expected_inventory(
    materials,
    quick,
    catalog_profile,
    high_energy_min_kev=None,
    *,
    allow_physics_fallback=False,
):
    """Data-only job input, reconstructed from the selected staged catalog."""
    from ._queue_scripts import _LEGACY_HIGH_ENERGY_MATERIALS

    request = {
        "allow_physics_fallback": allow_physics_fallback,
        "materials": list(materials),
        "quick": bool(quick),
        "catalog_profile": catalog_profile,
        "high_energy_floor_map": {
            material: high_energy_min_kev
            for material in materials
            if high_energy_min_kev is not None and material in _LEGACY_HIGH_ENERGY_MATERIALS
        },
    }
    return {"request": request, "inventory": resolve_inventory(request)}


def verify_inventory(expected: dict) -> None:
    """Fail closed before cache reads if host selection or consumed data differs."""
    current = resolve_inventory(expected["request"])
    recorded = expected["inventory"]
    if current != recorded:
        changed = sorted(
            key for key in current.keys() | recorded.keys() if current.get(key) != recorded.get(key)
        )
        raise ValueError(
            "remote physics identity differs from submission for "
            + ", ".join(changed)
            + "; compare the selected catalog and consumed tables, sync, and resubmit"
        )


def main() -> None:
    """Internal module entry used only by generated job scripts."""
    import sys

    try:
        verify_inventory(json.loads(sys.argv[1]))
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
