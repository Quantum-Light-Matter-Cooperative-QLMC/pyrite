"""Directional scoring of selected emission components, sharing one transport."""

from dataclasses import replace

import numpy as np

from .directions import validated_directions
from .emission_adapters import EmissionBatch, select_emission

#: Detector-free directional components: (selection attribute, output slot).
#: Each keeps the live runner's electron population: PXR/CBS uses the line
#: population; characteristic and continuum use the lower-cutoff brem one.
SOURCE_COMPONENTS = {
    "pxr-cbs": ("cxr", "spec"),
    "characteristic": ("characteristic", "spec_characteristic"),
    "bremsstrahlung": ("continuum", "brem_wide"),
}


def source_outputs(
    case, directions, *, components=("pxr-cbs",), direction_batch=16, transport_core="auto"
):
    """Score the requested escaped components on an explicit common energy grid.

    Keep one transport and one staged segment copy. Direction batches bound
    scratch lifetimes; only the requested (direction, energy) results are
    retained, one array per component. Use the production adapters, including
    layer radiation and target escape: a finite footprint traces each emission
    point to its first prism face. Unrequested components, coherent,
    channeling and TR scorers never run.
    """
    from . import _segments_on_device, _transport_case

    unknown = set(components) - set(SOURCE_COMPONENTS)
    if not components or unknown:
        raise ValueError(f"components must be a nonempty subset of {sorted(SOURCE_COMPONENTS)}")
    vectors = validated_directions(directions)
    transport = _transport_case(case, transport_core=transport_core, keep_segments_on_device=True)
    segments = _segments_on_device(transport["segs"])
    energy = np.asarray(transport["E_grid"])
    values = {name: np.full((len(vectors), len(energy)), np.nan) for name in components}
    # A finite footprint gives every direction a side exit; only the laterally
    # infinite slab leaves directions parallel to its faces without an escape.
    finite = case.get("crystal_width_mm") is not None
    supported = np.full(len(vectors), True) if finite else np.abs(vectors[:, 2]) > 1e-10
    selection = select_emission(case)
    populations = {"pxr-cbs": transport["Ne_lines"]}
    for start in range(0, len(vectors), direction_batch):
        table_cache = {}
        for index in range(start, min(start + direction_batch, len(vectors))):
            if not supported[index]:
                continue
            batch = EmissionBatch(
                segments,
                energy,
                case,
                vectors[index],
                case.get("abs_layers"),
                transport.get("groove"),
                electron_limit=transport["Ne_lines"],
                table_cache=table_cache,
            )
            for name in components:
                attribute, slot = SOURCE_COMPONENTS[name]
                limit = populations.get(name, transport.get("Ne_brem", transport["Ne_lines"]))
                adapter = getattr(selection, attribute)
                output = adapter.evaluate(replace(batch, electron_limit=limit), slot)
                values[name][index] = np.asarray(output.values)
    for name, array in values.items():
        if not np.all(np.isfinite(array[supported])):
            raise ValueError(
                f"directional {name} produced nonfinite density in supported directions"
            )
    return values, supported
