"""CXR-only directional scoring, sharing one transport and staged segments."""

import numpy as np

from .directions import validated_directions
from .emission_adapters import EmissionBatch, select_emission


def source_outputs(case, directions, *, direction_batch=16, transport_core="auto"):
    """Score only incoherent PXR/CBS on an explicit common energy grid.

    Keep one transport and one staged segment copy. Direction batches bound
    scratch lifetimes; only the requested (direction, energy) result is retained.
    Use the production adapter, including layer radiation and target escape.
    No continuum, characteristic, coherent, channeling or TR scorers run.
    """
    from . import _segments_on_device, _transport_case

    vectors = validated_directions(directions)
    transport = _transport_case(case, transport_core=transport_core, keep_segments_on_device=True)
    segments = _segments_on_device(transport["segs"])
    energy = np.asarray(transport["E_grid"])
    values = np.full((len(vectors), len(energy)), np.nan)
    supported = np.abs(vectors[:, 2]) > 1e-10
    adapter = select_emission(case).cxr
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
            values[index] = np.asarray(adapter.evaluate(batch, "spec").values)
    if not np.all(np.isfinite(values[supported])):
        raise ValueError("directional source produced nonfinite density in supported directions")
    return values, supported
