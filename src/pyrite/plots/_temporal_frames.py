"""Renderer-neutral time-domain intensity profiles (#292, #622)."""

import numpy as np
import pandas as pd

from ..results import records
from ._spectrum_frames import _peak_preserving_indices

_COMPONENTS = (
    ("temporal_intensity", "incoherent"),
    ("temporal_intensity_coherent", "coherent"),
)


def temporal_frame(results, *, hue="E0_keV", include_coherent=True, max_points=5000):
    """Long frame ``time_fs, intensity, <hue>, component`` over records with a profile.

    Time is relative to the intensity-weighted mean arrival of each trace, so
    traces from different cases share one readable axis. Empty when no record
    carries a temporal profile.
    """
    rows = []
    for record in records(results):
        t = record.get("temporal_t_fs")
        if t is None:
            continue
        t = np.asarray(t, dtype=float)
        for key, component in _COMPONENTS:
            intensity = record.get(key)
            if intensity is None or (component == "coherent" and not include_coherent):
                continue
            intensity = np.asarray(intensity, dtype=float)
            mass = intensity.sum()
            centre = (t * intensity).sum() / mass if mass > 0 else 0.0
            keep = _peak_preserving_indices(intensity, max(3, max_points // 4))
            rows.append(
                pd.DataFrame(
                    {
                        "time_fs": t[keep] - centre,
                        "intensity": intensity[keep],
                        hue: record["case"].get(hue),
                        "component": component,
                    }
                )
            )
    if not rows:
        return pd.DataFrame(columns=["time_fs", "intensity", hue, "component"])
    return pd.concat(rows, ignore_index=True)
