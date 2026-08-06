import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from cxr_mc.plots.trajectories import _draw_trajectory_panel


def test_matplotlib_vacuum_legs_are_separate_faint_energy_collection():
    material = {
        "px": np.array([0.0, 0.5, 1.0]),
        "py": np.array([0.0, 0.25, 0.5]),
        "pE": np.array([30.0, 25.0, 20.0]),
    }
    original_material = {name: values.copy() for name, values in material.items()}
    data = {
        **material,
        "nslab": np.array([0.0, 1.0]),
        "ndet": np.array([1.0, 0.0]),
        "thick": 2.0,
        "beam": np.array([0.0, 0.0, 1.0]),
        "detector": np.array([1.0, 0.0, 0.0]),
        "vacuum_start_xyz": np.array([[0.0, 0.0, 0.5], [0.5, 0.0, 1.0]]),
        "vacuum_end_xyz": np.array([[0.5, 0.0, 0.75], [1.0, 0.0, 1.25]]),
        "vacuum_E": np.array([24.0, 18.0]),
    }
    fig, ax = plt.subplots()
    try:
        _draw_trajectory_panel(
            ax,
            data,
            (-1.0, 2.0, -1.0, 2.0),
            30.0,
            px=80,
            spread_px=0,
            label=False,
        )

        vacuum_collections = [
            collection for collection in ax.collections if isinstance(collection, LineCollection)
        ]
        assert len(vacuum_collections) == 1
        vacuum = vacuum_collections[0]
        assert vacuum.get_alpha() == 0.35
        np.testing.assert_array_equal(vacuum.get_array(), data["vacuum_E"])
        assert len(vacuum.get_segments()) == len(data["vacuum_E"])
        for name, expected in original_material.items():
            np.testing.assert_array_equal(data[name], expected)
    finally:
        plt.close(fig)
