# TODO — feature/marimo-sweep-views

This branch carries one backlog item; the full triaged backlog lives on `main`.

## Marimo/Altair follow-up — the dense penetration grid (P3)

The other follow-ups originally scoped here are **done on `main`**: the x/y plot-limit entry
boxes now live in the tab they belong to (`detector_xmin_ui`, `polar_xmin_ui`, `narrow_xmin_ui`),
heatmap pixels are click-selectable (`plots/altair_sweeps.py::heatmap_select_chart`), and
parameter-sweep views over crystal thickness landed
(`results/selection.py::select_results` plus the per-tab thickness pins).

**What remains.** In `notebooks/analysis_app.py`, the "Dense penetration grid (datashader,
matplotlib)" accordion inside `_penetration_tab` calls

```python
plot_trajectory_grid(_traj, energy=30, Ne=120)
```

so it is hard-wired to `energy=30` and ignores `penetration_angle_ui`, even though the survival
chart and single-track view directly above it already respect that selector (and the new
`penetration_thickness_ui`). It also reportedly renders nothing when the accordion is expanded —
confirm before fixing; `trajectory_sweep(..., energies=(30, 60))` does supply a 30 keV case, so
the blank render is likely a separate bug from the hard-wired energy.

**Implementation path.** Add a beam-energy selector for the tab (or reuse an existing energy
control), pass the selected tilt through to `plot_trajectory_grid`, and verify for real with
`marimo export html notebooks/analysis_app.py`. Checkpoints exist for
`{hbn,hopg,mos2,graphite,diamond,mose2,sapphire}`. Prefer edits in `src/cxr_mc/` over notebook
logic, and keep the notebook output-free on commit.
