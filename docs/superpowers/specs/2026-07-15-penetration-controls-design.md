# Penetration control sources

The Penetration tab will let the user choose each transport parameter from the
active material's configured scan grid or enter a manual value.  This applies
to beam energy, crystal thickness, and polar tilt.

## Controls

Each parameter has a source selector and one corresponding value control:

- **Material scan grid** presents the active material's catalog values and
  defaults to the first configured value.
- **Manual value** presents a bounded numeric input.  Beam energy is 1--300
  keV, thickness is greater than zero and at most 10 mm, and polar tilt is
  0--89.9 degrees.

Manual beam energy above 30 keV is labelled exploratory because the transport
free-path fit is documented through 30 keV, although the Mott angle tables
cover the complete 1--300 keV control range.

## Data flow and failure handling

The resolved values drive both the ordinary penetration cases and the lazy
dense-grid cases.  The tab keeps the existing stack-aware `trajectory_sweep`
and direct CPU transport path.  Bounds are supplied by Marimo's numeric
controls; no invalid value reaches `trajectory_sweep`.

## Verification

Static AST/source regression tests will prove that the app reads the active
material's `scan.energy_keV`, `scan.thickness_ang`, and `scan.tilt_deg`, exposes
the three source controls and bounded manual inputs, and threads the resolved
values into `trajectory_sweep`.  `marimo check notebooks/analysis_app.py` is
the structural gate.
