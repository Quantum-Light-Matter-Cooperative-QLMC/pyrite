# Detector-free directional source heatmaps

```bash
pyrite app angular-heatmap launch
pyrite app angular-heatmap launch --smoke
pyrite app angular-heatmap export
```

The standalone app opens on 30 keV point electrons through a 2000 Å silicon
film with a finite 5 × 5 mm footprint, target tilts 30° and 45°, 128 incident
histories per tilt, 72 periodic observation angles and 1025 photon energies from
100 eV up to the kinematic line ceiling (2000 eV here). It needs no detector,
campaign or checkpoint. The 5 × 5 mm footprint is the project's historical
bounded crystal (`campaign.geometry.DEFAULT_FOOTPRINT`); set the width and
height to your sample.

Controls are grouped by role: **Beam**, **Target**, **Footprint**,
**Observation**, **Photon energy** and **Execution**. Submit **Run source maps**
to calculate a scene. The **Display** row (mean current, log color, tilts shown)
only redraws cached source arrays. A status rail reports the backend, elapsed
time, footprint, resolved photon grid, line ceiling and history count. Export
writes the default example into static HTML; static HTML cannot run new
transports. The four-entry in-memory calculation cache is local to the running
app process.

Only the existing **incoherent PXR/CBS total** is scored, including its PXR–CBS
interference. Bremsstrahlung, characteristic, cross-electron coherent,
channeling and transition radiation are excluded from the map. Electron
transport retains the existing interaction policy. Photons retain target
escape/self-absorption; there is no downstream filter, aperture, response,
acquisition, or angular acceptance integration.

## Coordinates and units

The nominal incoming beam defines lab +z. For observation-plane azimuth φ,
its transverse basis is `(cos φ, sin φ, 0)` and the direction at signed angle θ
is `(sin θ cos φ, sin θ sin φ, cos θ)`. Default φ = 0 gives the x–z plane:
0° is forward, +90° is +x, −90° is −x and ±180° is backward. The target pose
rotates these lab vectors into the sample frame using the existing target
convention; target angles resolve to the nearest half degree. A map is a
**plane slice**, not an azimuth-integrated distribution.

Angles are periodic point samples on [−180°, +180°), without a repeated endpoint.
The plot closes the circle by duplicating the first sample for drawing only.
Side-by-side tilts share both axes and one color scale: only the leftmost panel
carries the energy label and a single colorbar carries the intensity label.

The reusable result stores photons per incident electron per eV per sr. Its
`rate(current_na)` scales by the incident electron rate `I[nA] * 1e-9 / |e|`,
yielding **ph/s/eV/sr** with no solid-angle factor. Omitting current uses the
beam mean current: `Q[pC] * f[Hz] / 1000` nA. The initial beam has 1 pC at 5 kHz,
hence 5 nA. Supported bins vanish at zero current. This conversion assumes
independent incident electrons and applies to this incoherent component only.
See [spectral observables](../physics/radiation-physics/spectral-observables.md)
and the independently verified
[normalization claim](../validation/radiation-physics/directional-source-rate-normalization.md).

## Finite-target escape

Each emitting segment is attenuated along the exact ray to its first crystal
face, side faces of the finite footprint included, with the segment-mean
transmission split at face switches and layer interfaces. This is the existing
line scorer's escape (Validation: `finite-transverse-crystal`,
`segment-escape-average`); the directional source reuses it unchanged.

With a finite footprint every direction is evaluated. Directions parallel to the
slab faces (θ = tilt ± 90° at zero tilt and plane azimuths) leave through a
millimetre side face, so for keV photons they are a deep, **physical**
attenuation minimum, not a masked gap. Shrinking the footprint to 10 µm lifts
those directions by orders of magnitude; away from them, a 5 mm footprint agrees
with the infinite slab to machine precision. The **Infinite slab** option keeps
that limiting case for comparison. Only there are face-parallel directions
(`abs(n_sample,z) <= 1e-10`) unsupported, NaN and drawn gray. Zero or
underflowed intensity takes the lowest color, so gray always means unsupported.
Regression: `tests/montecarlo/test_directional_source.py::test_finite_target_escape_limits`.

## Photon window and relativistic forward lines

The line resonance `E = ħc v·g / (1 − n·v)` Doppler-shifts toward the beam
direction. At 1 MeV (β = 0.941) the dominant silicon line at 45° tilt rises from
2.35 keV at 30° to 7.3 keV at 0°, matching the factor
`(1 − β cos 30°)/(1 − β) = 3.1`. With the former fixed 5 keV stop, the
near-forward band therefore looked like a deep null that was absent at 30 keV
(β = 0.33, all lines below about 1 keV). It is window truncation: on a
100 eV–63 keV grid the 0° integrated yield is among the largest on the circle.

The app therefore stops the grid at the scene's **kinematic line ceiling**
`ħcβ|g|max/(1 − β)` over the scored reflections, capped by the beam kinetic
energy (Validation: `line-grid-kinematic-bandwidth`,
`line-grid-emission-ceiling`). A manual stop below the ceiling raises a warning.
Use logarithmic energy spacing for MeV beams, where the window spans decades.
The API reports the ceiling as `provenance["line_window_ceiling_eV"]`, and
`pyrite.directional.source_line_ceiling_eV(beam, target, numerics=...)` returns
it before any transport. Regression:
`tests/montecarlo/test_directional_source.py::test_relativistic_forward_band_is_window_truncation`.

## Reusable source calculation

```python
import numpy as np
from pyrite import Beam, Footprint, PointSpot, PointBunch, Slab, Numerics, directional_source
from pyrite.directional import source_line_ceiling_eV

beam = Beam(
    30.0, transverse=PointSpot(), longitudinal=PointBunch(), bunch_charge_pc=1.0, rep_rate_hz=5000.0
)
target = Slab("silicon", 2000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0))
numerics = Numerics(cxr_trials=128)
stop = source_line_ceiling_eV(beam, target, numerics=numerics)
source = directional_source(
    beam,
    target,
    angle_deg=np.linspace(-180, 180, 72, endpoint=False),
    energy_eV=np.linspace(100, stop, 1025),
    numerics=numerics,
    seed=1,
    direction_batch=16,
)
rate = source.rate()  # shape (angle, energy), ph/s/eV/sr
rate_at_10_na = source.rate(10.0)  # rescales without transport
```

The provisional API accepts flat slabs and stacks, finite or infinite.
Grooves, direction-targeted longitudinal policies and adaptive precision are
unsupported and rejected. Existing target geometry restrictions still apply. It
transports once per scalar target/tilt, then evaluates bounded direction batches
on the same segments. `brem_trials` is replaced by `cxr_trials` to avoid an
unnecessary second population; only the CXR adapter is evaluated. Result
metadata records the case, seed, resolved numerics, plane azimuth, energy-grid
choice, line ceiling, package/model/table provenance and source identity.
Component identity and independent energy coordinates leave room for other
future source components without assuming additive semantics.

## Resolution, statistics and limits

The grid is an explicit common photon-energy **point grid** for the whole
circle. It must increase, start at or above 10 eV, and end at or below the
incident electron kinetic energy. A narrow single-direction automatic line
window is never substituted.

A 2000 Å film resolves much narrower lines than a thin film. For the default 30°
map (seed 1), doubling the energy points changes the energy-integral L1 by
3.6% from 257→513, 0.55% from 513→1025 and 0.017% from 1025→2049; the peak is
unchanged from 1025. Hence the 1025-point default. Statistics, not grids, then
dominate: seed-to-seed spread of the angular yield profile is 58% at 8
histories, about 8% at 128 and 2% at 512; per-pixel spread falls as 1/√N
(0.39 at 128). Raise histories before reading fine structure. Regression:
`tests/notebooks/test_angular_heatmap_app.py::test_default_grid_refinement_and_history_normalization`.
The finite plotted window is not a total yield.

The app has no demonstration caps on tilts, thickness, histories or grid size.
It validates physical inputs, keeps tilts distinct after half-degree pose
resolution, and refuses only result arrays larger than about 1.6 GB; the
spinner reports the electron × direction × energy work before it starts.
*Max reflections* (default 4) truncates the reflection set; raise it for a
reflection-convergence study. The existing kinematic PXR/CBS and escape model
validity limits apply unchanged.

## Execution backends

The array backend is fixed when PyRITE is imported, so choose it at launch:
`PYRITE_MC_BACKEND=cuda pyrite app angular-heatmap launch` runs transport and
scoring on a local GPU, and the app's status rail shows the resolved backend.
*Transport core* selects the CPU lockstep/per-electron or CUDA transport core;
*Direction batch* bounds scoring scratch per batch without changing
normalization. For sweeps beyond an interactive session, script the
`directional_source` API and run it on the GPU host through `pyrite remote`.
