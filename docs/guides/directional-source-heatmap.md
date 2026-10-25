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
**Radiation**, **Photon energy** and **Execution**. The material list is the
full catalog. *Tilts* takes a comma list and/or inclusive `start:stop:step`
ranges (`5:80:5, 45`); target azimuth accepts the same syntax. Each polar/azimuth
pair is one transport. Submit **Run source maps**
to calculate a scene. The **Display** controls (component, mean current, log
color and range, orientations compared side by side or browsed with polar and
azimuth sliders,
cut energy/angle) only redraw cached source arrays. Maps are interactive Plotly
heatmaps whose hover shows the true rate; below them, an angular cut at the
chosen energy and a spectrum at the chosen angle overlay every shown tilt. A
status rail reports the backend, transport core, elapsed time (or `cached`),
footprint, resolved photon grid, line ceiling and history count. Export
writes the default example into static HTML; static HTML cannot run new
transports. The four-entry in-memory calculation cache is local to the running
app process.

Selected **components** are scored from the same transport and kept as
separate maps, never summed: the **incoherent PXR/CBS total** (including its
PXR–CBS interference), **characteristic** atomic lines (isotropic emission, so
the map shows escape geometry) and the **bremsstrahlung continuum** (BremsLib
angular model when its tables are installed, otherwise isotropic EEDL).
**TR** adds the production history-coherent planar-crossing source. Select
**Infinite slab** explicitly before requesting it: the app refuses finite
footprints rather than substituting a planar geometry. TR also requires the
existing per-layer X-ray dielectric band and uses the same transported continuum
population. Its automatic photon stop is the beam energy, rather than the
PXR/CBS line ceiling; refine a suitable explicit band for the intended study.
Cross-electron coherent and channeling radiation are excluded. Electron
transport retains the existing interaction policy. Photons retain target
escape/self-absorption; there is no downstream filter, aperture, response,
acquisition, or angular acceptance integration.

Choose **Cherenkov (internal bulk)** in the radiation-view selector for a separate
angle × photon-energy map from the existing homogeneous Frank–Tamm source.
Declare a constant refractive index, beam kinetic energy, straight path length
and photon band. The band must be transparent with nearly constant index, and
photon energy must remain far below the electron kinetic energy. The app bins
the cone over full-azimuth polar bins about the velocity; values are
**photons/electron/eV/angular bin**, integrated over the declared path, rather
than pointwise per-steradian densities. Below threshold the yield is zero.
Target pose, material selection and footprint from the transported view do not
affect this homogeneous source. No interfaces, refraction, TIR, absorption or
finite-target escape are included. It is never added to the TR map.

Cherenkov download includes photon energies, polar-bin centres and edges,
per-bin/path yields, per-metre yields, threshold mask, declared dielectric and
source settings. Transported downloads include the resolved polar and azimuth
angles for every orientation, selected components and provenance. Changing a
cached display selection performs no transport; computing new target
orientations requires submitting the source form.

## Coordinates and units

The nominal incoming beam defines lab +z. For observation-plane azimuth φ,
its transverse basis is `(cos φ, sin φ, 0)` and the direction at signed angle θ
is `(sin θ cos φ, sin θ sin φ, cos θ)`. Default φ = 0 gives the x–z plane:
0° is forward, +90° is +x, −90° is −x and ±180° is backward. The target pose
rotates these lab vectors into the sample frame using the existing target
convention; target angles retain full precision. A map is a
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
unnecessary second population; only the requested adapters are evaluated.
`pyrite.directional_sources(..., components=("pxr-cbs", "characteristic",
"bremsstrahlung"))` returns one `DirectionalSpectrum` per component from one
transport; characteristic and continuum use the live runner's lower-cutoff
population, as in production. Pick a manual photon stop for the continuum,
which extends to the beam energy rather than the PXR/CBS line ceiling. Result
metadata records the case, seed, resolved numerics, plane azimuth, energy-grid
choice, line ceiling, package/model/table provenance and source identity.
Each result's `component` names its identity; components are not additive
by assumption.

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
It validates physical inputs, requires distinct target tilts, and refuses only
result arrays larger than about 1.6 GB; the spinner reports the electron × direction × energy work before it starts.
*Max reflections* (default 4) truncates the reflection set; raise it for a
reflection-convergence study. The existing kinematic PXR/CBS and escape model
validity limits apply unchanged.

## Run on molab

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/src/pyrite/apps/angular_heatmap_app.py)

[molab](https://docs.marimo.io/guides/molab/) opens the app straight from
GitHub and can attach an NVIDIA GPU. The notebook's inline script metadata
(PEP 723) declares Python 3.13+, `pyrite-mc` from `main` (until a release
contains `directional_source`), Linux CuPy for CUDA 13, marimo and Matplotlib.

1. Open the badge link (or paste the file's GitHub URL into molab) and turn on
   the GPU in the notebook specs **before** the first run: PyRITE selects its
   array backend at import, so a later toggle needs a kernel restart.
2. If molab does not install the declared packages itself, install
   `pyrite-mc @ git+https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite@main`
   (or the branch under review) and `cupy-cuda13x[ctk]` explicitly. Do not let
   molab guess a package for `import pyrite`: the unrelated PyPI project
   `pyrite` is not PyRITE.
3. Run all cells. The first cell calls `pyrite.fetch_tables()`, which downloads
   the pinned data (about 60 MB, public GitHub Releases) into a fresh session
   and returns immediately when it is already complete.
4. Check the status rail: `Backend` should read `cuda`. *Share → Run as app*
   hides the code.

molab keeps no generated files, and sessions stop after 90 idle minutes or
12 hours. Use **Download results (.npz)** for maps worth keeping: it contains
per-tilt, per-component `t<i>_<component>_density` (per incident electron) and
`t<i>_<component>_rate` (at the displayed current), per-tilt `supported`, lab/sample directions, the shared angle/energy/tilt coordinates,
and JSON `metadata` (submitted config, current, units, line ceilings,
identity digests). Static molab previews show no outputs; use a live session.

## Execution backends

The array backend is fixed when PyRITE is imported, so choose it at launch:
`pyrite app angular-heatmap launch --backend cuda` (equivalently
`PYRITE_MC_BACKEND=cuda`) runs transport and scoring on a local GPU, and the
app's status rail shows the resolved backend. *Transport core* offers the CPU
lockstep/per-electron cores, plus the CUDA core only on a CUDA launch;
*Direction batch* bounds scoring scratch per batch without changing
normalization. For sweeps beyond an interactive session, script the
`directional_source` API and run it on the GPU host through `pyrite remote`.
