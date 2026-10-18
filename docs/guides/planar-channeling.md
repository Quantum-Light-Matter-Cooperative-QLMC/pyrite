# Planar channeling runs

Planar channeling is an opt-in transport and emission model
([ADR-0018](../adr/0018-quantum-channeling-zone.md)). When it is selected, a
quantum zone owns each electron from crystal entry until it dechannels. The zone
evolves a density matrix in the transverse Bloch basis of one plane family and
then hands the electron to native Monte Carlo transport. Channeling radiation
(CR) is stored as its own `spec_channeling` component on the CXR photon grid,
beside the PXR/CBS lines, characteristic lines and bremsstrahlung, which keep
their meanings. Channeling defaults to off, and off cases, identities and outputs
are unchanged.

This page covers the run paths: the Python API, profile commands and the
packaged `channeling_demo` profile. The [channeling app](working-with-results.md#planar-channeling-demo)
and the [axial diagnostics](axial-diagnostics.md) are separate exploratory
tools.

## Prerequisites

- `pyrite tables fetch`, including SBETHE (`pyrite tables fetch sbethe`). The
  zone's electronic scattering kernel is built from SBETHE shell data. Without
  it, case construction stops with "SBETHE shell data is not installed".
- A non-mosaic, single-layer slab of a crystal with rows in
  `data/channeling_thermal.toml` (silicon, diamond, tungsten and 2H-MoS2 at
  present). The zone runs at 293.15 K. Other crystals raise
  `ChannelingDataError`.
- A beam energy at which the plane has at least one bound transverse state, and
  no more than 100 MeV. Above 30 MeV the run warns: transport after dechanneling
  has no radiative energy loss yet (#439), so physical-support claims stop
  there. A beam within three axial critical angles of a low-index axis also
  warns, because the planar zone omits axial channeling.

## The packaged demo

`channeling_demo` is one 17 MeV case through a 2 µm silicon slab, 3° from the
(110) face normal inside the (1-10) planes, with 200 fixed line and
bremsstrahlung trials. It uses the zone's built-in resolution controls. The
packaged `forward` detector puts a 3 × 3 mm ideal sensor 1 m away on the beam
axis. CR is forward-peaked, so the line energy falls steeply off axis; the 90°
`reference` detector would see lines below about 10 eV. The 1–24 keV line grid covers
Si K and the on-axis 1→0, 2→1 and 3→2 lines near 18.5, 10.2 and 7.0 keV.

```bash
pyrite tables fetch
pyrite profile show channeling_demo
pyrite run channeling_demo
pyrite app analysis launch silicon
```

The checkpoint gets a `channeling.h5` companion. In the analysis app, the
**Components** selector adds **Channeling** separately from **Characteristic**.
Packaged profiles are read-only. To edit the demo, clone it with
`pyrite profile create my_channeling --from channeling_demo`; the clone keeps
the detector, grids, physics and numerics.

## Python API

`simulate` takes the same selection through `Physics` and `Numerics`:

```python
import numpy as np

import pyrite as pr

result = pr.simulate(
    pr.Beam(energy_keV=17_000.0, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
    pr.Slab(
        "silicon",
        thickness_ang=20_000.0,
        tilt_deg=3.0,
        tilt_azim_deg=45.0,
        footprint=pr.Footprint(5.0, 5.0),
    ),
    pr.Detector(
        0.0, acceptance=pr.SensorAcceptance(distance_mm=1000.0, width_mm=3.0, height_mm=3.0)
    ),
    physics=pr.Physics(
        channeling_model="planar",
        channeling_plane=(1, -1, 0),
        radiative_model="uncoupled",
    ),
    numerics=pr.Numerics(
        cxr_trials=200,
        brem_trials=200,
        energy_bins=pr.EnergyBins(
            cxr=np.arange(1000.0, 24000.0, 50.0),
            brem=np.arange(1000.0, 17_000_000.0, 500.0),
        ),
    ),
)
cr = result.channeling  # SpectralDensity on the CXR grid; None when off
print(cr.photon_energy_eV[np.argmax(cr.density)], cr.density.max())
total = result.combined_cxr()  # includes CR; include_channeling=False drops it
```

`cr.density` is in photons / incident electron / eV / sr. It holds the bound
lines and tracked free-to-bound capture. `Numerics` defaults to midpoint
integration and `Physics` to incoherent emission, so neither is spelled out
above; `radiative_model="uncoupled"` must be. `channeling_basis_size` (default
16), `channeling_bloch_cells` (16), `channeling_dz_ang` (250 Å) and
`channeling_zone_max_ang` (`None`, the slab thickness) set the zone's plane-wave
band, Bloch supercell, depth step and length cap. Smaller values run faster;
a small basis can warn about overflow at its band edge.

## Profile workflow

To enable the zone on your own single-crystal profile:

```bash
pyrite numerics precision disable my_scan --cxr-trials 300 --brem-trials 150
pyrite physics set my_scan \
  --channeling-model planar --channeling-plane 1 -1 0 \
  --radiative-model uncoupled --emission incoherent
pyrite numerics set my_scan --energy-model midpoint
pyrite profile set my_scan --detector forward
pyrite run my_scan
```

`pyrite numerics set` also takes `--channeling-basis-size`,
`--channeling-bloch-cells`, `--channeling-dz-ang` and
`--channeling-zone-max-ang`. Enabled zone controls enter the dataset identity
and checkpoint stem, and inactive ones change nothing. Disable channeling with
`pyrite physics reset my_scan channeling-model channeling-plane`. The plane is
chosen relative to the crystal cut and the beam; the command does not change
the sample pose. [Sweep profiles](sweep-profiles.md#planar-channeling) has the
same commands in the context of the other profile controls.

## Supported and refused paths

Supported for planar cases:

- fixed-count runs, locally or through `pyrite remote`;
- scalar detectors, physical-detector directions and observations, all
  scoring the same deterministic ensemble source;
- trajectory replay and line recompute, which rescore CR on the requested line
  grid; bremsstrahlung repair keeps the stored channel.

Refused before any transport or output write:

- adaptive precision: CR is an ensemble expectation, not a per-electron
  estimate, so it has no variance observable for the stopping rule. Use fixed
  counts.
- trajectory capture and electron-block transport;
- coherent emission, temporal profiles, secondary transport, coupled radiative
  transport and frozen integration;
- stacks or other multilayer targets, mosaic crystals, grooved entrance faces
  and GDF beams.

Compute split: the zone solve, the keyed handoff sampling and the CR source run
on the CPU. Native transport after the handoff runs on the per-electron CPU
core or on CUDA (`PYRITE_MC_TRANSPORT_CORE=cuda`, or
`Numerics(transport_core="cuda")`). CUDA rows are downloaded once to join the
zone rows, so `keep_segments_on_device` does not keep channeling rows resident.
The backend does not change stream keys.

## Model limits

The CR source uses the constant mean beam energy and a laterally infinite slab
for photon attenuation. Finite-face and line-cutoff limits censor the electron
paths, but side-face photon escape, in-plane angular diffusion accumulated in
the zone and detector response convolution are not modeled. Literature peak
positions are anchored; measured linewidths are not reproduced. Dechanneling
uses the declared 300 eV margin above the planar barrier, calibrated on
Kephart et al. (1989). See [source geometry and normalization](../validation/channeling/cr-persisted-channel.md),
the [linewidth audit](../validation/channeling/cr-line-shape.md) and the
[channeling ledger](../validation/ledger-channeling.md) for each claim's
validation status.

## Diagnostic APIs versus run selection

`Physics(channeling_model="planar")`, profile physics settings and
`channeling_demo` are the only ways to put channeling into a run. The entry
always comes from the beam's plane-wave incidence and divergence. The
`pyrite.montecarlo.channeling` solvers (`DensityMatrixSolver`, `MCWFSolver`,
`channeling_spectrum`) are diagnostic APIs. Their `entry_state=` argument takes
user-defined transverse states (Gaussian, vortex, Airy and others; see
[shaped entry wavefunctions](working-with-results.md#shaped-entry-wavefunctions-experimental)), but
`simulate`, `Scene` and profiles cannot select an entry state. The axial tier
is diagnostic-only; `Physics(channeling_model="axial")` is refused.
