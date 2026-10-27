# Radiation sources

Reviewed spectral source routines and experimental directional-source operations. See the [directional contract](../directional-sources.md).

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.radiation.mc_cxr_spectrum
   pyrite.radiation.mc_cxr_spectrum_solid_angle
   pyrite.radiation.mc_brem_spectrum
   pyrite.radiation.directional_source
   pyrite.radiation.directional_sources
```

## Photon-energy comparisons

{py:func}`pyrite.photon_energies` computes vacuum CXR line-centre energies
without transport. Beam and photon energies are in eV; angles are in degrees.
Broadcast mode pairs inputs; grid mode forms their Cartesian product. These
are kinematic energies, with no prediction of intensity or broadening. See
[comparison examples and reflection rules](../../reference/python-scene-recipes.md#compare-photon-energies).

## Provisional directional source API

`pyrite.radiation.directional_source(beam, target, energy_eV=..., angle_deg=..., ...)`
returns `pyrite.results.directional.DirectionalSpectrum` without a detector or checkpoint. The result
contains explicit coordinates, escaped incoherent PXR/CBS density per incident
electron/eV/sr, supported-direction mask and resolved provenance. `.rate()` uses
the beam mean current; `.rate(current_na)` rescales stored arrays to ph/s/eV/sr.
`pyrite.radiation.directional_sources(..., components=...)` returns one such result per
component (`"pxr-cbs"`, `"characteristic"`, `"bremsstrahlung"`, `"transition"`)
from one transport. The opt-in `"transition"` component uses the production
history-coherent planar-crossing model; it requires `footprint=None` and the
supported X-ray dielectric band. Cherenkov remains an internal bulk diagnostic
in the app, outside this escaped-source API.
This surface is provisional; see [directional heatmaps](../../guides/directional-source-heatmap.md)
for a runnable visual example and the [directional source reference](../../reference/directional-sources.md)
for the full-circle convention, supported geometry and limits.

## Source contracts and migration

Radiation energy grids are in **eV**; low-level segment electron energies are
in **keV**. Directions use the frame and angle unit specified by each function.

| Canonical import (`pyrite.radiation.`) | Status | Previous import | Input/output and effects |
| --- | --- | --- | --- |
| `photon_energies` | Advanced-stable | `pyrite.photon_energies` (retained convenience), `pyrite.materials.emission.photon_energies` | Catalog materials, electron energy (eV), geometry angles (degrees); returns vacuum line centres (eV), with broadcast or Cartesian axes. Reads catalog when called; no transport/yield/provenance result. Invalid values, unresolved cuts or catalog keys raise. |
| `mc_cxr_spectrum` | Advanced-stable | `pyrite.montecarlo.spectrum.mc_cxr_spectrum` | Segments, crystal/reflections, photon grid (eV), direction/angles (rad); escaped CXR density in ph/incident-electron/eV/sr, optionally separate PXR/CBS components. Uses stored incident population or explicit electron limit, with existing coherent population controls; reads material response and may use accelerator arrays. Unsupported coherence/geometry/quadrature combinations raise. |
| `mc_cxr_spectrum_solid_angle` | Advanced-stable | `pyrite.montecarlo.spectrum.mc_cxr_spectrum_solid_angle` | Same source inputs plus directions and weights (sr); returns ph/incident-electron/eV, already integrated over solid angle. Shape mismatch or tiled grooves raise. Never multiply by solid angle again. |
| `mc_brem_spectrum` | Advanced-stable | `pyrite.montecarlo.spectrum.mc_brem_spectrum` | Segments, photon grid (eV), composition (atoms/angstrom³), direction/angles (rad); escaped incoherent density in ph/incident-electron/eV/sr. Existing cutoff/population and cross-section selection remain in force; reads cross-section/material tables as needed; invalid models/inputs or missing required coverage raise. |
| `directional_source`, `directional_sources` | Experimental | `pyrite.directional.directional_source`, `pyrite.directional.directional_sources`; deprecated root aliases | Public scalar beam and flat target, explicit grid (eV), periodic angle points (degrees), fixed sampling and seed. One shared transport returns one `DirectionalSpectrum` per component with ph/incident-electron/eV/sr, case, provenance and support mask; unsupported infinite-slab grazing points are masked NaN. Invalid grids/angles, adaptive precision and grooves raise. |

The low-level spectrum arrays carry no new provenance wrapper: retain their
input segments, table/model choices and normalization metadata. Directional
results retain their existing case/provenance and current-scaled `rate()`;
`DirectionalSpectrum` remains an experimental **results** type at
`pyrite.results.directional`. No additional result class is introduced here.
These escaped densities include source self-absorption; they are distinct from
transported photon events, aperture tallies and acquisition counts. Neither
facade applies detector efficiency, electronic response or exposure.

Previous implementation imports remain supported compatibility paths. Only the
two root directional aliases enter the existing versioned deprecation registry;
use their canonical radiation imports. Future kernel exports require explicit
review and manifest entries; `montecarlo.__all__` is frozen by its export guard.
