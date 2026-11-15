# Scientific calculators

Use `pyrite.tools` for small deterministic calculations. All entries are
advanced-stable aliases of existing functions; their scientific domains retain
ownership of the implementations and validation. Signatures and detailed units
come from the generated docstrings below.

```python
from pyrite import tools
from pyrite.units import Q_

wavelength = tools.wavelength_q(Q_(8, "keV"))
```

Importing `tools` loads no numerical engines, plotting or material datasets.
Accessing a function loads its existing implementation; evaluating material
calculators may read bundled reference tables. This facade excludes simulation,
random sampling, artifact/checkpoint I/O and dataset fetching. Existing imports,
including the root `photon_energies` convenience alias, keep working. See
[compatibility](compatibility.md#scientific-calculator-imports).

## Numeric and quantity pairs

Numeric functions retain documented raw magnitudes, even when a name has no unit
suffix. The explicit `*_q` adapters require Pint quantities for physical inputs
and return quantities; dimensionless selectors such as Miller indices remain
numeric. They do not change numeric records or scientific identity. See
[units and quantities](units.md) for the shared registry and input conventions.

| Scientific owner | Numeric function in `tools` | Quantity alternative in `tools` | Units / result |
| --- | --- | --- | --- |
| Materials | `cromer_mann_f0` | `cromer_mann_f0_q` | Reciprocal magnitude Å⁻¹; electron-relative dimensionless amplitude |
| Materials | `henke_dispersion`, `atomic_form_factor` | `henke_dispersion_q`, `atomic_form_factor_q` | Photon energy eV; electron-relative dimensionless amplitudes |
| Materials | `reciprocal_g_vector` | `reciprocal_g_vector_q` | Lattice lengths Å, angles degrees; vector and magnitude Å⁻¹ |
| Materials | `structure_factor` | `structure_factor_q` | Energy eV, B in Å²; amplitude and reciprocal magnitude Å⁻¹ |
| Materials | `chi_g`, `U_g` | `chi_g_q`, `U_g_q` | Energy eV, B in Å²; susceptibility dimensionless, potential eV |
| Materials | `absorption_length_ang` | `absorption_length_q` | Energy eV, density Å⁻³; intensity length Å |
| Materials | `dominant_reflections` | `dominant_reflections_q` | Reflection records with documented amplitudes and reciprocal magnitudes |
| Materials | `linear_attenuation_inv_mm` | `linear_attenuation_q` | Energy eV; attenuation mm⁻¹ |
| Radiation | `photon_energies` | `photon_energies_q` | Beam energy keV, geometry in documented units; line energies eV |
| Detectors | `detector_efficiency` | `detector_efficiency_q` | Energy eV; dimensionless efficiency |
| Detectors | `convolve_detector` | `convolve_detector_q` | Energy and FWHM eV; caller spectral normalization preserved |
| Transport | `depth_loss_profile` | — | In-memory transport block, depth edges Å; loss table in keV |

`crystal_constants` returns fresh quantities for the named material constants.
Grating and CCD quantity calculators include explicit geometry constructors
`grating_q`, `ccd_q`, their read-back helpers `grating_parameters`,
`ccd_parameters`, wavelength and spacing conversions, dispersion, throughput,
pixel integration and sensor response. Their raw counterparts remain in the
[detector domain](detectors.md). The generated entries specify each result's
units, including mm positions, angular dispersion, and integrated pixel values.

## Calculator reference

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.tools.U_g
   pyrite.tools.U_g_q
   pyrite.tools.absorption_length_ang
   pyrite.tools.absorption_length_q
   pyrite.tools.angular_dispersion_q
   pyrite.tools.atomic_form_factor
   pyrite.tools.atomic_form_factor_q
   pyrite.tools.bin_to_pixels_q
   pyrite.tools.ccd_parameters
   pyrite.tools.ccd_q
   pyrite.tools.charge_cloud_sigma_q
   pyrite.tools.chi_g
   pyrite.tools.chi_g_q
   pyrite.tools.coating_number_density_q
   pyrite.tools.convolve_detector
   pyrite.tools.convolve_detector_q
   pyrite.tools.cromer_mann_f0
   pyrite.tools.cromer_mann_f0_q
   pyrite.tools.crystal_constants
   pyrite.tools.depth_loss_profile
   pyrite.tools.detected_image_q
   pyrite.tools.detector_efficiency
   pyrite.tools.detector_efficiency_q
   pyrite.tools.detector_position_q
   pyrite.tools.diffraction_angle_q
   pyrite.tools.disperse_spectrum_q
   pyrite.tools.dominant_reflections
   pyrite.tools.dominant_reflections_q
   pyrite.tools.energy_fwhm_q
   pyrite.tools.grating_parameters
   pyrite.tools.grating_q
   pyrite.tools.grating_reflectivity_q
   pyrite.tools.grating_throughput_q
   pyrite.tools.groove_spacing_q
   pyrite.tools.henke_dispersion
   pyrite.tools.henke_dispersion_q
   pyrite.tools.linear_attenuation_inv_mm
   pyrite.tools.linear_attenuation_q
   pyrite.tools.photon_energies
   pyrite.tools.photon_energies_q
   pyrite.tools.qe_absorption_q
   pyrite.tools.reciprocal_g_vector
   pyrite.tools.reciprocal_g_vector_q
   pyrite.tools.resolving_power_q
   pyrite.tools.structure_factor
   pyrite.tools.structure_factor_q
   pyrite.tools.wavelength_q
```

## Related saved-capture diagnostics

These deterministic reductions read saved HDF5 artifacts, so they stay with their
own domains. They run no transport. Use explicit bin edges and bounds as
specified in their docstrings.

| Scientific owner | Canonical import | Result units |
| --- | --- | --- |
| Transport | `pyrite.transport.summarize_trajectories` | Fixed-bin tables: depth/range Å, energy keV, angle rad, counts/fractions |
| Transport | `pyrite.transport.iter_primary_summaries` | Per-primary summaries in the documented trajectory units |
| Transport | `pyrite.transport.capture_secondary_energy_balance` | Ledger table with `energy_keV` |
| Beams | `pyrite.beams.capture_beam_metrics` | Beam metrics plus sampled/available particle counts |

See [electron transport](transport.md) and [beam diagnostics](beams.md#advanced-saved-diagnostics)
for generated contracts. Simulation spectrum functions and directional-source
builders remain under [radiation](radiation.md); `fetch_tables` remains a dataset
operation. They are outside the calculator membership rule.
