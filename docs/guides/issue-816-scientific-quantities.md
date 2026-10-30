# Scientific helper quantities (#816)

These adapters are **opt-in**, separate from the long-established numeric
functions. No `*_eV`, `*_ang`, `*_mm` solver entry point changes behavior and
numeric scientific contracts remain supported without deprecation.

## Constants and materials

`pyrite.materials.crystal` remains the **canonical owner** of the five scalar
constants; the numeric values, precision and reference conventions are
unchanged. `pyrite.materials.quantities.crystal_constants()` returns new Pint
quantities reconstructed from them. The constant `M_E_EV` is a *rest energy*,
not mass in kg. `E2_EV_ANG` is the Gaussian-convention Coulomb coupling
`alpha * hbar*c` (eV Å), not SI charge squared.

```python
from pyrite.units import Q_, ureg
from pyrite.materials.quantities import (
    crystal_constants,
    atomic_form_factor_q,
    reciprocal_g_vector_q,
    chi_g_q,
    linear_attenuation_q,
)

rest_energy = crystal_constants()["M_E_EV"].to("keV")
g, g_mag = reciprocal_g_vector_q((1, 1, 1), {"system": "cubic", "a": Q_(5.431, "angstrom")})
f = atomic_form_factor_q("Si", g_mag, Q_(1000, "eV"))
chi = chi_g_q("silicon", (1, 1, 1), Q_(1, "keV"), b_factor=Q_(0, "angstrom**2"))
mu = linear_attenuation_q("silicon", Q_([800, 1000, 1200], "eV"))
```

`f`, `chi`, and `mu` preserve complex values, unsupported-region NaNs, and
array shape. Atomic form factors are measured **relative to a single electron**:
Pint's `electron` is an electric charge, so the appropriate stored unit is
`dimensionless`. Composition uses number density `1/angstrom**3` (atoms are
counts, not charge). Crystal structure-factor results use the same convention.

Other adapters: `cromer_mann_f0_q`, `henke_dispersion_q`,
`structure_factor_q`, `U_g_q`, `absorption_length_q`,
`dominant_reflections_q`, and `photon_energies_q`. Scalar and array
conventions, reference-energy policy, reflection ranking and out-of-range
handling all come from the original numeric owner. Every physical argument on
these wrappers is explicitly a Pint quantity; `hkl`/element names, flags and
selectors remain ordinary numbers or strings. Direct catalog schema
constructors still take numeric payloads; use their semantic read-back
properties when inspecting them.

## Detectors and grating

```python
import numpy as np
from pyrite.detectors.quantities import (
    grating_q,
    grating_parameters,
    ccd_q,
    detected_image_q,
    convolve_detector_q,
)

energy = Q_(np.linspace(150, 300, 101), "eV")
density = Q_(np.ones(101), "1/eV")
gr = grating_q(groove_density=Q_(1200, "1/mm"), incidence_angle=Q_(87, "degree"))
ccd = ccd_q(n_pix=256, pixel_pitch=Q_(0.02, "mm"))
convolved = convolve_detector_q(energy, density, Q_(10, "eV"))
pixel_x, counts = detected_image_q(energy, density, gr, ccd, Q_(200, "mm"))
```

`convolve_detector_q` preserves the supplied spectral-density unit; with
`return_outside=True`, lost mass has the density unit multiplied by eV.
`disperse_spectrum_q` transforms density per eV into density per mm; pixel
integration in `bin_to_pixels_q` and `detected_image_q` multiplies by mm or eV,
respectively. No photon-per-electron, detector acceptance or current scaling
is inferred. These helpers require an explicitly unit-bearing `spectrum` so
no arbitrary unit is fabricated for the input spectral samples.

Additional geometry and optical adapters: `wavelength_q`, `groove_spacing_q`,
`coating_number_density_q`, `qe_absorption_q`, `charge_cloud_sigma_q`,
`energy_fwhm_q`, `diffraction_angle_q`, `angular_dispersion_q`, `grating_reflectivity_q`,
`grating_throughput_q`, `detector_position_q`, `resolving_power_q`, and
`detector_efficiency_q`. `grating_parameters` and `ccd_parameters` return
fresh read-back quantities; `Grating` and `SimpleCCD` remain numeric frozen
records. `detected_image_q(physical=True)` invokes the existing physical-CCD
model with optional thickness/voltage/absolute-temperature quantities.

## Contract

The Python numeric functions stay numeric and their floating-point work is
unchanged. Quantity adapters convert exactly once at the boundary, allocate
fresh arrays for results, and use the shared `pyrite.units` registry. They do
not store Pint values in catalog hashes, response buffers or simulation case
records. This is an intentional *wrapper interface*: no overloaded return
behavior that depends on whether a caller accidentally supplied a quantity.
