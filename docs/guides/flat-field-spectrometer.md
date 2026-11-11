# Model a private flat-field spectrometer

Forward-model a scored spectrum through an entrance slit, a flat-field grating
and a greateyes ALEX-s 2k512 CCD, keeping restricted vendor data in a private
catalog layer. The model is basic: an analytic slit kernel and a flat focal
plane, with no ray tracing. Its scope and limits are in the
[grazing-incidence grating note](../research/instrumentation/grazing-grating.md).

Every number below is illustrative. Put your vendor's values only in your
private layer.

## 1. Put the grating and its curves in the private layer

Set up the user layer as described in
[private instrument catalogs](../reference/catalogs-and-workspaces.md#private-instrument-catalogs),
then add the grating record with its mount at the two band ends:

```toml
# instrument-catalog/gratings/lab-ff.toml
label = "Lab flat-field grating"
groove_density_per_mm = 1000
order = 1
order_convention = "sum"        # or "difference"; PyRITE maps it to its own sign
efficiency_curve = "lab-ff-efficiency"

[mount]
r_mm = 300.0                    # slit to grating
alpha_deg = 86.0
band_nm = [6.0, 24.0]
beta_deg = [-82.55, -76.80]     # diffraction angles at the band ends
beta_H_deg = -86.5              # focal-plane normal
r_H_prime_mm = 280.0            # grating to focal plane, along that normal
focal_plane_length_mm = 28.56
```

The efficiency curve is the vendor's **absolute** efficiency, coating included,
in `curves/lab-ff-efficiency.csv` or `.toml` with `quantity = "efficiency"`
and `confidential = true`. The file format is in the
[schema reference](../repo-design/materials-catalog-schema.md#instrument-curves-and-gratings).

## 2. Check the mount against its table

```python
from pyrite.detectors.grating import Grating

grating = Grating.from_catalog("lab-ff")
print(grating.mount_residuals())
```

`beta_error_deg` compares the grating equation with the tabulated band-end
angles; `focal_plane_length_rel_error` compares the computed band-end
separation on the focal plane with the tabulated length. Small residuals
(below about 0.05° and 1 %) confirm the order sign, angle convention and
focal-plane geometry. To keep this check in your tests without committing
data, run PyRITE's own check with
`PYRITE_INTERNAL_USER_CATALOG=<layer> PYRITE_INTERNAL_GRATING=lab-ff`.

## 3. Form the ALEX-s image

`photons_per_eV` is the photon spectrum entering the slit during the exposure:
a scored source spectrum multiplied by the slit's accepted solid angle, the
beam rate and the exposure time.

```python
import numpy as np

from pyrite.detectors import AlexS
from pyrite.detectors.grating_image import (
    AlongSlitSpread,
    SlitImage,
    alexs_spectrometer_image,
)

response = AlexS(readout_rate_hz=1.0e6, gain_mode="standard")  # QE fallback
slit = SlitImage(width_mm=0.02, resolving_power=500.0)
astigmatism = AlongSlitSpread(energy_eV=(60.0, 200.0), height_mm=(15.0, 25.0))

frame = alexs_spectrometer_image(
    energy_eV,
    photons_per_eV,
    grating,
    response,
    slit,
    exposure_s=10.0,
    along_slit=astigmatism,
    rng=np.random.default_rng(1),
    shot_noise=True,
    read_noise=True,
)
frame.adu  # (515, 2048) ADU
frame.row_summed()  # 1D dispersion-axis view
```

Take the astigmatic height `H(λ)` from the vendor's spot diagrams, or pass
`AlongSlitSpread(capture_fraction=...)` when the vendor tabulates the captured
fraction. Energies outside the efficiency curve or the spread table are not
delivered: tabulated data are never extrapolated.

Without a digitized QE curve, `AlexS` uses the Beer-Lambert model with the
datasheet 15 µm epitaxial layer, an approximation. Once the curve is in the
layer, reference it from the detector record (`response.qe_curve`) and lower
the detector with `physical_detector_from_config(row, curves=catalog.curves)`,
or pass `AlexS(qe=TabulatedEfficiency.from_catalog("lab-ccd-qe"))`.

## 4. View the source directly

Without a grating, the ALEX-s can score the source through the pixelated
detector path:

```python
from pyrite.detectors import AlexS, PlanarDetector, PlanarPose

pose = PlanarPose.from_observation(500.0, 90.0, 0.0, 0.0, (0.0, 0.0))
detector = PlanarDetector.alexs_2k512(pose, response=AlexS())
```

`SpatialResult.image(..., measured=True)` then returns noise-free ADU per
incident electron. The ALEX-s integrates charge, so it cannot drive a
photon-counting `[acquisition]`; frame noise comes from
`pyrite.detectors.alexs_response.alexs_frame`.
