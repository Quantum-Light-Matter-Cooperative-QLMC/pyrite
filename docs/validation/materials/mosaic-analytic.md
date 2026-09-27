# `mosaic-analytic`

## Claim and source

- Claim: `montecarlo/detector.py::mosaic_fwhm_eV` returns the first-order Gaussian energy broadening `FWHM = E * abs(tan(psi)) * eta`.
- Source: the resonance relation and kinematical mosaic-block derivation in `docs/physics/materials/crystal-mosaicity.md`.
- Signature: `mosaic_fwhm_eV(E_eV, psi_rad, mosaic_fwhm_rad)`; both angles are radians and the result is eV.

## Independent derivation

At fixed electron velocity and detector direction, rotating the reciprocal vector changes only the numerator of

$$
E=C\,\mathbf v\cdot\mathbf g=Cvg\cos\psi,
\qquad C=\frac{\hbar c}{1-\mathbf v\cdot\hat{\mathbf n}}.
$$

For a small longitudinal mosaic rotation `dpsi`,

$$
\frac{dE}{E}=d\ln(\cos\psi)=-\tan\psi\,d\psi.
$$

A linear scaling of a Gaussian scales its FWHM by the absolute scale factor. If the rocking-curve angular FWHM is `eta`, the energy FWHM is therefore

$$
\Delta E_{FWHM}=E\,|\tan\psi|\,\eta.
$$

No `2 sqrt(2 ln 2)` conversion appears because input and output widths are both FWHM, not standard deviations.

## Cheap filters

- Units: eV times two dimensionless quantities gives eV.
- Limits: `eta -> 0`, `E -> 0`, or `psi -> 0` gives zero broadening. The linearized model diverges as `psi -> pi/2`, correctly exposing its grazing breakdown rather than defining a finite physical prediction there.
- Signs/conventions: broadening is nonnegative and even under `psi -> -psi`; the derivative's sign shifts the line centroid for one tilt direction but cannot affect a symmetric Gaussian width.

## Implementation comparison

Production returns exactly

```text
E_eV * abs(tan(psi_rad)) * mosaic_fwhm_rad
```

so the symbolic comparison is exact. At `E=1500 eV`, `psi=35 degrees`, and `eta=0.4 degrees`, production gives `7.332556193426062 eV`. Mapping the two angular half-width endpoints through the unlinearized cosine resonance gives `7.332541302606146 eV`, a relative linearization error of `2.03e-6`. Existing tests pin zero width at `psi=0`, linearity in spread, and the same analytic expression; `checks/mosaic_mc_check.py` compares it with the exact orientation average in the small-spread regime.

Follow-up applied after independent verification: the function docstring now contains `Validation: mosaic-analytic`, and its stale "future work" wording now describes the implemented Gauss--Hermite route.

## Verdict

`rederived`. Units, limits, width convention, symmetry, exact prefactor, and a numeric small-angle comparison match. Ledger status advanced from `unverified` to `rederived` after verification.
