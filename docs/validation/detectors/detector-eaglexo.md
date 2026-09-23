# Validation: detector-eaglexo

## Claim

- **ID:** `detector-eaglexo`
- **Symbol:** `detectors/eaglexo_response.py::EagleResponse`
- **Source:** rectangular-detector solid angle and measured Eagle XO quantum efficiency.
- **Intended quantity:** expected detected photon spectrum from incident spectral-angular photon yield.

## Independent derivation

For a centered rectangle of half-width `a`, half-height `b`, normal to the source direction at distance `d`, direct integration of `dOmega = cos(theta) dA / r^2 = d dA / r^3` gives

```text
Omega = 4 atan[a b / (d sqrt(d^2 + a^2 + b^2))].
```

For `a,b << d`, `Omega -> (2a)(2b)/d^2 = A/d^2`.  If `S(E) = d^2N/(dE dOmega)` is approximately constant over the aperture and `QE(E)` is the probability that an incident photon is detected, then

```text
dN_det/dE = S(E) Omega QE(E).
```

If an upstream stage already integrated the angular yield over `Omega`, the detector response itself must apply only `QE(E)`; multiplying by `Omega` again would double-count acceptance.  Optional energy redistribution must conserve the detected line integral apart from explicit finite-grid edge loss.

Units: steradian is dimensionless in SI; `QE` is dimensionless and bounded in `[0,1]`; detected and incident spectra share energy-bin units after the stated solid-angle convention.  Limits: zero area or `QE=0` gives zero detections; `QE=1` gives pure geometric acceptance; far distance gives inverse-square scaling; elementwise QE multiplication cannot create negative counts.

The manufacturer's tabulated QE values and any extrapolation beyond their range are empirical subclaims.  Formula validation can check interpolation bounds, endpoint continuity, and stated tail behavior, but cannot certify the measurements without the source datasheet.

## Implementation comparison

`solid_angle_sr` matches the centered-rectangle expression exactly. `results.store_result` owns angular acceptance through `scale = domega_sr * PER_NA`; `EagleResponse.apply` then multiplies that already-scaled incident spectrum by `QE(E)` only.  Thus the full pipeline is `S(E) Omega QE(E)` without double-counting `Omega`.  The QE implementation uses log-energy interpolation, a continuous endpoint-anchored power-law tail, and a final `[0,1]` clamp.

Independent numeric points: 100-point-per-axis Gauss--Legendre integration of `dOmega` for a `27.6 mm` square at `400 mm` agreed with `solid_angle_sr` to relative `5.5e-16`; the exact result differed from `A/d^2` by `1.19e-3`, as expected.  This is `0.119%`, so the `solid_angle_sr` docstring's illustrative claim that the difference is `<0.1%` is slightly too tight; the implemented exact formula is unaffected.  QE over zero, every table energy, and `1 MeV` stayed in `[0,0.9562]`; the extrapolation endpoint jump under a `1e-9` relative energy step was `2.5e-11`; `EagleResponse.apply(spec) == spec*QE` exactly at three energies.  Focused tests passed (`36 passed`).

The digitized manufacturer QE ordinates were not compared to the original datasheet.  Their provenance remains an empirical validation gap.  Follow-up after independent verification added `Validation: detector-eaglexo` to `EagleResponse` and corrected the non-load-bearing far-field estimate to about `0.12%`.

## Verdict

- **Claim**: `detector-eaglexo` — `detectors/eaglexo_response.py::EagleResponse` — rectangular `Omega` times measured `QE(E)` detector operator
- **Filters**: units `pass`; limits `pass`; signs/conventions `pass`
- **Re-derivation**: `matches` — operator and acceptance ownership match; empirical QE ordinates remain source-unverified
- **Verdict**: `filtered`
- **Write-up**: `docs/validation/detectors/detector-eaglexo.md`
- **Suggested ledger change**: applied — status `filtered`, pending datasheet-to-CSV comparison recorded, exact back-reference added, estimate corrected
