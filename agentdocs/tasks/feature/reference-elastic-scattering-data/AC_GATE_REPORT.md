# SRD 64 gate report: checklist A--C

Investigation date: 2026-08-09. This is a task-local decision record, not a
license grant or an independent validation of the physics.

## Source and version

The authoritative product is NIST Standard Reference Database 64, *NIST
Electron Elastic-Scattering Cross-Section Database*. The current online product
identifies itself as Version 5.0 (released in 2023), with its last data-content
update in 2002. Version 4.0 was the first web version (2016). The current data
are therefore not accurately identified by a bare "Version 4.0" provenance
label.

Official sources:

- [SRD 64 product page](https://www.nist.gov/srd/database-64-version-40)
- [current Version 5.0 interface](https://srdata.nist.gov/srd64/)
- [Version 5.0 history](https://srdata.nist.gov/srd64/More/VersionHistory)
- [NIST NSRDS 64-2023 users' guide](https://doi.org/10.6028/NIST.NSRDS.64-2023)
- [SRD copyright and licensing statement](https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications)
- [Standard Reference Data Act summary](https://www.nist.gov/srd/public-law)

NIST states that SRD 64 covers differential, total elastic, and transport cross
sections for atomic numbers 1--96 and electron energies from 50 eV through
300 keV. The users' guide states that the relativistic DCS was calculated by a
Dirac partial-wave method using self-consistent free-atom Dirac-Hartree-Fock
densities and the Furness-McCarthy local exchange potential. Transport cross
sections are derived from the DCS. Cross sections use `a0^2`, with
`a0^2 = 2.8002852e-21 m^2 = 2.8002852e-17 cm^2`; DCS in solid-angle
coordinates uses `a0^2 / sr`.

The online interface is free and offers text/CSV downloads. It does not
document a stable bulk API. Its element and energy forms use per-request
anti-forgery state, so a reproducible unattended acquisition cannot be claimed
from URL templates alone. A future acquisition record must pin the product
version, content vintage, query specification, access time, downloader
version, raw hashes, and conversion code; a manually downloaded opaque table
is insufficient.

## Redistribution decision

**Blocked pending written NIST permission.** The fact that the interface is
free does not grant redistribution rights. NIST's SRD-specific terms state
that SRD compilations are copyrighted by the Secretary of Commerce, all rights
are reserved, and SRD may not be reproduced, stored in a retrieval system, or
transmitted without prior permission. The generic public-domain/fair-use text
for other NIST data explicitly excludes SRD.

Consequences:

- Do not commit SRD 64 total, DCS, CDF, sampler, or newly downloaded transport
  values, including transformed or compressed tables, until written permission
  covers redistribution in cxr-mc distributions.
- The five already tracked `DisplayCalcTCSTableFor{C,Ge,Mo,Se,Si}.csv` files
  have the shape, units, and naming of SRD 64 downloads but no task-local
  permission record. Their continued redistribution needs the same audit; this
  slice does not delete or rewrite them.
- A permission request should identify the repository/distribution license,
  the exact source product/version, all 24 supported elements, intended raw or
  derived forms, package indexes/wheels/source archives, attribution text, and
  whether downstream redistribution and modification are permitted.

## Proposed offline schema (version 1)

This schema is approved as a contract proposal only. It authorizes no data and
does not select the runtime angular model.

Each dataset root contains `manifest.json` plus per-element NPZ payloads. The
manifest records `schema_version`, dataset and product identifiers, SRD product
version, data-content vintage, source URLs/DOIs, access timestamp, permission
record identifier, acquisition/conversion tool versions, request
specification, raw and normalized SHA-256 hashes, supported atomic numbers,
and explicit interpolation and endpoint-policy identifiers. A builder must
make archives deterministic or hash canonical uncompressed arrays.

`integral-Z###.npz`:

- `energy_eV`: float64, shape `(n_energy,)`, finite, strictly increasing,
  duplicate-free, exact endpoints 50 and 300000 eV.
- `total_cm2`, `transport_cm2`: float64 with the same shape, finite and
  positive. No monotonicity requirement is imposed on physical cross sections.
- Validation requires `0 <= transport_cm2 / total_cm2 <= 2`, consistent with
  the `(1 - cos(theta))` transport weighting, plus reproduction of source-unit
  conversions within declared tolerances.

`dcs-Z###.npz`:

- `energy_eV` as above; `theta_rad`: float64, finite, strictly increasing,
  duplicate-free, with exact endpoints 0 and pi.
- `dcs_cm2_sr`: float64, shape `(n_energy, n_theta)`, finite and nonnegative.
- Numerical quadrature must reproduce, within declared source/grid tolerances,
  `total = 2*pi*integral(dcs*sin(theta), theta)` and
  `transport = 2*pi*integral(dcs*(1-cos(theta))*sin(theta), theta)`.
- Any resampling onto a common angle grid is a derived product with its method,
  tolerance, and source-array hash recorded separately.

`cdf-Z###.npz` (only if the later D--F decision accepts direct CDF sampling):

- `energy_eV`, `theta_rad`, and `cdf` with shape `(n_energy, n_theta)`.
- Each row is finite and nondecreasing, starts exactly at 0, ends exactly at 1,
  and has no negative probability increments beyond a declared roundoff
  tolerance. The CDF construction method and normalization residual are stored
  in the manifest.

Endpoint behavior is fail-closed by default: energies outside 50--300000 eV
are rejected. Any later clamping or analytic fallback requires a distinct
policy identifier and explicit run metadata. Interpolation is part of the
physics model, not silently implied by storage; the manifest must name it, and
its acceptance belongs to D--G.

## Browning comparison contract and present evidence

Current `mott` collision rates use the Browning empirical total at every
energy in every CPU/CUDA transport core:

```text
sigma_B(Z,E_keV) = 3e-18 * Z^1.7 /
    (E_keV + 0.005*Z^1.7*sqrt(E_keV) + 0.0007*Z^2/sqrt(E_keV))  [cm^2]
```

The implementation documents this fit as valid only from 0.1 to 30 keV, yet
the supported run range extends to 300 keV. The catalog supports 24 elements:
`B C N O Al Si P S Ti V Fe Ge Se Zr Nb Mo Pd Te Hf Ta W Re Pt Bi`; only C,
Si, Ge, Se, and Mo currently have packaged SRD transport tables. Those tables
contain 401 linearly spaced energies from 50 eV through 300 keV in `a0^2` and
do not contain NIST totals.

When legally acquired NIST totals are available, compare all 24 elements on
the union of source knots and a declared logarithmic 1--300 keV benchmark grid,
including exact 30 keV and 300 keV boundaries. Record
`(sigma_B-sigma_NIST)/sigma_NIST`, absolute relative error, and
`log10(sigma_B/sigma_NIST)`, with per-energy, per-element, and validity-region
summaries. Error-surface artifacts must mark `E > 30 keV` as Browning
extrapolation and show endpoint behavior separately. Preserve source hashes and
the comparison program/version; do not commit source-valued or invertible
derived surfaces until permission covers them.

No authoritative NIST total values were captured in this branch. The official
interface requires interactive state and publishes no stable bulk API, and the
redistribution terms forbid committing downloaded/derived tables absent prior
permission. Therefore checklist C is blocked, not passed; no numerical error
surface or boundary agreement claim is made.

## Next dispatchable slice

Obtain written permission from the NIST Standard Reference Data Program
(`data@nist.gov`) for the precise redistribution described above, or obtain a
written determination that a specified derived representation may be
redistributed. Attach the response/permission identifier to this task record.
After permission, dispatch a bounded A--C continuation to implement a pinned
acquisition tool, capture raw hashes outside the package, normalize the
version-1 schema, validate units/integrals/endpoints, and generate the Browning
comparison. Without permission, the alternate planning slice is to identify an
authoritative redistributable source or a reproducible first-principles
generator; it must not relabel that result as NIST SRD 64.
