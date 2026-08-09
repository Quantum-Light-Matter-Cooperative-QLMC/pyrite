# Alternative elastic-data source and license gate

Investigation date: 2026-08-09. This is a task-local source-selection record,
not legal advice, a license grant, or independent physics validation. No
numerical cross-section data were downloaded or retained.

## Production-data requirement

An acceptable production path must be reproducible without credentials or an
interactive license acceptance, cover the catalog elements and 50 eV--300 keV,
provide total, first-transport, and differential elastic cross sections (or
enough primary output to derive them), preserve the version-1 schema contract
in [`AC_GATE_REPORT.md`](AC_GATE_REPORT.md), and permit cxr-mc to redistribute
the packaged source or derived data to commercial and noncommercial users.
Model-form agreement with SRD 64 cannot be inferred from similar method names:
the atomic potential, exchange, polarization, absorption, target state, grids,
and numerical solver configuration are part of the provenance.

## Candidate matrix

| Candidate | Official terms | Coverage and reproducibility | Schema/accuracy decision |
| --- | --- | --- | --- |
| ELSEPA 2020/2021, archive version 1, DOI [`10.17632/w4hm5vymym.1`](https://doi.org/10.17632/w4hm5vymym.1), program article DOI [`10.1016/j.cpc.2020.107704`](https://doi.org/10.1016/j.cpc.2020.107704) | The [official Mendeley archive](https://data.mendeley.com/datasets/w4hm5vymym/1) labels the distribution **CC BY-NC 3.0**. That permits copying and adaptation with attribution only for noncommercial use. | The official description says the Fortran program calculates differential and integrated electron/positron elastic cross sections with RADIAL, configurable real or complex central potentials, and an optical model for elemental solids. It includes plotting scripts. The archive description does not establish exact Z/energy coverage or a canonical configuration matching SRD 64; proving those would require a controlled source inspection/run under its license. | Technically promising generator, but fails the unrestricted-redistribution requirement. The free-atom versus elemental-solid optical-potential choice is unresolved. Do not package its source or derived tables without a broader written grant and model evidence. |
| Original ELSEPA release, archive version 1, DOI [`10.17632/5zzrz874tt.1`](https://doi.org/10.17632/5zzrz874tt.1), program article DOI [`10.1016/j.cpc.2004.09.006`](https://doi.org/10.1016/j.cpc.2004.09.006) | The archive identifies the **Computer Physics Communications Program Library licence**. The [official Elsevier CPC licence](https://www.elsevier.com/about/policies/open-access-licenses/elsevier-user-license/cpc-license) limits the acquired program to academic/nonprofit use and forbids source or executable redistribution to third parties without written author permission; commercial use requires contacting the author. It does not expressly grant redistribution of generated cross-section tables. | A reproducible Fortran generation path exists in principle, but the distribution is license-restricted and the rights for packaged outputs are not established. | Rejected as a packaged generator and blocked as a packaged-data source. Similarity to the SRD partial-wave method is not evidence of numerical equivalence. |
| Geant4 11.3.2 DPWA plus G4EMLOW 8.6.1 | The [official release page](https://geant4.web.cern.ch/download/11.3.2.html) points to the [Geant4 software licence](https://geant4.web.cern.ch/download/license), which permits source/binary use, modification, and redistribution with notices. The release page separately downloads G4EMLOW, but the cited software licence does not expressly identify the dataset or its upstream numerical-data rights. | The official 11.3.2 [`G4eDPWAElasticDCS.hh`](https://github.com/Geant4/geant4/blob/v11.3.2/source/processes/electromagnetic/standard/include/G4eDPWAElasticDCS.hh) documents free-atom DPWA DCS, total, first-transport, and second-transport cross sections for Z through 103 and 10 eV--100 MeV. The implementation requires G4EMLOW 7.12 or later. It explicitly warns that the free-atom result may be questionable below a few hundred eV. Reproduction therefore requires a pinned Geant4 source release and separately pinned G4EMLOW archive, hashes, build/compiler details, and an exporter whose normalization is validated. | Numerically and structurally the closest alternative, and it covers the required nominal domain. It is not currently defensible for cxr-mc packaging because G4EMLOW redistribution/provenance is not expressly closed by the cited terms, and the low-energy warning conflicts with an unqualified 50 eV accuracy claim. Candidate only after written dataset-rights confirmation and benchmark evidence. |
| Geant4 screened-Mott `G4eSingleScatteringModel` | Same Geant4 software licence. | The [official Physics Reference Manual](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/elastic_scattering/elecnuc.html) describes a fitted screened-Mott treatment suitable from about 200 keV; the underlying fit is stated for several keV upward and Z no greater than 90. It is an analytic runtime model, not the required versioned offline DCS source. | Rejected: materially weaker energy and element coverage and no 50 eV--300 keV reference-table path. |
| PENELOPE 2023, NEA-1525/24 | The [official OECD NEA package record](https://www.oecd-nea.org/tools/abstract/detail/nea-1525/) requires an order/request and links a restricted-access repository. The repository requires an NEA-registered official email, authentication, and two-factor authentication. Public redistribution rights are not stated on the package record. | The record states 50 eV--1 GeV transport but warns the adopted models are not expected to be accurate below about 1 keV. Obtaining and auditing the generator requires credentials and acceptance of end-user terms unavailable in this investigation. | Rejected for this slice under the access stop condition; also materially weakens the low-energy accuracy claim. It cannot establish a public reproducible acquisition path. |
| User-supplied NIST SRD 64 Version 5.0 oracle | Governed by the NIST restrictions recorded in [`AC_GATE_REPORT.md`](AC_GATE_REPORT.md); cxr-mc may not package or retain source-valued results without permission. | The user manually supplies legally obtained exports outside the repository. A local comparison tool can validate candidate totals/DCS on the declared grid, calculate non-invertible aggregate pass/fail metrics, and delete transient normalized values. Raw exports, source-valued plots, and invertible error surfaces remain local. | Accepted only as a validation-oracle design, not a production-data source. It cannot make another model "NIST data" or authorize redistributed derived tables. |

## Reproducible generation and validation contract

If Geant4 DPWA is licensed and selected for evaluation, pin `Geant4 11.3.2`
and `G4EMLOW 8.6.1` initially; record immutable archive URLs, SHA-256 hashes,
the Geant4 and dataset licence texts, compiler/CMake versions and flags, target
atomic numbers, energy/angle grids, projectile, and every potential/exchange/
polarization/absorption switch. Export total, first-transport, second-transport,
and DCS values before converting into the version-1 schema. Validate positivity,
grid ordering/endpoints, units, DCS quadrature against integral cross sections,
determinism across two clean builds, and canonical payload hashes. Generated
tables remain outside version control until redistribution rights are explicit.

The SRD oracle comparison must use the grid and metrics already specified in
[`AC_GATE_REPORT.md`](AC_GATE_REPORT.md), extended to DCS quadrature, first two
angular moments, and backscatter probability. Report separate 50--300 eV,
300 eV--30 keV, and 30--300 keV summaries so low-energy model limitations and
Browning extrapolation cannot be hidden by aggregate metrics. Until source
permission covers richer artifacts, commit only tool code, input hashes,
configuration, tolerances, and non-invertible pass/fail summaries; keep raw
and source-reconstructable values local.

Known validation limitations remain:

- NIST SRD 64 and Geant4 DPWA may use related partial-wave methods while using
  different potential details or grids. Agreement must be measured, not
  assumed.
- Free-atom elastic scattering is not automatically an accurate condensed-
  matter model at the lowest energies; both Geant4 and PENELOPE publish
  low-energy cautions.
- A total-cross-section comparison alone cannot qualify the angular model.
  DCS normalization, first two moments, and tail/backscatter behavior are
  required before any D--F runtime choice.
- Oracle results cannot be advertised as independent validation of either
  source; this is implementation-context comparison against SRD 64.

## Decision and exact external gates

**No investigated candidate is presently defensible as unrestricted packaged
production data.** Keep SRD 64 as a user-run validation oracle and do not
select a runtime model from this report.

The closest production candidate is Geant4 DPWA/G4EMLOW. Before another
implementation slice, obtain written confirmation from the Geant4 data owner
or CERN that the exact G4EMLOW 8.6.1 DPWA files, and cxr-mc transformations of
them, may be redistributed in source distributions, wheels, and package
indexes for commercial and noncommercial use, with modification and downstream
redistribution. The confirmation must identify required notices and any
upstream ELSEPA restrictions. Separately, a project physics owner must approve
an evidence threshold for the 50--300 eV region or narrow cxr-mc's supported
accuracy claim; this report cannot waive the documented low-energy caveat.

An alternative is written permission from the ELSEPA authors covering
commercial use, redistribution of the selected source version and generated
tables, modification, downstream redistribution, and the exact attribution.
The project must then select and document the free-atom/solid potential and
other model configuration before generation.

## Next dispatchable slice

After one of those written grants is attached to this task record, dispatch a
bounded, non-runtime qualification slice: implement a pinned external builder
and schema validator, generate candidate data outside version control, and run
the local user-supplied SRD oracle comparison. Commit only legally cleared data
and non-invertible validation evidence. If neither grant is obtainable, the
required product decision is to narrow the supported physics claim/range or
fund a separately licensed first-principles implementation and validation;
proceeding to checklist D--H is not justified.
