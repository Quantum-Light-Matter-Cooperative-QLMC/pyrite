# Reference electron stopping data

## Problem and scope

Joy--Luo plus elemental Bragg additivity is a useful lightweight CSDA fallback,
but it is not a reference stopping model across every material from 1 to
300 keV. It simplifies shell, exchange, relativistic, condensed-phase, and
radiative corrections. The current runtime exposes no tabulated collisional
versus radiative stopping provenance or fallback metadata.

Source: [`docs/electron_transport_physics_recommendations.docx`](../../../../docs/electron_transport_physics_recommendations.docx), Stage 2 stopping work.

This task owns offline, packaged stopping-power data and the runtime stopping
model. It does not own numerical energy substeps, elastic scattering, explicit
hard inelastic events/straggling, or bremsstrahlung angular distributions.

## Implementation path and likely owners

Current owners are `materials/_transport_data.py`, the Joy--Luo helpers and
four propagation cores in `montecarlo/transport.py` and
`transport_jit_kernel.py`, catalog material composition, packaged data, and
run/checkpoint metadata.

- Evaluate ESTAR above roughly 10 keV and a PENELOPE-style or corrected-Bethe
  source for the low-energy region and compound/material effects.
- Establish source version, assumptions, units, interpolation, material mixing,
  generation/reproduction, and redistribution policy.
- Package collisional and radiative stopping components for offline Numba/CUDA
  interpolation; retain Joy--Luo/analytic radiative stopping only as explicit
  fallbacks.
- Make selected model/version, endpoint clamp, composition approximation, and
  fallback use visible in result metadata/checkpoint identity.

## Checklist

- [ ] A -- Define the material/energy benchmark matrix and accuracy target for
      low/high Z, compounds, 1--10 keV, and 10--300 keV.
- [ ] B -- Resolve ESTAR/PENELOPE/sbethe source versions, redistribution, units,
      and reproducible table-generation workflow.
- [ ] C -- Quantify current Joy--Luo bias against candidate sources, including
      low-energy and compound/additivity behavior.
- [ ] D -- Define and validate a versioned packaged schema for collisional,
      radiative, and total stopping with explicit endpoint policy.
- [ ] E -- Integrate the accepted interpolator/model into CPU transport and
      expose model/fallback/clamp metadata.
- [ ] F -- Port the same model to per-electron and CUDA cores; measure table
      footprint, interpolation cost, and device-memory impact.
- [ ] G -- Check stopping ranges, energy deposition, cutoff behavior, and
      line/bremsstrahlung spectra across the benchmark matrix.
- [ ] H -- Reconcile radiative stopping with generated bremsstrahlung so energy
      accounting is documented and not silently double-counted or omitted.
- [ ] I -- Update package manifests, public science/provenance docs, validation
      ledger, and golden data.

## Decisions and open questions

- **Decided:** Joy--Luo remains available only as a documented explicit
  fallback once the reference path is accepted.
- **Open/blocking:** exact primary/reference source by energy/material region
  and its redistribution terms.
- **Open:** compound handling: packaged material tables, elemental mixing, or a
  generation tool tied to catalog compositions.
- **Open:** low-energy splice/interpolation between sources and its uncertainty
  budget; ESTAR alone does not close 1--10 keV.
- **Open:** whether radiative stopping is enabled by default over the present
  range and how it is coupled consistently to `mc_brem_spectrum`.
- **Open:** checkpoint/case identity and compatibility when stopping model
  version changes.

## A--D gate investigation (2026-08-09)

### Verified primary-source facts

- NIST identifies ESTAR/PSTAR/ASTAR as **NIST Standard Reference Database
  124**, DOI
  [`10.18434/T4NC7P`](https://doi.org/10.18434/T4NC7P), with data content last
  updated in July 2017. Treat that date and DOI as the candidate source version;
  a fetched table without both is not provenance-complete.
- The official ESTAR method page defines collision, radiative, and total mass
  stopping powers in MeV cm^2/g. Total is the sum of the two components. User
  energies span 1 keV--10 GeV, but NIST recommends restricting collision
  stopping use to at least 10 keV because shell corrections are omitted.
- NIST reports estimated collision-stopping uncertainties of 1--2% above
  100 keV; from 10--100 keV, 2--3% for low-Z and 5--10% for high-Z media. At
  1 keV it estimates about 10% error for low-Z media and expects ESTAR to
  overestimate very-low-energy stopping. Radiative stopping is estimated at
  5% uncertainty below 2 MeV. Therefore ESTAR cannot by itself set a defensible
  1--10 keV acceptance target or splice policy.
- ESTAR accepts density, elemental weight fractions, and a mean excitation
  energy. Its default compound I-value uses a modified Bragg-additivity rule
  with approximate binding and phase effects. An ESTAR compound result is thus
  not interchangeable with elemental stopping-power mixing and must preserve
  the exact composition, density, and I-value inputs.

Primary sources:

- [NIST SRD 124 record](https://doi.org/10.18434/T4NC7P)
- [ESTAR description and method](https://physics.nist.gov/PhysRefData/Star/Text/method.html)
- [NIST SRD/data licensing policy](https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-and-software)
- [ESTAR-linked database disclaimer](https://www.nist.gov/physical-measurement-laboratory/database-disclaimer)

### Redistribution decision and stop

**Fail closed: do not package or commit ESTAR-derived tables yet.** NIST's
official policy says SRD compilations are copyrighted by the U.S. Secretary of
Commerce, all rights are reserved, and licensing information applies where an
SRD is licensed. The SRD 124 record and its linked database disclaimer provide
access and warranty terms but no express redistribution or derivative-data
grant. The broader permission for non-SRD NIST data does not apply because the
record explicitly classifies ESTAR as SRD 124. Free CGI access is not evidence
of redistribution permission.

This triggers the dispatched stop condition. PENELOPE/sbethe version and terms,
the benchmark matrix, Joy--Luo bias measurements, and the packaged schema remain
unresolved; no authoritative table was downloaded or retained. Continuing
would also require a defensible low-energy and compound policy, which current
ESTAR evidence does not supply.

### Exact next dispatchable slice

Obtain an explicit written NIST license/permission determination for
redistributing generated SRD 124 stopping values in cxr-mc, including modified
or interpolated subsets and required notices. Record the response or governing
license text here. If redistribution is denied, limit ESTAR to an optional
user-run validation oracle and select a separately redistributable production
source. Only after that gate closes should A--D resume with official
PENELOPE/sbethe terms, the 1--300 keV benchmark, Joy--Luo comparisons, and a
versioned schema proposal.

## Alternative-source license gate (2026-08-09)

### Official PENELOPE record

The current OECD Nuclear Energy Agency Data Bank record is
[`PENELOPE2023`, NEA-1525/24](https://www.oecd-nea.org/tools/abstract/detail/nea-1525/),
tested 2024-09-12. The record states that PENELOPE2023 retains the 2018
physics and the same material data files. It transports electrons, positrons,
and photons from 50 eV to 1 GeV, while warning that its interaction models are
not expected to be accurate below about 1 keV. It covers arbitrary materials
and models electron inelastic loss and bremsstrahlung as separate interaction
classes. Those properties cover the requested 1--300 keV interval in principle,
but the public abstract does not specify a standalone stopping-table export,
its units, or a reproducible `sbethe` interface.

The official distribution is controlled rather than public. The record requires
an order/request for NEA-1525/24 and identifies the source repository as
restricted. Following the linked repository or end-user route reaches the
[NEA GitLab authentication page](https://git.oecd-nea.org/penelope/package/pensuite),
which requires an official registered email and two-factor authentication for
controlled content. The public record is OECD-copyrighted and supplies no
license granting redistribution of the source, its material files, or generated
tables. No controlled package, click-through terms, or numerical data were
accessed.

The public NEA abstract does not identify `sbethe` as an independently versioned
or distributed product. The cited
[`PENELOPE-2018` manual](https://doi.org/10.1787/32da5043-en) and restricted
package are therefore the authoritative places to resolve whether `sbethe` is
a bundled executable/subroutine, its inputs and output units, and whether its
outputs may be redistributed. Direct unauthenticated access to the manual and
package was unavailable during this audit. Treating `sbethe` as a separate
open source would be unsupported.

### Candidate decision matrix

| Candidate | Version/provenance | Energy/components/materials | Reproduction path | Redistribution gate | Decision |
|---|---|---|---|---|---|
| ESTAR | NIST SRD 124, July 2017, DOI `10.18434/T4NC7P` | 1 keV--10 GeV; collision/radiative/total; elements and user materials, but >=10 keV recommended | Public CGI with exact composition, density, I-value, and energies | SRD rights reserved; no express table grant located | Validation oracle only unless NIST grants permission |
| PENELOPE | NEA-1525/24 PENELOPE2023; 2018 physics/material files | 50 eV--1 GeV; inelastic and bremsstrahlung interactions; arbitrary materials; accuracy warning below about 1 keV | Controlled Fortran suite and material generator; exact compiler, inputs, and export procedure require package/manual access | Request, registered official email, 2FA, restricted repository; public page gives no redistribution grant | Technically plausible, legally and reproducibly blocked |
| `sbethe` | No independently versioned public official record found | Not established from accessible primary material | Must be resolved inside the authoritative PENELOPE manual/package | Same controlled-access terms; independent license not established | Not a selectable source |

None of these candidates presently closes the production-data gate. In
particular, physical coverage is not equivalent to permission to package
generated values, and PENELOPE's roughly 1 keV accuracy boundary does not by
itself justify the endpoint uncertainty target or a compound/splice policy.

### Required external/user decision

Choose one of these mutually exclusive next steps:

1. An eligible institutional user requests NEA-1525/24, accepts the governing
   terms outside this workflow, and obtains written confirmation covering
   redistribution of generated stopping tables and required notices. The next
   task slice can then record, without committing controlled files, the exact
   PENELOPE/`sbethe` version, hashes, compiler, material inputs, units, and
   generator invocation.
2. Keep PENELOPE and ESTAR as user-run validation oracles and authorize a new
   source-gate slice restricted to publicly licensed generator implementations
   (for example official Geant4 or EGSnrc releases). That slice must separately
   audit code and bundled-data licenses, output-table rights, component export,
   compound inputs, and 1 keV validity before proposing either source.

Until one path is selected and its rights evidence recorded, do not define the
production benchmark tolerance, compound/splice policy, packaged schema, or
Joy--Luo bias against retained reference values.

## Delegation slices and required skills

- A--D require `lead-task`, `repo-orientation`, `monte-carlo`,
  `scientific-library`, `physics-review`, and `documentation-maintenance`;
  not `one-shot` while source and compound policy remain open.
- E--F require `lead-task`, `monte-carlo`, `performance`, and
  `remote-gpu-jobs`.
- G--H require `regression-testing`, `physics-review`, and fresh-context
  `physics-validation`; heavy/GPU matrices use `cxr remote`.
- I uses `regen-golden` and `documentation-maintenance`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test-suite core
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test-suite packaging
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev lint
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev typecheck
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev verify
```

- Every packaged table/model is versioned, reproducible, source-attributed,
  unit-validated, and legally redistributable.
- Documented benchmark errors cover 1--300 keV, low/high Z, and representative
  compounds; the 1--10 keV uncertainty is explicit rather than inferred from
  ESTAR alone.
- Runtime metadata identifies collisional/radiative model versions, clamps,
  composition approximations, and every fallback.
- CPU and CUDA implementations share the accepted interpolation/model and pass
  range, deposition, cutoff, statistical, and performance gates.
- Radiative stopping and bremsstrahlung energy accounting are mutually
  consistent under documented assumptions.
- Changed physics has validation markers/ledger rows, fresh-context validation,
  and regenerated goldens where required.
