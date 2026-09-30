# Elemental transport data

The X-ray side of PyRITE resolves atomic data from external databases. The electron-transport catalog uses a hand-curated table, `materials/_transport_data.py::TRANSPORT_ELEMENTS`, to provide atomic number, weight, and mean excitation energy for its supported elements. Production collision stopping uses these values to generate material-level SBETHE tables; BremsLib supplies separate evaluated bremsstrahlung cross sections. This page records the elemental inputs and their coverage limit.

## The three constants

```{list-table} Per-element constants and their role.
:name: tbl-transport-data-roles
:header-rows: 1

* - Symbol
  - Quantity
  - Consumed by
* - $Z$
  - atomic number
  - SBETHE composition and BremsLib cross-section $Z^2$ scaling; the CBS
    coupling's net screened-nucleus combination $Z_j - f_j$; the Henke forward
    factor $f_1 = Z + f'$
* - $A$
  - standard atomic weight
  - material density supplied to SBETHE
* - $J$
  - mean excitation energy [keV]
  - SBETHE compound mean excitation energy $I$
```

$Z$ and $A$ use the CIAAW 2024 standard atomic weights.{cite:p}`ciaaw2024` $J$ uses the PDG *Atomic and Nuclear Properties* elemental tables,{cite:p}`pdg2025` which follow the ICRU stopping-power compilation.

The table currently covers 24 elements: B, C, N, O, Al, Si, P, S, Ti, V, Fe, Ge, Se, Zr, Nb, Mo, Pd, Te, Hf, Ta, W, Re, Pt, Bi.

## Where $A$ actually enters

$A$ enters the mass density supplied to SBETHE. For atomic number densities $n_i$ in Å$^{-3}$, the catalog derives

```{math}
:label: eq-transport-data-density

\rho = \frac{10^{24}}{N_{\rm A}}\sum_i n_i A_i,
\qquad \rho~\text{in g cm}^{-3}.
```

The $n_i$ come from the catalog composition and lattice volume. See [Stopping power and the energy cutoff](../beam-transport/stopping-power.md) for the compound mean excitation energy and the table interpolation.

The catalog therefore derives density from the composition instead of maintaining a separate mass-density value.

## The $J$ provenance caveat

$J$ in this table is the elemental mean excitation energy $I_i$ supplied to SBETHE. The material value follows the logarithmic Bragg rule, weighted by $n_iZ_i$. The older Joy–Luo reference fit used Berger–Seltzer values, which differ from these PDG/ICRU inputs for some light elements; that discrepancy applies to historical reference comparisons, not the SBETHE production formula.

The catalog uses one cited elemental input set across its materials.

## The table is a gate

Catalog validation requires that **every element in a runnable material's full composition** — the film crystal plus every substrate and stack layer, crystal or amorphous medium — appears in `TRANSPORT_ELEMENTS`. A material referencing an element outside it is rejected at load time with the offending symbols named, rather than failing partway through a sweep.

The asymmetry with the X-ray side is intentional and worth stating plainly:

```{list-table} What adding a new element costs.
:name: tbl-transport-data-cost
:header-rows: 1

* - Data
  - Source
  - Cost of a new element
* - $f_0$, $f'$, $f''$, $Z$
  - external database, resolved by symbol
  - free — see [Atomic form factors](atomic-form-factors.md)
* - $Z$, $A$, $J$
  - this table
  - one hand-added row with citations
* - Mott elastic transport cross sections
  - NIST SRD 64 CSV per element
  - a user-downloaded table (`elastic_model="mott"` only)
```

PyRITE ships no NIST Mott transport tables: NIST SRD 64 may not be redistributed (#263). The opt-in `elastic_model="mott"` reads tables the user downloads and fails naming the element when one is missing; the default ELSEPA model and the analytic `"sr"` model need none. See [Elastic scattering](../beam-transport/elastic-scattering.md#installing-the-mott-tables). So an element can be fully supported for X-ray couplings, adequately supported for stopping and bremsstrahlung, and still need data the user supplies for one elastic model — three different coverage tiers over the same periodic table.

## Assumptions and limits

* SBETHE receives a compound $I$ from the logarithmic Bragg rule. Its corrected Bethe calculation includes a density-effect correction above its material-dependent `ECUT`; chemical-environment effects enter only through the supplied $I$ or optional band gap.
* Neutral, ground-state atoms at nominal conditions; no charge-state or chemical-environment dependence of $J$.
* Standard atomic weights are terrestrial-abundance averages; isotopically enriched samples are not represented.
* The constants are static: nothing here carries a temperature or phase dependence.

## Validation

These are tabulated input constants; `sbethe-material-inputs` validates their composition into a material table and `sbethe-corrected-stopping` validates its use in transport. The BremsLib rows cover the $Z^2$ cross-section scaling.

See the [validation ledger](../../validation/physics-validation-ledger.md).

Implementation owners: `pyrite.materials._transport_data.TRANSPORT_ELEMENTS`; consumed by `pyrite.montecarlo.transport`, `pyrite.montecarlo.spectrum.brem`, and the catalog validator in `pyrite.materials.catalog`.
