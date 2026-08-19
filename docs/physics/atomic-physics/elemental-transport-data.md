# Elemental transport data

The X-ray side of PyRITE resolves any element on demand from an external
database. The **electron-transport** side does not: stopping power and
bremsstrahlung need per-element atomic constants that come from a different
literature, and those live in one small hand-curated table,
`materials/_transport_data.py::TRANSPORT_ELEMENTS`. This page states what is in
it, where each number comes from, and what the table's existence constrains.

## The three constants

```{list-table} Per-element constants and their role.
:name: tbl-transport-data-roles
:header-rows: 1

* - Symbol
  - Quantity
  - Consumed by
* - $Z$
  - atomic number
  - Joy–Luo stopping power; bremsstrahlung cross-section $Z^2$ scaling; the CBS
    coupling's net screened-nucleus combination $Z_j - f_j$; the Henke forward
    factor $f_1 = Z + f'$
* - $A$
  - standard atomic weight
  - the mass-density form of the Bethe law, $\rho Z/A$
* - $J$
  - mean excitation energy [keV]
  - the logarithm argument of the Joy–Luo stopping power
```

$Z$ and $A$ use the CIAAW 2024 standard atomic weights.[^ciaaw] $J$ uses the PDG
*Atomic and Nuclear Properties* elemental tables,[^pdg] which follow the ICRU
stopping-power compilation.

The table currently covers 24 elements: B, C, N, O, Al, Si, P, S, Ti, V, Fe, Ge,
Se, Zr, Nb, Mo, Pd, Te, Hf, Ta, W, Re, Pt, Bi.

## Where $A$ actually enters

$A$ appears in the textbook mass-density form of the stopping power through the
combination $\rho Z / A$. The production compound path never evaluates that
form. It works from number densities instead, storing the per-element
coefficient

```{math}
:label: eq-transport-data-coefficient

c_i = \frac{n_i Z_i}{0.602214076},
\qquad n_i~\text{in Å}^{-3},
```

which is an exact rewrite of $\rho Z/A$ using $\rho/A = n/N_{\rm A}$ — so $A$
cancels identically. See
[Stopping power and the energy cutoff](../beam-transport/stopping-power.md) for
the full expression.

This matters for catalog design: because number densities are derived from the
CIF unit-cell volume and site count, the catalog never has to carry a mass
density that stays consistent with the lattice, and an inconsistent $A$ cannot
silently perturb transport. $A$ is retained for the exported single-element
reference helper and for anyone comparing against the mass-density literature
form.

## The $J$ provenance caveat

$J$ is the one constant here that is a *fit input* rather than a measured atomic
property, and the fit PyRITE uses and the table PyRITE reads do not come from the
same compilation. Joy and Luo calibrated their low-energy correction against
Berger–Seltzer mean excitation energies; the PDG/ICRU values used here disagree
for light elements — carbon is 78 eV against 100 eV in the original fit, worth
roughly 4 % in stopping power at 25 keV, while silicon agrees to 0.3 %.

This is a deliberate trade: one consistent, citable dataset across the whole
catalog, with the discrepancy recorded, rather than exact fidelity to a single
1989 fit. It is documented as a known offset in the stopping-power page, not
tuned away.

## The table is a gate

Catalog validation requires that **every element in a runnable material's full
composition** — the film crystal plus every substrate and stack layer, crystal
or amorphous medium — appears in `TRANSPORT_ELEMENTS`. A material referencing an
element outside it is rejected at load time with the offending symbols named,
rather than failing partway through a sweep.

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
  - a downloaded table, or accept the analytic fallback
```

Only five elements — C, Si, Ge, Se, Mo — ship NIST Mott transport tables.
Everything else falls back to the analytic screened-Rutherford screening
parameter; see
[Elastic scattering](../beam-transport/elastic-scattering.md) for what that
fallback does and does not preserve. So an element can be fully supported for
X-ray couplings, adequately supported for stopping and bremsstrahlung, and still
be on the approximate branch for elastic deflection — three different coverage
tiers over the same periodic table.

## Assumptions and limits

* Elemental values only. There is no compound $J$, and no condensed-phase
  density-effect correction; compounds are handled by Bragg additivity over
  these per-element constants, each keeping its own $J_i$ and its own low-energy
  correction $k_i$, with no effective $Z$ or $J$ for the mixture.
* Neutral, ground-state atoms at nominal conditions; no charge-state or
  chemical-environment dependence of $J$.
* Standard atomic weights are terrestrial-abundance averages; isotopically
  enriched samples are not represented.
* The constants are static: nothing here carries a temperature or phase
  dependence.

## Validation

These are tabulated input constants, not derived expressions, so they carry no
validation row of their own. Their consumers do: `transport-midpoint-stopping`
and `electron-transport` for the stopping and elastic models, and the
bremsstrahlung rows for the $Z^2$ scaling. The $J$-source discrepancy above is
recorded against the stopping-power claim rather than hidden in the table.

See the [validation ledger](../../validation/physics-validation-ledger.md).

Implementation owners: `pyrite.materials._transport_data.TRANSPORT_ELEMENTS`;
consumed by `pyrite.montecarlo.transport`,
`pyrite.montecarlo.spectrum.brem`, and the catalog validator in
`pyrite.materials.catalog`.

[^ciaaw]: Commission on Isotopic Abundances and Atomic Weights, standard atomic
    weights (2024 revision), <https://ciaaw.org/atomic-weights.htm>.

[^pdg]: Particle Data Group, *Atomic and Nuclear Properties of Materials*,
    <https://pdg.lbl.gov/2025/AtomicNuclearProperties/>, following the ICRU
    stopping-power compilation.
