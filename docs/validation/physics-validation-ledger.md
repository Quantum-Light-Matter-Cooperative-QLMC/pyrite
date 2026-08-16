# Physics validation ledger

The single source of truth for **what physics PyRITE claims and whether it has been verified.** One labeled record per atomic physics claim. Goal: every load-bearing equation reaches `signed-off` before publication. See the [validation methodology](methodology.md) for the status lifecycle and re-derivation workflow.

> [!note]
> **Seeded, not complete.** The ledger parts are the core physics, extracted from `docs/repo_map.md` + the in-code citations. Remaining formulas (detector internals, geometry helpers, edge corrections) still need ledgering — grep the physics modules for un-annotated `def`s. Anchor on `file::symbol`, never a line number.

**Status:** `unverified` → `filtered` (units+limits+signs) → `rederived` (independent derivation matches) → `anchored` (regression test green) → `signed-off` (human-certified). `discrepancy` = a check failed.

For current totals, use the generated [status summary](status-summary.md). For a
short claim-by-claim view, use the generated [domain inventories](domain-inventories.md).
The detailed records in the ledger parts below remain authoritative. Each
validation ID is a stable heading followed by the seven fields `Claim`, `Code`,
`Source`, `Status`, `Checks`, `Anchor`, and `Notes`. Anchor on `file::symbol`,
never a line number.

Each part file is one domain: its `H1` is the domain name and every `H2` is one
validation ID. Add a record by editing the matching part, then run
`pyrite-dev validation-ledger --write` to refresh the generated views. Adding a
part means adding its file to the toctree below, which is the order the
generated views follow.

## Ledger parts

| Part | Contents |
|---|---|
| [Core coherent physics](ledger-core-coherent-physics.md) | highest-risk PXR/CBS line physics: amplitudes, lineshape, coherent sums, batching |
| [Crystallography & atomic data](ledger-crystallography-atomic-data.md) | structure and form factors, Debye–Waller policy, CIF/COD adapters, X-ray optical constants |
| [Crystal structure & CIF provenance](ledger-crystal-structure-provenance.md) | per-material lattice, basis, and Debye–Waller provenance rows for the catalog |
| [Transport & background](ledger-transport-background.md) | electron transport, bunch sampling, bremsstrahlung, error estimators |
| [Mosaicity & multilayer](ledger-mosaicity-multilayer.md) | mosaic spread models and multilayer stack reflectivity |
| [Detector forward models](ledger-detector-forward-models.md) | detector response, efficiency, and readout models |
| [Grazing-incidence grating spectrometer](ledger-grazing-grating-spectrometer.md) | grating dispersion, efficiency, and grazing optics |

```{toctree}
:maxdepth: 1
:hidden:

ledger-core-coherent-physics
ledger-crystallography-atomic-data
ledger-crystal-structure-provenance
ledger-transport-background
ledger-mosaicity-multilayer
ledger-detector-forward-models
ledger-grazing-grating-spectrometer
```
