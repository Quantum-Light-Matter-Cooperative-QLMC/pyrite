# TODO / Backlog — feature/energy-grid-cli

User-reported `cxr energy-grid` CLI problems, folded from a `>user<` item on
`main` (2026-07-28, via `feature/cli-fixes` split). Original user text
preserved verbatim; triage notes inline.

## User text

1. `cxr energy-grid` options are confused and need to be fixed. What does `defaults` apply to? `brem`, `line`, both, or is it just used for deriving the appropriate upper-bounds for each? What does `apply` do? I think this section needs to be reworked, or at a minimum, the help explanations made more detailed
2. I think once values are set for `cxr energy-grid defaults`, they cannot be removed (even though when I started, the fields for `tilts` and `azimtuhs` were both empty--not sure what empty would mean here)
3. How does the `fidelity` field contained in the backend code relate to this? Where does `full` vs `survey` come into play?

## Triage / implementation path

- A1 (rework or document): audit `energy-grid` command group; decide whether `defaults` scopes to `brem`, `line`, or both, and whether it only derives upper bounds. If semantics are sound, expand `--help` text; if not, rework option structure. Breaking CLI change → follow `cli-ui-ux` skill + migration note.
- A2 (bug): no way to unset/clear a stored `energy-grid defaults` value. Add reset/unset path (e.g. `--clear` per field or `defaults --reset`); define and document what empty `tilts`/`azimuths` means.
- A3 (docs): explain mapping between backend `fidelity` field (`full` vs `survey`) and the energy-grid CLI surface.
