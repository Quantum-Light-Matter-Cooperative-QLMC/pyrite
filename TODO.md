# TODO — feature/zhai-supplementary-coverage-gaps

This branch carries one backlog item; the full triaged backlog lives on `main`.

## Zhai supplementary — missing HOPG + h-BN plots (P2)

`checks/anchor_figures.py::ZHAI_SUPPLEMENTARY_STUDIES` only defines `wse2`, `mose2`, and
`hbn` (921 nm only). The user flagged that the supplementary-info recreation is missing:

- **HOPG**, at thicknesses 29 nm, 76 nm, 150 nm, 17 um, 500 um, and 1 mm — no `hopg` entry
  exists in `ZHAI_SUPPLEMENTARY_STUDIES` at all. Polar/azimuthal tilts are already
  transcribed in [`docs/validation/zhai-supplementary.md`](docs/validation/zhai-supplementary.md)
  (Supplementary Table 4), and plotting should otherwise match the h-BN treatment.
- **h-BN**, additional thicknesses 42 nm, 109 nm, 219 nm (two tilt conditions reported at
  this thickness), 659 nm, and 170 um — only 921 nm is currently modeled. Tilts for these
  are likewise already transcribed in the same Table 4 rows.

**Implementation path.** Add a `hopg` `SupplementaryCoherentStudy` entry and extend the
`hbn` entry's `thicknesses_nm`, using the tilt (`theta_til`/`phi_til`) values already in
`docs/validation/zhai-supplementary.md`'s Table 4 transcription. Each condition's polar
tilt/azimuth pair is per-thickness (not shared across thicknesses like the current
`hbn`/TMD entries), so check whether `SupplementaryCoherentStudy`/`SupplementaryCondition`
need reshaping to key tilt by thickness rather than assuming one tilt set applies to every
listed thickness. Wire the new entries through the validation app + `cxr check --export`
figure builders alongside the existing WSe2/MoSe2/h-BN panels.

No new physics — this reuses the already-ledgered `line-energy-dispersion`,
`coherent-line-spectrum`, and `electron-transport` claims with different geometry inputs.
