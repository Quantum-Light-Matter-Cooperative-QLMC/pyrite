# TODO: crystallography adapter review (TODO P2 #1)

Read-only review of `origin/feature/diffpy` (`67f27e0`, diffpy.structure CIF importer) and
`origin/feature/dans-diffraction` (`d4dbeff`, optional Dans_Diffraction validation oracle) —
neither branch was checked out, merged, or executed; see
[`docs/crystallography-adapters-review.md`](docs/crystallography-adapters-review.md) for the
full findings, drift/conflict analysis, and MERGE-AFTER-FIXES recommendation for both.

Note: `TODO.md` P2 #1 on `main` names these branches `codex/diffpy-structure-importer` and
`codex/dans-diffraction-research`; those do not exist. The correct refs are the two above —
fix this reference when this review's findings are triaged back into `main`'s `TODO.md`.

No unique-value research doc for the original `crystals` package exists anywhere in this
repo's history (the dans-diffraction branch's own research doc cites one that was never
committed); the review's answer to that open question is general knowledge, not repo-verified,
and recommends a short install-and-probe spike before deciding.
