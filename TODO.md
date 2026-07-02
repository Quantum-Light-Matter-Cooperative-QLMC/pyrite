# TODO / Backlog

Items live on `feature/...` / `bugfix/...` / `docs/...` branches, not `main`, until finished.
Full detail for an in-progress item lives on its branch (or its design doc);
`main` keeps only a one-line summary + pointer, enforced by /docs:todo-sync.
Priorities weigh value-to-goal (line-flux / enhancement predictions + the publication's validation story)
against effort and risk.

Item generation:
----------------

1. Create and move to branch of relevant type
2. Overwrite branch TODO.md with concise, 2-3 sentence problem summary + implementation path, scoped only to the relevant item, then publish to `origin`
3. Move to `main`, create 1 sentence summary of new item, then triage into existing TODO.md items and push tightly scoped `docs(todo)` commit to main

**NOTE:** If the user has written a detailed item summary directly into `TODO.md` on
main, fold it into a branch (steps 1-2 above), then slim it back to a one-line summary
on `main` once the branch exists.

## P1 - high value (physics accuracy + publication validation)

1. **Multilayer film-on-substrate.** ***VERY IMPORTANT*** Add sapphire as a crystalline material,
   not just an amorphous backing. Replace `al2o3` altogether. *Do this on main, not a separate branch.*
2. **Crystal mosaicity - measured-data validation.** MC route implemented; validate
   broadened line widths vs. a measured HOPG rocking-curve / EDS dataset
   (data-dependent). Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).
3. **Multilayer film-on-substrate - measured-data validation.** Model implemented;
   validate vs. a measured film-on-substrate dataset (data-dependent). Design:
   [`docs/multilayer-materials.md`](docs/multilayer-materials.md).

## P2 - medium (experiment match + usability)

1. **`crystals` Library.** Evaluate and, if found to be valuable, implement use of the `crystals`
   Python library, changing our syntax to match that of the library as needed, and reserving
   our hand-made database as only a fallback for those materials missing from theirs.
2. **Polars investigation.** Evaluate Polars for packaging large parameter-sweep metadata.
3. **pyelsepa / ELSEPA transport.** -> `feature/elsepa-port` Adapter landed + **validated** (C 2.19%,
   Si 4.42% max rel vs NIST); image now builds tarball-free from
   `github.com/eScatter/elsepa`. Remaining gate: the image/venv live outside the repo
   (`C:/dev/pyelsepa`), so the driver stays gated in CI. Tied to P2 #2.

## P3 - lower / exploratory

1. **Marimo/Altair follow-ups.** Core migration landed from `feature/marimo-transfer`;
   `marimo` notebook/plot cleanup & fixes remain.
2. **Grazing-incidence soft X-ray diffraction grating.** -> `feature/grazing-grating`.
   Dispersion scaffold implemented; next is grating reflectivity + detected-image model.

## Meta / cleanup

1. **Repo ownership & name change.** DONE. `cxr-mc` ownership perms given to Alex.
2. **dev/remote.py bugfix.** Right now, when user runs `uv run dev/remote.py start mote2 --follow`,
   they get an output stating the task was launched, but no progress bar. They must disconnect
   and rerun `uv run dev/remote.py attach` to see the progress.

## Long term features

1. Geant4 or similar integration to support high-energy electron beams
