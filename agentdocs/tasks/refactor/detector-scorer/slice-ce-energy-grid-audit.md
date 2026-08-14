# Slices C/E — where the photon-energy grids actually live

Prerequisite mapping for moving the grids onto `Detector.energy_bins` and
removing them from the scene objects. Line numbers against `main` @ `7094130`.

## The three names appear in 68 files. Only one of those is a scene object.

The grep is misleading, so the layers must be separated before any edit:

| Layer | Carrier | In scope for slice E? |
| --- | --- | --- |
| Catalog resolution result | `catalog.ScanSpec.E_grid_line` / `.E_grid_line_by_energy` / `.E_grid_brem` (`catalog.py:204-213`) | **no** — this is the *source*, and stays. It becomes the input to the detector's binnings instead of to `Sweep`. |
| **Scene object** | **`Sweep.E_grid_line`, `.E_grid_line_by_energy`, `.E_grid_brem`, `.e_grid_eV` (`sweep.py:299-302`)** | **yes — this is the whole of slice E** |
| Per-case record | `case["E_grid_line"]`, `case["E_grid_brem"]` (`sweep.py:758-759`) | no — case identity is hashed; changing it breaks bit-for-bit |
| Stored result record | `r["E_grid"]`, `r["E_grid_brem"]` (`store.py:147-148`) | no — "stored record layout unchanged" is an acceptance criterion |
| Artifact / catalog TOML | `materials.toml` (45 refs), `tests/data/material_catalog_golden.json` (147 refs) | no — the artifact store is explicitly out of scope |

So slice E deletes **four fields on one class** and repoints their readers. The
other layers keep their names.

## Resolution order, as implemented

Line grid, `_line_grid_for_energy` (`sweep.py:524-534`):

1. `sweep.E_grid_line` — explicit
2. `sweep.e_grid_eV` — deprecated alias for the same thing
3. `sweep.E_grid_line_by_energy[energy_keV]` — per-beam-energy mapping; a
   missing key is a hard error, not a fallback
4. `cp["E_grid"]` — the per-material catalog default, i.e.
   `CrystalSpec.E_grid` (`catalog.py:157`) via `crystal_params`

Brem grid (`sweep.py:586-598`):

1. `sweep.E_grid_brem` — explicit, used exactly as given
2. otherwise derived: start at the resolved line grid's first bin (or the
   minimum start across the per-energy mapping, or the material default's
   first bin), then `np.arange(start, max_beam_eV + 50.0, 50.0)`

The derived brem grid is per-case: it extends to *that case's* beam energy,
because bremsstrahlung cuts off at the particle energy.

## Writers of the scene-object fields

Only three, all in one file — `campaign/config.py` copies the catalog's
`ScanSpec` grids straight onto the `Sweep`:

| Site | Function |
| --- | --- |
| `config.py:120-122` | `material_grid` — a mapping projection for display |
| `config.py:206-208` | the main `Sweep` builder |
| `config.py:283-285` | the tilt-study `Sweep` builder |

That narrowness is the good news for slice C: the catalog → scene hand-off is a
single three-line pattern repeated three times, so redirecting it into
`Detector(energy_bins=...)` is a contained change.

## Open question resolved: does catalog resolution have to supply `energy_bins`?

Yes, and the documented order already accommodates it.
`docs/repo-design/configuration-resolution.md:55-63` states the run resolution
order as:

```text
CLI context -> catalog profile -> material override -> named beam/detector/grid
            -> fidelity preset -> explicit run overrides -> cases and identity
```

The detector and the grid are already **one stage** — "named
beam/detector/grid". Moving `energy_bins` onto `Detector` therefore does not
perturb the documented precedence; it merges two things that already resolve
together. The same page (line 52-53) notes energy-grid references are resolved
for the selected profile before a material sweep is built, which is the stage
that must now populate the detector's binnings.

No design change is needed to satisfy the open question, and none is made.

## Why the line/brem split is physical, not incidental

To be documented on the object per the RFC, sourced from `sweep.py:291-298`:

- **line binning** — fine and *narrow*. This is where the coherent lines are
  evaluated, via the expensive `sinc^2`. The lines are kinematically capped at
  a few keV, so the binning need not extend past ~4 keV.
- **brem binning** — coarse and *wide*. The bremsstrahlung is smooth and cheap
  to evaluate, and must reach 20-40 keV (the beam energy) to model the full
  measured spectrum. Using the line binning for it would inflate the line cost
  by an order of magnitude for no resolution gain.

The split exists because the two spectra have opposite cost/extent profiles.
That rationale currently lives in a comment on a `Sweep` field and is the
folklore the RFC wants moved onto the object.

## Tests that pin this behavior

| File | What it pins |
| --- | --- |
| `tests/scan/test_sweep.py` (47 refs) | grid resolution, precedence, the `e_grid_eV` alias |
| `tests/materials/test_material_catalog.py` (62 refs) | catalog-side grid parsing/validation |
| `tests/data/material_catalog_golden.json` (147 refs) | the catalog golden — **regenerating this is a `catalog-golden` job, and it must not change** |
| `tests/energy-grid/*` | derivation, artifacts, gc, provenance — untouched by this task |
| `tests/scan/test_run.py`, `tests/checkpoint/test_slim.py` | case-level and checkpoint-level grid keys |

## Bit-for-bit constraint

`case["E_grid_line"]` / `case["E_grid_brem"]` feed the per-case content key and
`parameter_sha256`. Slice C/E must produce byte-identical case dicts: the
resolution *result* has to be the same array, from the same branch order, for
every existing input. The safest shape is to keep `_line_grid_for_energy`'s
branch structure exactly and change only where it reads its inputs from.
