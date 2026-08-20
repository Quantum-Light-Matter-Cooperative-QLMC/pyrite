# Analysis surface review: notebooks, exports, and the HyperSpy question

Status: review only. No code changed. Not authority for any branch; promote
accepted items into GitHub Issues before acting.

Two questions:

1. How does PyRITE's notebook/analysis surface compare with Geant4, FLUKA,
   PENELOPE, abTEM, and adjacent Python scientific codebases?
2. Would adopting HyperSpy simplify the analysis apps?

Short answers: PyRITE's *data* layer is at or above peer standard, its
*presentation* layer is ahead of all of them, and its *handoff* layer (what
leaves the machine and reaches a collaborator or a paper) is behind all of
them. HyperSpy should not become a dependency — PyRITE's sweep output violates
HyperSpy's core data-model invariants — but RosettaSciIO is worth considering
as an optional interchange writer.

---

## 1. Current state

### 1.1 What exists

There are no working Jupyter notebooks. Interactive analysis is four marimo
apps stored as importable package modules:

| App | Path | Cells | Role |
|---|---|---|---|
| analysis | `src/pyrite/apps/analysis_app.py` | 35 | checkpoint comparison |
| trace | `src/pyrite/apps/trace_app.py` | 17 | direct transport/lattice viewer |
| validation | `src/pyrite/apps/validation_app.py` | 24 | validation studies |
| scan | `src/pyrite/apps/scan_app.py` | 7 | sweep runner |

Non-UI logic is factored out: `apps/analyze.py` (520 lines, pure functions),
`apps/analysis_ui/` (controls, axes, data, models, views), and
`pyrite.results`. The apps import from these; they do not compute.

Adjacent surfaces:

- `src/pyrite/apps/anchor_figures.py` (1612 lines) — headless publication
  figure builder for the Zhai anchor, with caching and a validation table.
- `checks/*.py` — standalone physics anchors printing `PASS`/`FAIL`.
- `checks/cxr_analysis_feranchuk.ipynb` — the only `.ipynb`, explicitly legacy
  and frozen by the `notebook-workflow` skill contract.

### 1.2 Data layer

`checkpoints/<stem>/meta.json` carries `cxr.checkpoint-manifest.v2` with a
nested `cxr.dataset-identity.v1` block: `material`, `fidelity`, `variant`,
`catalog_profile`, `parameter_sha256`, and the full `resolved_parameters`
tree. `src/pyrite/checkpoints/_checkpoint_io.py` writes HDF5 (`dump`, line
153) and reads HDF5 plus three legacy pickle generations (`load`, line 184).

Payload shape, measured on `checkpoints/hopg@all_mats-4fbb008fe28a/line.pkl`:

```
dict[geometry_label -> dict[E0_keV -> record]]
  27 geometry keys x 4 beam energies = 108 possible, 99 present
  record: E_grid (float64), spec (float32), E_pk, fwhm, eta,
          hit_frac, scale, source_current_na, case
```

Two properties of this layout matter later:

- **The navigation space is sparse.** 99 of 108 grid cells exist.
- **The signal axis is not shared.** Across the 99 records there are four
  distinct `E_grid` definitions — lengths 864/998/1098/1231, spanning
  10–2600/3000/3300/3700 eV, one per beam energy. Each is internally uniform;
  none is common to the whole set. Bremsstrahlung lives on its own separate
  grid (`E_grid_brem`).

### 1.3 Testing

`tests/notebooks/` holds 2388 lines. `analysis_app/test_app.py` asserts
*structural* invariants against the app source: every `mo.ui` value is read
downstream of creation, no private names cross cell boundaries, the material
menu owns its checkpoint-directory dependency, tab structure and action names
are pinned, control rows wrap at narrow widths, and the file is
mojibake-free. This is static analysis of notebook code as a regression gate.
No peer toolkit reviewed here does this.

---

## 2. Peer comparison

| Toolkit | Analysis path | Artifact | Reproducibility mechanism |
|---|---|---|---|
| Geant4 | `G4AnalysisManager` → ROOT/HDF5/CSV/XML ntuples; analysis in ROOT macros, `uproot`, or RDataFrame | flat TTree, `.h5` | examples ship reference `.out` diffed in CI; geant-val portal for literature comparison |
| FLUKA | Flair (Python/Tkinter) + gnuplot back end; Fortran mergers in `src/tools/` | binary USRBIN → `.dat` | none automated; benchmarks are documents |
| PENELOPE / penEasy | Fortran tallies → column ASCII with per-bin 2σ; gnuplot scripts | `.dat` | the uncertainty column *is* the contract |
| abTEM | Dask-lazy `Measurement` objects, `.to_zarr()` / `from_zarr()`, `.show()` | `.zarr` | docs are executed notebooks, version-tagged |
| OpenMC | `StatePoint` → `Tally.get_pandas_dataframe()` | `statepoint.*.h5` | version-tagged notebook gallery + hashed regression suite |
| HyperSpy / py4DSTEM | Signal objects over HDF5/Zarr/EMD | `.hspy`, `.zspy`, `.emd` | CI-executed demo notebooks, Binder |
| **PyRITE** | marimo apps over HDF5 checkpoints; `analyze.py` reusable headlessly | `.html` snapshot | static app tests, validation ledger, `checks/` scripts |

Structural observations:

- Every peer hands the user a **data object** (TTree, DataFrame, Measurement,
  Signal). PyRITE hands the user an **HTML page**. That is the single largest
  divergence.
- Only PyRITE and abTEM/OpenMC have a first-class Python analysis API at all;
  FLUKA's plotting logic is unreachable from a script, which is the failure
  mode PyRITE's thin-app rule already avoids.
- PyRITE's `parameter_sha256` identity is stronger than anything in the table.
  OpenMC records inputs; it does not canonicalise them to a digest.
- PyRITE's validation ledger is structurally the geant-val idea, versioned
  in-repo with sign-off gating — better than FLUKA's benchmark-PDF model.

---

## 3. Strengths worth protecting

1. **Notebooks as package modules.** marimo `.py` files under `src/` are
   subject to lint, typecheck, import-linter contracts, and unit tests. Geant4
   ROOT macros, Flair plot definitions, and `.ipynb` walkthroughs are all
   unreviewable by comparison. The reactive DAG also removes the
   execution-order bug class that dominates public notebook corpora.
2. **Thin-app discipline.** `analyze.py` and `analysis_ui/views/` mean a
   headless caller can already reach the numbers the UI displays. This is the
   precondition for every recommendation in §4 and it is already paid for.
3. **Canonical dataset identity.** Digest plus resolved parameters, schema-
   versioned.
4. **Validation ledger with human sign-off.** Correct separation between
   "a check passed" and "a claim is verified".

---

## 4. Gaps, ranked

### G1 — The shareable export is not the view you were looking at

`src/pyrite/apps/export.py:29` builds `marimo export html` with no `--`
arguments, so the exported page always renders default selector state.
`src/pyrite/apps/analyze.py:411` (`_smoke_command`) *does* pass
`-- --material <material>`. The CI smoke path is parameterised; the scientific
handoff path is not. `pyrite app analysis export hopg-analysis` therefore
produces a file whose name asserts something its content may not contain.

Peer contrast: this is what papermill solved for Jupyter; OpenMC sidesteps it
by making the notebook read an explicit statepoint path.

Fix: thread selector state (material, face, profile, emission, active view)
through `mo.cli_args()` into the export argv. Small change, highest value.

### G2 — Exported artifacts carry no machine-readable provenance

`docs/guides/analysis-tutorial.md` §4 instructs the *human* to record the
checkpoint stem, identity digest, PyRITE revision, and coordinates beside
exported figures. That is a procedure, not a mechanism, and it will be
skipped. `analyze.py:_stem_dataset_identity` already has the identity in hand
at load time.

Fix: a pinned provenance cell rendering digest, stem, `pyrite.__version__`,
git revision, and export timestamp; the same emitted as an HTML `<meta>`
block. PyRITE is the only toolkit here with a canonical digest and the only
one not stamping it.

### G3 — `checks/` is a dead-end output format

`multilayer_check.py`, `mosaic_mc_check.py`, etc. print margins and
`PASS`/`FAIL` to stdout, produce no artifact, and are outside the
`test-suite` selectors. The mapping from check to ledger id lives in a
hand-maintained table in `checks/README.md`.

Peer contrast: Geant4 diffs example reference outputs in CI; OpenMC hashes
regression results; geant-val stores every comparison in a queryable database.

Fix: each check emits one JSON record — ledger id, measured value, reference
value, tolerance, verdict, code revision — into `results/checks/`. A
`pyrite-dev checks` aggregator renders `docs/validation/status-summary.md`
from those records rather than from prose. That converts "a successful check
is evidence" into evidence that can be dated and diffed.

### G4 — One export format, and it is the least useful one

Static HTML only. Missing:

- `marimo export html-wasm`, which ships the *interactive* app with no Python
  install — the closest thing to a paper supplement, and something no peer in
  §2 can do.
- Data export of the plotted series. Table stakes everywhere else: abTEM
  `.to_zarr()`, OpenMC `get_pandas_dataframe()`, Flair's gnuplot `.dat`,
  penEasy's ASCII columns. Today a collaborator who wants a curve must re-run
  PyRITE.

Fix: `--format {html,html-wasm,data}` plus a "download plotted data" control
in the app.

### G5 — No executed-example surface in the docs

`docs/conf.py` loads `myst_parser` and autodoc only — no myst-nb, nbsphinx, or
sphinx-gallery. Every code block in `docs/guides/*.md` is unverified prose
that rots silently. abTEM's walkthrough, OpenMC's `openmc-notebooks`, and
HyperSpy's demos are all executed.

Fix (cheapest consistent with the architecture): keep guides as prose but add
a doctest pass over the fenced blocks; optionally embed the parameterised
marimo exports as the gallery.

### G6 — Hidden machine-local state in the analysis entry point

`analyze.py:get_default_material` / `set_default_material` persist a default
into Click's app directory, and `DEFAULT_CHECKPOINT_DIR` resolves through
workspace configuration. Two users running the same documented command can
get different figures. Acceptable for interactive launch; not acceptable for
`export`, which should require explicit identity and fail loudly rather than
fall back.

### G7 — Housekeeping

- `notebooks/` at repo root now contains nothing but `__pycache__` trees that
  shadow `analysis_ui` module names. Leftover from the move into
  `src/pyrite/apps/`. Delete.
- `TODO.md` used to point at the stale `notebooks/analysis_ui/controls` path;
  moot now that `TODO.md` is a short pointer stub (backlog moved to GitHub
  Issues) with no per-item bodies left to go stale.
- Checkpoint payloads are named `line.pkl` / `brem.pkl` while actually being
  HDF5 (or zstd-framed pickle for legacy generations). A collaborator with
  `h5py` will never think to open a `.pkl`. Writing new checkpoints as `.h5`
  costs one line in the writer and makes the store legible to outside tooling.

---

## 5. HyperSpy evaluation

### 5.1 What HyperSpy offers

HyperSpy 2.x is a framework for multi-dimensional datasets that separates
*navigation* dimensions from *signal* dimensions. A `Signal1D` of shape
`(20, 10 | 30)` is a 20×10 navigation grid of 30-channel spectra, with an
`AxesManager` carrying name, units, scale, and offset per axis. On top of this
it provides interactive plotting with a navigator, `map()` over the navigation
space, Dask-backed lazy signals, model fitting, multivariate decomposition,
and IO through RosettaSciIO. Since the 2.0 modularisation, EDS and EELS live
in the separate `exspy` extension, which supplies X-ray line databases,
per-line Gaussian model construction, background-window subtraction, detector
energy-resolution calibration, and Cliff-Lorimer / zeta-factor quantification.

On paper this is an excellent match: PyRITE produces spectra over a
multi-dimensional parameter sweep, for electron-beam X-ray emission, and the
apps spend most of their code on navigating that sweep.

### 5.2 Why the data model does not fit

The measured checkpoint layout in §1.2 violates two HyperSpy invariants.

**(a) The navigation space must be a dense rectangular array.** HyperSpy
signals are backed by a single ndarray; navigation dimensions are ndarray
dimensions. The measured checkpoint has 99 of 108 cells populated. Sweeps are
built from case lists and pruned, so partial occupancy is normal, not an
artefact. Loading into HyperSpy requires NaN-padding to the full Cartesian
product, after which every reduction, fit, and decomposition must be
NaN-aware, and the distinction between "not simulated" and "simulated to
zero" is lost from the array (recoverable only from a side mask).

**(b) All navigation positions must share one signal axis.** `AxesManager`
holds one signal axis per signal. The measured checkpoint has four distinct
energy grids — 864/998/1098/1231 channels over 10–2600/3000/3300/3700 eV, one
per beam energy, because the grid is sized to the physics of each case.
Putting these in one HyperSpy signal requires resampling onto a common grid.
For line spectra whose peak positions are the physical observable, that is a
lossy operation performed for the convenience of a container. Cross-beam-
energy comparison — the primary thing the analysis app is *for* — is exactly
the operation that forces it.

HyperSpy's escape hatches do not help here:

- Non-uniform `DataAxis` allows an arbitrary axis vector, but still one vector
  for the whole signal. It solves "my axis isn't evenly spaced", not "my axis
  differs per navigation position", and it disables a set of operations
  (rebinning and FFT-family methods) as a side effect.
- Ragged signals allow per-position variable length, but explicitly do not
  support `isig` slicing, transposition, or plotting — i.e. all of the reasons
  one would adopt HyperSpy.

**(c) Two signals, not one.** Lines (`E_grid`) and bremsstrahlung
(`E_grid_brem`) are on deliberately different grids. One HyperSpy signal
cannot hold both, so the "everything in one object" benefit halves before any
of the above.

**(d) Navigation coordinates are not materialised.** Checkpoint keys are human
labels (`"HOPG 1um pol=15 az=100 footprint=5x5mm ne=300/150"`), not coordinate
tuples. Building an `AxesManager` means parsing labels or re-deriving
coordinates from `case` — work that exists either way, and that is not made
easier by HyperSpy.

### 5.3 Dependency cost

`hyperspy==2.4.0` resolves cleanly on Python 3.14 alongside the current pins
(verified with `uv pip compile --python-version 3.14`; numpy resolves to
2.5.2, satisfying the `>=2.4.6` floor). It adds **18 packages** not currently
in the environment:

```
cloudpickle  dask  flexcache  flexparser  fsspec  hyperspy
importlib-metadata  locket  mpmath  natsort  partd  pint
prettytable  python-box  rosettasciio  sympy  traits  zipp
```

Notable entries: `traits` (Enthought, C extension, its own release cadence),
`sympy` + `mpmath`, `dask`, and `pint`. For a package whose value proposition
is a container class, that is a large surface — and `traits` and `pint` in
particular introduce their own object models that would leak into any code
touching HyperSpy signals.

### 5.4 Feature-by-feature: what would actually be gained

| HyperSpy feature | PyRITE status | Verdict |
|---|---|---|
| Navigation over sweep dimensions | `results/selection.py` (453 lines), `analysis_ui/` | already built, and handles sparse sweeps HyperSpy cannot |
| Interactive plotting | marimo + Altair, reactive, browser-native | HyperSpy's matplotlib-widget plotting is a regression here, not an upgrade |
| Peak finding / line metrics | `results/metrics.py` (scipy `find_peaks`) | equivalent for current needs |
| Model fitting (Gaussian per line + background) | not present | **genuine gap HyperSpy/exspy would fill** |
| Lazy/out-of-core arrays | not needed at current checkpoint sizes | no |
| PCA/NMF decomposition | not present, not obviously needed | speculative |
| EDS detector response | `pyrite.detectors` — physics-validated, ledgered | PyRITE's is the more defensible one; exspy's is empirical calibration |
| IO / interchange | HDF5 + digest manifest | **RosettaSciIO is interesting on its own** — see §5.6 |

The only unambiguous gain is model fitting. That does not require HyperSpy:
`scipy.optimize` or `lmfit` fits a sum of Gaussians plus a background over an
arbitrary energy grid, with no container constraints and no 18-package tail.

There is also a physics argument against exspy's EDS layer specifically.
`exspy` calibrates detector resolution empirically by fitting known lines, and
quantifies via Cliff-Lorimer. PyRITE's detector response is a forward physical
model with ledger entries (`detector-eaglexo`, `detector-line-broadening`) and
a validation methodology. Introducing a second, empirically-calibrated
detector model into the same codebase would create exactly the ambiguity the
validation ledger exists to prevent.

### 5.5 Verdict

**Do not adopt HyperSpy as a dependency.** The fit is superficially excellent
and structurally poor: PyRITE's sweeps are sparse and its signal axis varies
per navigation position, which are the two things HyperSpy's array-backed
model cannot express without lossy padding and resampling. The features that
would be gained are either already built (navigation, selection, plotting,
peak metrics), better solved directly (`scipy`/`lmfit` for line fitting), or
in tension with the validation ledger (exspy's empirical detector model).

The instinct behind the question is right, though: PyRITE's sweep output *is*
a labelled N-dimensional array problem, and `results/selection.py` is
partially a re-implementation of one. If that ever becomes painful, the
container to evaluate is **xarray**, not HyperSpy — it handles named
coordinates and missing cells natively, is a much lighter dependency, and
carries no domain model to conflict with the ledger. Even then, the varying
per-case energy grid pushes toward a `DataTree`/per-E0-group layout rather
than one array, so this is a "revisit if selection code keeps growing" item,
not an action.

### 5.6 The narrow option worth keeping: RosettaSciIO

Separable from HyperSpy proper and relevant to G4 and the deferred
result-format-interchange work. `rosettasciio` is HyperSpy's IO package,
installable standalone; with the `hdf5`/`zspy` extras it reads and writes
`.hspy` (HDF5), `.zspy` (Zarr), NCEM `.emd`, and NeXus `.nxs`. Its own
dependency set is much smaller than HyperSpy's — `dask[array]`, `numpy`,
`pint`, `python-box`, `pyyaml`, `python-dateutil`, plus `h5py`.

The case for it: PyRITE's audience is electron microscopists. A
`pyrite checkpoint export --format hspy` (or `emd`, or `nxs`) would let a
collaborator open PyRITE-simulated spectra in the tooling they already run,
next to their measured data, without installing PyRITE. That is precisely the
comparison the Zhai anchor work is about, and it is the deliverable that G4
identifies as missing.

The case against: it is a one-way, per-case export (one signal per beam
energy, to respect the shared-axis constraint), it cannot round-trip the full
sparse sweep, and `dataset_identity` would have to be carried in the signal
metadata tree where nothing enforces its schema. It is interchange, never a
replacement for the checkpoint.

Recommendation: treat as an **optional extra** (`pyrite[interchange]`),
gated behind an explicit CLI subcommand, exporting one signal per (material,
geometry, beam energy) with the identity digest written into
`metadata.General` and `original_metadata`. Sequence it after G1/G2 — an
export path is only worth building once exports carry provenance.

---

## 6. Sequenced recommendations

1. **G1 + G2 together.** Parameterise the export and stamp provenance into it.
   One change to `export.py` and one new cell. Unblocks everything else.
2. **G7 housekeeping.** Delete stale `notebooks/`, write new checkpoints as
   `.h5` (the `TODO.md:143` path issue is now moot — see §4 G7).
3. **G3.** Machine-readable check records; generate the validation status
   summary from them.
4. **G4.** `--format {html,html-wasm,data}` and in-app data download.
5. **G5.** Doctest the guide code blocks.
6. **G6.** Require explicit identity for `export`.
7. **The one rework worth doing.** Collapse "launch", "export", and "figure
   build" onto a single parameterised render path. Three exist today:
   interactive marimo, `marimo export html` (unparameterised), and
   `anchor_figures.export_all_figures` (headless, well-tested). A
   `pyrite app analysis render --identity <digest> --view energy
   --format {html,svg,csv}` that both the app and `anchor_figures` route
   through yields reproducible exports, one place to stamp provenance,
   publication and interactive figures that cannot disagree, and a CI target
   catching figure drift. That is the property Geant4, FLUKA, and PENELOPE
   users have never had, and it is one refactor away.
8. **RosettaSciIO interchange** (§5.6), optional extra, only after 1.
9. **HyperSpy: no.** Revisit xarray only if `results/selection.py` keeps
   growing.

---

## References

- [Geant4 Analysis Manager classes](https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/Analysis/managers.html)
- [Flair GUI](https://fluka.cern/documentation/running/flair-gui) ·
  [FLUKA package structure](https://fluka.cern/documentation/installation/fluka-package-file-structure)
- [abTEM walkthrough](https://abtem.readthedocs.io/en/latest/user_guide/walkthrough/walkthrough.html) ·
  [scan and detect / `to_zarr`](https://abtem.readthedocs.io/en/main/user_guide/walkthrough/scan_and_detect.html)
- [openmc-notebooks](https://github.com/openmc-dev/openmc-notebooks) ·
  [`openmc.Tally.get_pandas_dataframe`](https://docs.openmc.org/en/stable/pythonapi/generated/openmc.Tally.html)
- [HyperSpy signal basics and axes](https://hyperspy.org/hyperspy-doc/current/user_guide/axes.html) ·
  [eXSpy EDS user guide](https://hyperspy.org/exspy/user_guide/eds.html)
- [RosettaSciIO supported formats](https://hyperspy.org/rosettasciio/supported_formats/index.html)
- [marimo vs Jupyter](https://marimo.io/features/vs-jupyter-alternative)
