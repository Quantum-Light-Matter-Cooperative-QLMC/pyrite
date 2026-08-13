# Case payload inventory

Slice B inventory for the typed `Case` boundary. `build_cases` emits 31 keys
unconditionally and up to 12 conditionally. Six additional legacy controls are
read by `run_case` but are not emitted by `build_cases`; the compatibility
constructor must accept them during the D7 mapping-support window.

`EnergyGrid` below means either a uniform `(start_eV, stop_eV, step_eV)` tuple
or an exact one-dimensional numeric array. `Composition` means
`list[tuple[str, float]]`, with number density in atoms/angstrom^3. `HKL` and
`UVW` mean integer 3-tuples.

## Emitted fields

The table order is the insertion order in the legacy dict. `Conditional` means
the key is absent, rather than present with a default value, when its condition
is false. That absence is identity-significant and `Case.to_dict()` must retain
it exactly.

| # | Key | Type | Unit / meaning | Presence |
| ---: | --- | --- | --- | --- |
| 1 | `name` | `str` | Display/configuration label | Always |
| 2 | `crystal` | `str` | Catalog crystal key | Always |
| 3 | `composition` | `Composition` | Element number densities, atoms/angstrom^3 | Always |
| 4 | `hkl_list` | `list[HKL]` | Coherent reflection indices | Always |
| 5 | `B_ang2` | `float` | Isotropic Debye-Waller factor, angstrom^2 | Always |
| 6 | `E0_keV` | `float` | Incident electron kinetic energy, keV | Always |
| 7 | `thickness_ang` | `float` | Film/slab thickness, angstrom | Always |
| 8 | `crystal_width_mm` | `float \| None` | Full transverse width, mm; `None` is infinite | Always |
| 9 | `crystal_height_mm` | `float \| None` | Full transverse height, mm; `None` is infinite | Always |
| 10 | `beam_fwhm_mm` | `float \| None` | Beam x-plane Gaussian FWHM, mm | Always |
| 11 | `bunch_charge_pc` | `float` | Bunch charge, pC | Conditional: charge or repetition rate diverges from legacy defaults |
| 12 | `rep_rate_hz` | `float` | Repetition rate, Hz | Conditional: paired with `bunch_charge_pc` |
| 13 | `beam_fwhm_y_mm` | `float \| None` | Beam y-plane Gaussian FWHM, mm | Conditional: differs from x-plane FWHM |
| 14 | `energy_spread_frac` | `float` | Relative RMS energy spread, dimensionless | Conditional: configured |
| 15 | `long_shape` | `str` | Legacy longitudinal distribution shape | Conditional: legacy bunch length or offsets configured |
| 16 | `bunch_length_fs` | `float` | Legacy bunch FWHM/duration parameter, fs | Conditional: configured |
| 17 | `long_offsets_fs` | `tuple[float, ...]` | Explicit electron time offsets, fs | Conditional: configured |
| 18 | `longitudinal_distribution` | `dict[str, object]` | Resolved longitudinal policy; time fields in fs, target energy in eV, wavelength in angstrom | Conditional: declarative policy configured |
| 19 | `transverse_distribution` | `dict[str, object]` | Resolved x/y Twiss second moments; positions mm, slopes rad, emittance mm*rad | Conditional: declarative policy configured |
| 20 | `E_grid` | `EnergyGrid` | Legacy line-energy grid, eV | Always |
| 21 | `E_grid_line` | `EnergyGrid` | Line-spectrum energy grid, eV | Always |
| 22 | `E_grid_brem` | `EnergyGrid` | Bremsstrahlung energy grid, eV | Always |
| 23 | `theta_obs_rad` | `float` | Detector observation polar angle, rad | Always |
| 24 | `tilt_deg` | `float` | Target polar tilt, deg | Always |
| 25 | `tilt_azim_deg` | `float` | Target tilt azimuth, deg | Always |
| 26 | `groove_spacing_ang` | `float` | Grating groove spacing, angstrom | Conditional: grooved target |
| 27 | `coherent_emission` | `Literal[True]` | Also evaluate coherent segment sum | Conditional: coherent/both emission; absent means false |
| 28 | `xray_dispersion` | `Literal["refractive"]` | In-medium line kinematics | Conditional: non-vacuum; absent means vacuum |
| 29 | `beam_uvw` | `UVW \| None` | Direct-lattice beam direction | Always |
| 30 | `surface_hkl` | `HKL \| None` | Reciprocal surface normal; exclusive with `beam_uvw` | Always |
| 31 | `mosaic_fwhm_rad` | `float \| None` | Analytic mosaic FWHM, rad | Always |
| 32 | `mosaic_mc_fwhm_rad` | `float \| None` | Exact MC mosaic FWHM, rad | Always |
| 33 | `mosaic_mc_nodes` | `int` | Gauss-Hermite nodes per mosaic axis | Always |
| 34 | `abs_layers` | `list[tuple[float, float, Composition]] \| None` | Layer top/bottom depths in angstrom plus composition | Always |
| 35 | `layer_radiators` | `list[dict[str, object] \| None] \| None` | Per-layer crystal/HKL/B/orientation records | Always |
| 36 | `brem_file` | `str \| PathLike[str] \| None` | Optional external bremsstrahlung spectrum path | Always |
| 37 | `Ne` | `int` | Line-transport electron count | Always |
| 38 | `Ne_brem` | `int` | Bremsstrahlung-transport electron count | Always |
| 39 | `seed` | `int` | Counter-based transport RNG seed | Always |
| 40 | `spec_chunk` | `int \| None` | Line-kernel segment chunk cap | Always |
| 41 | `brem_chunk` | `int \| None` | Bremsstrahlung-kernel segment chunk cap | Always |
| 42 | `dtheta_obs_rad` | `float` | Detector polar acceptance, rad | Always |
| 43 | `domega_sr` | `float` | Detector solid angle, sr | Always |

Nested resolved-policy records retain the exact dictionaries produced by
`dataclasses.asdict`; they must not import their campaign dataclass types into
`pyrite.montecarlo`. A layer radiator contains `crystal`, `hkl_list`, `B_ang2`,
`beam_uvw`, `surface_hkl`, and optional `azimuth_rad`.

## Read-only legacy controls

These are accepted by runner consumers and handwritten tests but never emitted
by `build_cases`.

| Key | Type | Unit / default | Consumer |
| --- | --- | --- | --- |
| `azimuth_rad` | `float` | Crystal in-plane rotation, rad; `0` | line kernel and detector geometry |
| `recip_miscut_rad` | `tuple[float, float] \| None` | Reciprocal-vector polar/azimuth miscut, rad; `None` | line kernel and detector geometry |
| `E_cut_lines_keV` | `float` | Line-transport cutoff, keV; `5` | transport and line kernel |
| `E_cut_brem_keV` | `float` | Brem-transport cutoff, keV; `1` | transport and brem kernel |
| `sinc_cutoff` | `float \| None` | Dimensionless line-shape window; exact when `None` | line kernel |
| `brem_step_eV` | `float` | Legacy single-grid brem spacing, eV; `10` | runner grid fallback |

## Non-schema compatibility observations

- `case_content_key` hashes every field except its historical denylist; dict
  insertion order does not affect the sorted JSON digest, but `to_dict()` still
  reproduces legacy insertion order for byte-level fixtures and pickles.
- `run_case` and helpers read the 49 fields above. Result/plot/checkpoint
  consumers additionally read emitted presentation fields (`name`, geometry,
  detector acceptance, pulse source, mosaicity, and `brem_file`).
- `results.selection` adds `thickness_fallback` only to a derived presentation
  copy. Checkpoint cleanup reads `content_key` only from a case-manifest entry.
  Neither belongs to the simulation `Case` schema.
- `source_current_na` belongs to a stored result record, not its nested case.
  Historical denylist-only names (`beam_current_na`, `catalog_profile`,
  `variant`, `fidelity`) are likewise not case inputs.
- No maintained notebook constructs a simulation case dict by hand. Checks use
  `build_cases` or call lower-level kernels directly. Handwritten partial case
  mappings remain in unit tests for runner/helper isolation.
- `apps.anchor_figures._resolved_case` mutates four fields on a dict returned by
  `build_cases`. It must switch to immutable replacement or explicitly request
  a compatibility dict when `build_cases` begins returning `Case`.
- GPU OOM retry currently mutates `spec_chunk`/`brem_chunk` in-place. The
  `run_case(Case | Mapping)` adapter must normalize a `Case` to a private dict
  before retry logic, while preserving existing Mapping behavior for D7.
