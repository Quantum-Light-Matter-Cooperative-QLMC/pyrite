# Import a GPT electron beam

PyRITE accepts native General Particle Tracer (GPT) `.gdf` **time-output**
snapshots and **screen** outputs through EasyGDF. Analytic beams remain the default. Imported records
supply position, direction, energy, and relative particle weight together.

## Export and inspect

In the GPT input deck, request time output, for example:

```text
tout(0, 1e-9, 5e-11);
```

Run GPT and inspect the native output directly:

```bash
gpt -o beam.gdf beam.in
pyrite beam gdf-times beam.gdf
```

If you do not yet have GPT output, save this small fixture generator as
`make_example_gdf.py`. It uses the same EasyGDF structure as PyRITE's tests:

```python
import easygdf
import numpy as np
from scipy.constants import electron_mass, elementary_charge

arrays = {
    "x": np.array([1e-5, -2e-5]),
    "y": np.array([3e-5, 4e-5]),
    "z": np.array([0.1, 0.1]),
    "Bx": np.array([0.01, -0.02]),
    "By": np.array([-0.01, 0.02]),
    "Bz": np.array([0.328, 0.328]),
    "m": np.full(2, electron_mass),
    "q": np.full(2, -elementary_charge),
    "nmacro": np.array([1.0, 9.0]),
}
blocks = [
    {
        "name": "time",
        "value": 1e-9,
        "children": [{"name": name, "value": value} for name, value in arrays.items()],
    }
]
easygdf.save("beam.gdf", blocks=blocks, creator="GPT")
```

Generate and inspect `beam.gdf`:

```bash
python make_example_gdf.py
pyrite beam gdf-times beam.gdf
pyrite beam gdf-inspect beam.gdf --time-s 1e-9
```

The equivalent Python workflow is:

```python
from pyrite.montecarlo.gdf import list_gdf_times, load_gdf_beam

print(list_gdf_times("beam.gdf"))
beam = load_gdf_beam("beam.gdf", time_s=1e-9)
print(beam.energy_keV)
```

Keep the native particle output, not a `gdfa` statistical summary or an ASCII
conversion. See the [GPT user manual](https://wiki.jlab.org/ciswiki/images/4/42/UserManual.pdf)
for `tout` and output-field controls, and the
[EasyGDF API](https://github.com/electronsandstuff/easygdf) for the raw GDF format
adapter. Select screen (`position`) groups explicitly with `gdf_screen_position_m`;
without this selector, mixed files use `time` groups. Screens additionally require
finite per-particle `t` crossing times in seconds. Their relative arrival times
are retained, referenced to the earliest crossing in the selected input.

| Array | Meaning and unit | Requirement |
|---|---|---|
| `x`, `y`, `z` | Lab position, meters | Required, finite |
| `Bx`, `By`, `Bz` | Velocity components $\beta_i=v_i/c$, dimensionless; **not magnetic fields** | Required; $0<|\boldsymbol\beta|^2<1$ |
| `m` | Particle mass, kg | Electron mass, relative tolerance $10^{-6}$ |
| `q` | Signed particle charge, C | Finite, negative, electron charge magnitude (relative tolerance $10^{-6}$) |
| `G` | Lorentz factor | Optional; compared against beta-derived gamma |
| `nmacro` | Represented electrons per macroparticle | Positive and finite when present; required for `gdf_charge` |
| `ID` | GPT particle identifier | Optional; row association is preserved without requiring an ID |

All consumed arrays must be one-dimensional and equally sized. The adapter
reads raw blocks, so EasyGDF's convenience routines cannot synthesize missing
required arrays. Optional `G` must agree with $1/\sqrt{1-|\boldsymbol\beta|^2}$
to relative tolerance $10^{-6}$ and absolute tolerance $10^{-9}$.
Energy is $(\gamma-1)mc^2$, converted to keV. By default, the selected distribution replaces
the analytic energy sweep; the maximum imported energy bounds automatic photon
grids and labels the single source-energy case. Each history keeps its own
energy. Existing energy ceilings and per-history transport cutoffs still apply.

## Choose the snapshot and coordinate origin

Multiple time blocks require `gdf_time_s`. A sole block may be selected without
it. The default absolute tolerance is `1e-15` seconds: `1e-9` matches
`1.0000000000000003e-9`. No nearest-time fallback occurs. Zero tolerance requests
exact equality. Requested times and tolerances must be finite and non-negative.
No match or multiple matches raises an error listing times and particle counts.

`gdf_z_origin_m` is required: it explicitly places the target origin at
`(0, 0, gdf_z_origin_m)` in GPT lab coordinates. GPT x/y axes are PyRITE lab x/y
axes. No transverse recentering or inferred origin occurs. The adapter:

1. Subtracts the specified lab-z origin.
2. Applies PyRITE's existing target-tilt rotation to positions **and** velocities.
3. Intersects each individual ray with the sample entrance plane `z=0`.
4. Converts positions to angstroms and retains the signed flight-time offset.

Projection is a signed, field-free extrapolation of the snapshot. It is not
beamline or space-charge tracking. A snapshot already on the untilted entrance
keeps its transverse positions exactly. Rays must point into the material after
rotation. A projected ray outside a finite target counts as a missed incident
history, as for the analytic source. Only flat slab/stack entrances and
incoherent emission are currently supported.

## Configure normalization

`pyrite_current` retains PyRITE's existing configured current normalization
(`bunch_charge_pc` × `rep_rate_hz`, or the existing analysis-current fallback).
Positive `nmacro` supplies relative sampling probabilities; missing `nmacro`
means uniform probabilities. Rescaling all `nmacro` values leaves this mode's
current and phase-space probabilities unchanged.

`gdf_charge` retains the absolute total represented charge:

$$
Q_{\mathrm{signed}}=\sum_i q_i n_i,\qquad
|Q|=\sum_i |q_i|n_i,\qquad I=|Q|f_{\mathrm{rep}}.
$$

It requires positive `nmacro` and a finite, strictly positive resolved shared
`rep_rate_hz`. The existing default is 5000 Hz; an inherited profile beam rate
also applies. `bunch_charge_pc` is derived as `|Q| × 1e12`, replacing any
configured charge in this mode. Charge alone cannot
determine an average current: identical bunches delivered at different rates
produce different rates of photons. A 1 pC bunch at 1 MHz gives 1 µA.

The existing estimator stores yields per incident electron. Complete records
are sampled with replacement using probabilities `nmacro / sum(nmacro)`;
**no additional `nmacro` contribution multiplier** is applied. Absolute charge
and repetition rate enter the existing read-time current conversion once.
This applies even when the history count equals the input record count.
Sampling uses child stream 6 of the existing NumPy `SeedSequence(seed)` scheme;
the same seed, snapshot, and requested count reproduce the same tuples. The
transport RNG retains its separate stream.

## Catalog example

Add this named beam and profile to your existing material catalog (retain its
crystal, material, and standard-profile tables):

```toml
[beams.gpt_import]
source = "gpt_gdf"
gdf_path = "beam.gdf"
gdf_time_s = 1.0e-9
gdf_time_tolerance_s = 1.0e-15
gdf_z_origin_m = 0.09844470106002952
gdf_normalization = "gdf_charge"
rep_rate_hz = 1.0e6

[profiles.gpt_import]
materials = ["hopg"]
beam = "gpt_import"
emission = "incoherent"
```

The example origin is an explicit placement choice; choose an origin appropriate to
**your** beam and target. Catalog GDF paths resolve
relative to the catalog file. Inline `[profiles.NAME.beam]` tables accept the
same fields. Run this profile with `pyrite run gpt_import -m hopg`.

For configured-current normalization, replace the final normalization settings
in `[beams.gpt_import]` with:

```toml
gdf_normalization = "pyrite_current"
bunch_charge_pc = 1.0
rep_rate_hz = 5000.0
```

The removed legacy `gdf_repetition_rate_hz` field is rejected: replace it
with `rep_rate_hz`. Do not combine the GDF source with
analytic spot, energy-spread, Twiss, or longitudinal-distribution settings.

## CLI example

Create a reusable beam, attach it, and run the profile:

```bash
pyrite beam gdf-inspect beam.gdf --time-s 1e-9
pyrite beam create gpt_beam \
  --source gpt_gdf --gdf-path beam.gdf \
  --gdf-time-s 1e-9 --gdf-time-tolerance-s 1e-15 \
  --gdf-z-origin-m 0.1 \
  --gdf-normalization gdf_charge --rep-rate-hz 5000
pyrite profile set standard --beam gpt_beam
pyrite run standard -m hopg
```

The time, physical target origin, and repetition rate are examples. Creating or
editing the beam validates the selected particle block before the atomic catalog
write. CLI paths resolve relative to the working directory and are saved as
absolute paths. `beam show gpt_beam` displays its source and GDF fields;
`beam set ... --dry-run` validates and prints a diff without saving.

For configured-current normalization:

```bash
pyrite beam create gpt_current \
  --source gpt_gdf --gdf-path beam.gdf --gdf-time-s 1e-9 \
  --gdf-z-origin-m 0.1 --gdf-normalization pyrite_current \
  --bunch-charge-pc 1 --rep-rate-hz 5000
pyrite profile set standard --beam gpt_current
pyrite run standard -m hopg
```

GDF settings belong to named beams; `run` accepts no source/GDF overrides.
GDF beams require local execution; remote file staging and developer `--nsys`
re-execution remain unsupported. File-content hashes join case and dataset
identities; mutation after case construction is rejected before transport.
Changing the shared repetition rate changes normalization and dataset identity,
while sampled particle records remain identical for the same seed. Beam names
and labels do not affect physical parameter hashes.

Switching to `--source analytic` removes GDF fields. Switching an analytic beam
to `gpt_gdf` removes its old analytic distribution fields; explicitly combining
analytic distribution options with the GDF source is an error. Setting a time
selector retires a previous screen selector, and conversely.

For the Python API, set `Beam(source="gpt_gdf", energy_keV=30,
transverse_fwhm_x_mm=None, transverse_fwhm_y_mm=None, gdf_path="beam.gdf",
gdf_time_s=1e-9, gdf_z_origin_m=...)`. `energy_keV` is a scalar scene placeholder;
actual initial energies come from the GDF. The other normalization settings
have the same names as the catalog fields.

## Errors and limits

- Missing/unreadable/malformed/non-GPT file: supply native GPT output at the
  reported path, not processed statistics.
- No time blocks: select a screen with `gdf_screen_position_m`, or export `tout`.
- Missing arrays, empty block, or unequal lengths: export complete particle
  records. Missing data are never padded with zeros.
- Unavailable/ambiguous time: inspect `beam gdf-times`, then choose a time and
  an absolute tolerance smaller than the neighboring time spacing.
- Invalid beta, gamma mismatch, non-electron mass/charge, or bad weights:
  correct the upstream particle output; the error identifies the field and
  reports the maximum gamma discrepancy when applicable.
- Missing origin, invalid shared rate, or conflicting analytic settings: supply
  the explicit origin, use a finite positive `rep_rate_hz`, and remove analytic
  distribution fields. The shared default rate already satisfies charge mode.
- Outward/grazing sample-frame rays: correct the coordinate alignment or target
  tilt. Directions are never flipped or discarded.

Loading and sampling run in NumPy on the CPU. Only numeric initial-state arrays
reach the existing CPU/CUDA transport boundary. Standard tests exercise CPU
transport; actual GPU validation has not been performed for this source.

## Inspect coordinates and select a screen

List outputs and inspect the sole time snapshot in `beam.gdf` automatically:

```bash
pyrite beam gdf-inspect beam.gdf
```

Inspect a particular screen, including actual lab x/y/z ranges and weighted means:

```bash
pyrite beam gdf-inspect screen.gdf --screen-position-m 1.05
```

Add `-o json` for structured output. Use the physical target's lab-z coordinate
for `gdf_z_origin_m`. The printed centroid option places the target there by
choice; it does not infer where a physical target is. A screen label need not
equal actual lab z. Transverse offsets remain unchanged; compare the reported
x/y ranges with the finite target footprint.

For a target placed at lab z = 1.05 m, import that screen with:

```bash
pyrite beam create gpt_screen --source gpt_gdf \
  --gdf-path screen.gdf --gdf-screen-position-m 1.05 \
  --gdf-screen-tolerance-m 1e-9 --gdf-z-origin-m 1.05 \
  --gdf-normalization gdf_charge --rep-rate-hz 5000
pyrite profile set standard --beam gpt_screen
pyrite run standard -m hopg
```

`gdf_screen_position_m` and `gdf_time_s` are mutually exclusive. The screen
selector accepts signed coordinates; `gdf_screen_tolerance_m` defaults to
`1e-9` m, is absolute and nonnegative, and must select exactly one output.
CLI spellings use hyphens (`--gdf-screen-tolerance-m`). Catalog and Python
beam fields use underscores. Screen records undergo the same signed,
field-free projection as time snapshots, adding flight time to their
correlated crossing-time offsets.

## Use only the imported beam shape

`--gdf-shape-only` preserves imported positions, directions, and relative
particle weights, while assigning every electron the energy of the current
profile sweep case. The profile's full energy grid is retained. Imported
screen crossing-time differences are discarded; signed flight-time offsets
to the target are computed from the newly assigned energy. Normalization
and the selected target origin retain their usual meanings.

For the generated time snapshot and the standard profile energies:

```bash
pyrite beam set gpt_beam --gdf-shape-only
pyrite profile set standard --beam gpt_beam
pyrite run standard -m hopg
```

Catalog/Python field: `gdf_shape_only = true` / `gdf_shape_only=True`.
The default is false; `beam set gpt_beam --no-gdf-shape-only` switches back to imported
energies and crossing times. Native electron records are still validated.
This mode preserves angular divergence, not momentum, and does not model
acceleration or beam evolution through accelerating fields.
