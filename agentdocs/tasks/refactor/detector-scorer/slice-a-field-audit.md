# Slice A — `DetectorSpec` field disposition audit

Branch: `refactor/detector-scorer`. Audit only; no behavior change lands with
this document. Line numbers are against `main` @ `7094130`.

## Method

Every field of `DetectorSpec` (`src/pyrite/detectors/spec.py:71-79`) traced to
(a) its write sites, (b) any site that reads it into a *computed number*,
(c) whether it reaches the hashed identity payload, and (d) whether an existing
response model already has a parameter that means the same thing.

The distinction that decides each row is **read for behavior** versus
**round-tripped only**. A field that is validated, serialized, hashed and
handed back unchanged is not live, however many call sites mention it.

## Disposition table

| Field | Written by | Read for behavior | In hashed payload | Response-model counterpart | Disposition |
| --- | --- | --- | --- | --- | --- |
| `observation_angle_deg` | catalog `[detector]`, `Sweep(theta_obs_deg=)`, `Sweep(detector=)` | yes — `sweep.py:729,768` → case `theta_obs_deg` / `theta_obs_rad`; `geometry.py:414` groove-blaze rule | yes, flat key `theta_obs_deg` | — (acceptance, not response) | **KEEP live** |
| `polar_acceptance_deg` | same | yes — `sweep.py:600-604` → case `dtheta_obs_rad`; feeds `aperture_fwhm_eV` in `store.line_fwhm_eV:121` | yes, flat key `dtheta_obs_deg` | — | **KEEP live** |
| `solid_angle_sr` | same | yes — `sweep.py:608-609` → case `domega_sr`; `store_result:156` folds it into `scale` | yes, flat key `domega_sr` (content-key denylisted, `profiles.py:66`) | — | **KEEP live** |
| `response_model` | catalog `[detector]` only | **no** | yes, under `sweep.detector` when non-`None` | none — a string registry id is the placeholder that the `response` object supersedes | **DELETE** — replaced by `Detector.response` |
| `qe_curve` | catalog `[detector]` only | **no** | same | none. Timepix derives QE from Si thickness (`absorption_efficiency`, `timepix_response.py:138`); Eagle loads a *packaged* table selected by `coating="BN"/"BEN"` (`eaglexo_response.py:203,216`), never a caller-supplied path | **DELETE** — no consumer, and neither model wants a path |
| `pixel_pitch_um` | catalog `[detector]` only | **no** | same | `timepix_response.PIXEL_PITCH_UM = 55.0`, commented "fixed by the chip"; Eagle's `pixel_um` lives in the `SENSORS` table (`eaglexo_response.py:104-106`). Neither is a constructor parameter | **DELETE** — hardware constant, not a per-run knob |
| `sensor_thickness_um` | catalog `[detector]` only | **no** | same | **yes** — `TimepixResponse(thickness_um=...)` (`timepix_response.py:324-332`), default `SENSOR_THICKNESS_UM = 300.0`; also `build_response(thickness_um=)`, `sigma_diffusion_um(thickness_um=)`, `absorption_efficiency(thickness_um=)` | **BECOMES LIVE** — moves to `Timepix3(thickness_um=)` on the response object |
| `distance_mm` | catalog `[detector]` only | **no** | same | Eagle's `solid_angle_sr(width_mm, height_mm, distance_mm)` (`eaglexo_response.py:128`) and `geometry(distance_m=)` *derive* a solid angle from it; `grating.disperse_spectrum(distance_mm=)` takes its own | **DELETE** — an upstream input to `solid_angle_sr`, which is already a first-class field. Keeping both invites two disagreeing sources for one number |
| `threshold_eV` | catalog `[detector]` only | **no** | same | `timepix_response.THRESHOLD_E = 525.0` electrons, a module constant with no constructor parameter | **DELETE** — chip constant |

Net: of the six inert fields, exactly **one** (`sensor_thickness_um`) has a real
existing consumer, and it belongs on the response object rather than on the
acceptance object. The other five are deleted.

## Open question resolved: keep-or-delete for a plausible future consumer

The task doc defaults to delete and asks slice A to justify every survivor. No
field survives on the basis of a *plausible* consumer. `sensor_thickness_um` is
retained only because it has an **actual** one today, and it is retained by
being moved, not by staying inert. `bias_v` — Timepix's other genuine knob
(`BIAS_VOLTAGE_V = 100.0`) — has no `DetectorSpec` field at all, which is the
clearest evidence that the inert set was never derived from what the response
models consume.

## Why the inert fields are a real liability, not a cosmetic one

They reach the run identity hash. `profiles.py:294-305` pops the detector
payload, projects the three live fields onto their historical flat keys, and
re-attaches **every remaining non-`None` field** as `sweep_payload["detector"]`,
which is then serialized into `parameter_sha256` (`profiles.py:384-392`).

So today, two runs that differ only in `pixel_pitch_um` get different digests,
different checkpoint stems, and cannot reuse each other's transport — while
producing numerically identical arrays. That is the RFC's "inert field in a
hashed payload" made concrete.

## Bit-for-bit consequence of deleting the five

None. The re-attachment at `profiles.py:298-303` filters `value is not None`, so
an unset inert field contributes nothing to the digest. A repo-wide search over
`*.toml` / `*.json` finds **no** file that sets any of the six; the shipped
catalog (`src/pyrite/data/materials.toml`) sets only
`detector = {observation_angle_deg = 90.0}`. Every existing digest is therefore
computed from a payload in which these keys are already absent, and removing the
fields cannot perturb one.

## Migration constraint the deletion must respect

`catalog.py:1066` validates a profile's `[detector]` table with
`errors.keys(table, path, set(_DETECTOR_KEYS))` — an **unknown key is a hard
validation error**. Dropping the five names from `_DETECTOR_KEYS` would turn a
previously-valid user profile into a failing one.

Migration: remove the fields from `DetectorSpec` (satisfying "no inert field
survives on the public detector object", and removing them from the hashed
payload), but keep the five names accepted-and-ignored at the TOML boundary
with a deprecation warning for one support window. The keys are then dropped
before `DetectorSpec(**known)` is constructed, so nothing inert reaches the
object or the digest.

## Call sites that must move with the deletion

| Site | Change |
| --- | --- |
| `src/pyrite/detectors/spec.py:74-79,100-117` | drop five fields + their validators; keep `_portable_identifier` only if still used |
| `src/pyrite/materials/catalog.py:834-845,1066-1069` | retire the five to a deprecated-key set, warn, drop before construction |
| `src/pyrite/campaign/profiles.py:298-305` | `reserved_detector` becomes empty for the retired names; keep the mechanism for future live fields |
| `tests/detectors/test_spec.py:15-52` | round-trip assertions over the retired fields |
| `tests/scan/test_sweep.py:169-193` | same |
| `tests/materials/test_profiles.py:304-339` | same, plus the catalog-boundary cases |
