# Zhai reported tilt inputs

## Goal

Make the Zhai supplementary validation inputs match the orientations explicitly
reported in Supplementary Table 4 and avoid presenting an unreported TEM
azimuth as a measured value.

## Design

Represent each supplementary study with explicit polar tilts and an optional
azimuth. The 921 nm h-BN study becomes the single reported orientation
`theta_til = 17 degrees`, `phi_til = 130 degrees`, at the SEM beam energies
17.5, 20, 22.5, and 25 keV. MoSe2 and WSe2 retain the reported TEM polar series
10, 15, 17.5, and 20 degrees at 200 keV, while their azimuth remains `None`
because the paper does not report it.

The geometry call will receive the reported h-BN azimuth. For an unreported TEM
azimuth, the validation model will not silently substitute zero: attempting to
model those studies without an explicitly selected exploratory azimuth will
raise a clear error. This keeps published inputs distinct from later fitting or
azimuth sweeps.

The validation app will describe which values are reported and which remain
unknown. The provenance note will transcribe all graphite and h-BN pairs from
Supplementary Table 4 and retract the claim that azimuth zero is resolved by
the paper.

## Tests and validation

Focused tests will freeze the reported study metadata, verify that the h-BN
geometry receives 130 degrees, and verify that an unknown TEM azimuth cannot be
modeled implicitly. Existing anchor-figure tests, lint, type checking, and the
full test suite will be run before integration.

No production transport equation changes are included. This is a correction to
validation-study inputs and provenance; ledger claim states remain unchanged.
