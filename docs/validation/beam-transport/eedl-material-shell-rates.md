# EEDL material shell rates

Validation: `eedl-material-shell-rates`. Status: unverified. This is a host-side
rate preparation step; no electron transport mode consumes it.

## Source and construction

The packaged EPICS2025 EEDL ENDF tape supplies electron-impact subshell
ionization cross sections in MF=23, MT=534–572. The tape is pinned by its SHA-256
digest in `montecarlo/eedl_ionization.py`. Each section carries a shell
designator, binding energy, incident-energy grid, and cross section. The parser
accepts only ENDF linear-linear interpolation for these sections and converts
barns to cm². The catalog supplies each element's number density in atoms/Å³.

For element $i$ and shell $s$, the host rate is

$$
\lambda^{-1}_{i,s}(E)
  = n_i\,\sigma^{\rm EEDL}_{i,s}(E)\,10^{16}\quad {\rm \AA}^{-1},
\qquad
\lambda^{-1}_{\rm total}(E)=\sum_{i,s}\lambda^{-1}_{i,s}(E).
$$

The factor $10^{16}$ converts cm² to Å². The independent-atom mixture rule
assumes a homogeneous material and adds shell rates without a compound
correction. A shell below its binding energy has zero rate. An accessible shell
outside its EEDL projectile grid is rejected rather than extrapolated. Thus
the all-closed-shell limit has zero rate, and a one-element composition reduces
to its elemental rate multiplied by the element number density.

`material_shell_ionization_rates` preserves element, EEDL shell designator,
binding energy, and the native projectile-energy limits on every channel, so
a future hard event can identify the physical vacancy and its data coverage.
The total rates do not specify an energy-transfer distribution, secondary
kinematics, or a soft stopping moment. Those require the separately validated
GOS model and a positive stopping-closure procedure before any transport use.
The shared EEDL parser is a sibling of `transport` and `spectrum`, allowing
both to read the same pinned shell tables without importing either subsystem.

## Validation still needed

Compare interpolated per-shell and material rates with independently tabulated
EEDL values and the Bote–Salvat K/L/M reference across representative elements
and energies. Check that any later transfer spectrum reproduces selected shell
rates and closes to corrected SBETHE stopping without changing the inner-shell
vacancy rates. Fresh-context source-to-code verification remains pending.
