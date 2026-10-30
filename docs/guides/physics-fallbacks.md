# Physics-table fallback policy (issue #262)

PyRITE's `bremsstrahlung_model="auto"` now requires BremsLib tables for every
emitting element **before** a campaign starts. If a table is missing, the run
raises `TableNotFoundError`, with the command `pyrite tables fetch bremslib`.
A missing BremsLib checkout for an element not in the distributed release is
also an error; provide its table explicitly rather than silently degrading.

To intentionally run the isotropic EEDL continuum, set
`bremsstrahlung_model="eedl"` in the physics configuration. This is an explicit
model selection, not a fallback. For the previous *auto-if-installed, otherwise
EEDL* behavior in a **local** CLI sweep, add `--allow-physics-fallback` to
`pyrite run PROFILE`. This emits `PhysicsFallbackWarning`, resolves the whole
continuum to EEDL, and puts a `physics_fallbacks` event in the dataset's
identity metadata (requested/used model, elements, scope, and unavailable
preflight metrics). The executed source still participates in checkpoint keys.

This is the first slice of #262, not the final strictness contract. The
spectrum's per-segment coverage fallbacks and the remote queue flag/metadata
path still need a follow-up. A measured segment-energy span and affected
fraction require runtime aggregation; this preflight cannot infer them.
