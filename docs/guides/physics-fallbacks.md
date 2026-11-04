# Physics-table fallback policy

PyRITE's `bremsstrahlung_model="auto"` now requires BremsLib tables for every
emitting element **before** a campaign starts. If a table is missing, the run
raises `TableNotFoundError`, with the command `pyrite tables fetch bremslib`.
A missing BremsLib checkout for an element not in the distributed release is
also an error; provide its table explicitly rather than silently degrading.

To intentionally run the isotropic EEDL continuum without coupled radiative
transport, set `bremsstrahlung_model="eedl"` and `radiative_model="uncoupled"`
in the physics configuration. These are explicit model selections. For the previous *auto-if-installed, otherwise
EEDL* behavior in a CLI sweep, add `--allow-physics-fallback` to
`pyrite run PROFILE`. This emits `PhysicsFallbackWarning`, resolves the whole
continuum to EEDL, and puts a `physics_fallbacks` event in the dataset's
identity metadata (requested/used model, elements, scope, and unavailable
preflight metrics). The executed source and opt-in policy participate in
checkpoint and content-cache keys, so a strict request cannot reuse an opted-in
surrogate result. Existing checkpoint identities remain readable.

Both local and remote campaign workers refuse per-segment substitutions by
default. With `--allow-physics-fallback`, uncovered BremsLib rows may use EEDL;
missing EEDL coverage may in turn use Bethe--Heitler. This also applies below
the supplied table's floor: it is a surrogate, not a source-backed model splice.
Malformed or corrupted reference data remain errors. An explicit coupled
`bremslib-soft-hard` request is never downgraded by the flag.

Completed result rows carry `physics_fallbacks` in both line and bremsstrahlung
checkpoint components. Each event records requested/used models, element,
reason, `energy_span_keV`, `segments_total`, `segments_affected`, and
`fraction_segments_affected`. Continuum fractions count the rows actually
scored after electron/cutoff selection and escape-path splitting. They aggregate
across layers and batches per element; repeated photon-grid blocks count once.
Uncoupled-transport events count all transport rows in layers containing that
element. These are row fractions, not fractions of photon yield, path length,
or energy loss. Preflight metadata keeps spans/fractions as `None`; old records
without measurements are not retrospectively assigned them.

Programmatic campaign preflight also accepts `Settings(allow_fallback=True)`;
worker cases carry `allow_physics_fallback=True`. Direct low-level spectrum
calls retain warning-based substitution unless a strict warning filter is set.
Explicit EEDL and Bethe--Heitler choices remain available; selecting EEDL does
not imply consent to an additional Bethe--Heitler substitution in a campaign.

Explicit `uncoupled` scores photons after collision-only transport, omitting
radiative energy-loss feedback. Use it only when that approximation is
appropriate to the intended diagnostic; it has no general quantitative accuracy
guarantee. See the [radiative transport explanation](../physics/radiation-physics/hard-bremsstrahlung-events.md)
and [coverage review](../validation/radiation-physics/brem-spectrum.md#coverage-policy-verification-issue-262-2026-11-04).
