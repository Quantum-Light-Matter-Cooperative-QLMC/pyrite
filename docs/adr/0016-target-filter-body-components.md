# 0016 — Share descriptions across target and filter roles

- **Status:** Proposed
- **Date:** 2026-10-10
- **Issue:** [#401](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/401)
- **Design:** [Target and filter object semantics](../research/instrumentation/target-filter-bodies.md)

## Context

`MaterialSpec` describes a runnable target recipe. `CrystalSpec` and
`MediumSpec` describe substance data. `Slab`/`Stack` describe source targets;
`FilterPlate` describes a homogeneous, positioned downstream attenuator.
Their shared vocabulary obscures different material namespaces, reference
frames, and enabled interactions.

The same substance can occupy either role, but sharing its description does
not supply the missing electron navigation or photon production model.
[ADR-0008](0008-no-arbitrary-target-geometry.md) keeps the target geometry set
closed and downstream instruments outside electron transport.

## Proposed decision

Retain the specialized target and filter types. Share immutable descriptions
through explicit adapters, starting with homogeneous substance resolution in
the material layer. A subsequent bounded adapter may describe a scalar,
homogeneous rectangular body with separate dimensions, substance, and pose.
It must lower to the existing specialized types; it is not a new navigator.

Keep campaign recipes and sweep expansion outside scalar body descriptions.
Keep crystal orientation separate from body pose. Assign behavior explicitly
through the target or primary-attenuator wrapper; a display label or location
does not enable interactions. Reject unsupported policy requests rather than
silently dropping them. Layered bodies remain ordered layers and cannot be
coerced to a homogeneous composition.

Do not replace existing catalog tables, public constructors, stored payloads,
or identity encoders in the initial migration. Equivalent legacy and adapted
inputs must produce identical resolved cases and identity payloads. Introduce
new schema versions only when new semantics actually require them.

## Alternatives

Retaining all types with terminology changes alone has the lowest migration
cost, but leaves duplicated composition lookup and validation. One closed
physical-object set with separate scene assignments makes roles explicit,
but would require new lowering rules for incompatible geometries and
interaction capabilities without an immediate scientific benefit. Shared
components with specialized wrappers address the demonstrated duplication
while preserving the solver boundary.

## Consequences

Users can reuse substance and eventually rectangular-body descriptions while
the API continues to make solver ownership visible. An amorphous medium does
not become a supported entrance radiator merely because a body can hold it.
A filter intersected by electrons requires a separately reviewed transport
design; unifying navigation requires a superseding decision for ADR-0008.
Detector response, scoring, and acquisition remain separate.

This record is proposed for design review. It authorizes no implementation,
new physical claims, numerical changes, or schema migration. The linked study
defines compatibility gates and bounded follow-up slices.
