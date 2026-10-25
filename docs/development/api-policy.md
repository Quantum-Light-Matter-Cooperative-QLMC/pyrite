# Python API namespace and compatibility policy

PyRITE exposes one deliberately governed Python API. A name being importable
from an internal module does **not** make it supported. The supported contract
is described by [the API reference](../api.md) and its package-root and
canonical-facade inventory in `api-surface.toml`. This page is the review policy for new APIs.

## Classification

- **Stable:** supported and subject to documented compatibility/deprecation rules.
- **Advanced-stable:** specialized but subject to the same compatibility rules.
- **Experimental:** provisional; document limitations, validation requirements,
  and migration/promotion criteria before introducing production use.
- **Internal:** implementation detail; not part of compatibility guarantees.

The inventory's `root_role` is independent of stability: `core` identifies the
nine foundational names; `specialized` records preexisting root exports, **not**
a deprecation; `experimental` identifies clearly provisional root entries.
Existing exports keep working. Relocation, promotion, or deprecation is a
separate reviewed change.

## Canonical ownership

The **public** domains are `beams`, `geometry`, `materials`, `detectors`,
`observations`, `config`, `quantum`, `transport`, `radiation`, `campaign`,
`execution`, `optimization`, `results`, `storage`, `plots`, `experimental`,
`simulation`, and `datasets`.
These are target API domains, not a claim that every facade exists today.

Every new public concept has one owning domain. Do not create another root
re-export, a top-level module, or a duplicate public spelling merely because an
implementation already exists elsewhere. New root exports require explicit
architectural review. Domain facades may initially re-export a concrete
implementation without moving it. A compatibility alias must resolve to the
**same object** as its canonical import.

Public configuration values should be immutable, typed, validated, and safe to
import without initializing CUDA, plotting, schedulers, or loading datasets.
Distinguish physical models, numerical approximations, sampling, kernel tuning,
and scheduler resource policies. Specify physical units, defaults, side effects,
serialization, result units, scientific identity, and reproducibility behavior.

## Before approving a public feature

1. Identify its user-facing concept, owning domain, and API status.
2. Name its one canonical import and justify any compatibility aliases.
3. Review dependency direction and lightweight import behavior.
4. Document units, defaults, validation, errors, and side effects.
5. Assess simulation physics, numerical reproducibility, checkpoint schemas,
   and content-addressed identities; do not change them for an import-only move.
6. Add focused behavior/import tests, examples, and API documentation.
7. If replacing a supported path, use the existing `_api_deprecations` /
   `_module_deprecations` schedule; do not remove it immediately.

## Inventory and static check

`api-surface.toml` records **all existing root lazy exports**, the five active
canonical facades, and the supported package and leaf-module contracts documented
in the API reference. Its `target` is the *current concrete binding*, while
`domain` identifies the owning public domain; some domains have no facade yet.
Other importable package names remain provisional or internal. Register future
supported contracts explicitly rather than promoting whole implementation
packages.

Run:

```bash
uv run pyrite-dev api check
uv run pyrite-dev api check --imports  # additionally resolve bindings and object identity
```

The default check is AST/TOML-based and performs **no application imports**.
It rejects new unapproved root names, removed root names, changed bindings,
duplicate targets, invalid classifications, and missing source modules. It
also requires every governed root/facade name to be discoverable by static
type checkers via a direct declaration or a `TYPE_CHECKING` import. Lazy
`__getattr__` exports alone are insufficient for editor completion or type
checking. The check does not freeze other members of preexisting facades.
`--imports` additionally checks that each alias resolves to the actual
underlying object; it may import scientific engines and requires a normal
project environment. The static gate is included in `pyrite-dev verify`.
`tests/dev/test_api_surface.py` also resolves the complete registered inventory
in CI, catching missing canonical attributes and broken runtime aliases.
Deprecation schedules remain owned by `_api_deprecations` and
`_module_deprecations`, with their existing schedule and live-tree tests; the
API inventory does not introduce a second removal schedule.

A subprocess smoke test blocks optional GPU, plotting, and distributed
backends *before* imports and exercises the canonical namespaces plus core
data-model imports, including `Scene`, `Result`, and `Target`. This catches
attempted imports even when the caller catches `ImportError`; the ordinary
smoke test alone cannot prove a completely
uninstalled dependency environment.

## Canonical domain facades (first migration slice)

`pyrite.beams`, `pyrite.geometry`, and `pyrite.config` are supported import
facades. Their export names come from the existing root inventory, filtered by
its `domain` field. The approved set of active facades is listed in
`api-surface.toml` as `canonical_facades`; `pyrite-dev api check` validates
both names and concrete target bindings, without importing the runtime. The
opt-in `--imports` mode also checks that every facade symbol **is** the same
object as the historical root export. Adding a facade member therefore
requires updating the inventory and its domain assignment deliberately.

Examples:

```python
from pyrite.beams import Beam, TransverseSpot
from pyrite.geometry import Slab, CrystalCut, Target
from pyrite.config import Physics, Numerics, FixedSampling, load_physics
```

This slice is *additive*: all root spellings and historical implementation
module paths continue to work, without warnings. No scientific configuration,
serialization, or content identity is migrated. `ExecutionConfig` retains its
existing spelling and semantics pending the separate numerics/config issue;
`PhotonTransport` remains experimental even when accessed through
`pyrite.config`.

## Documented subpackage contracts and legacy star-import compatibility

`[package_exports."pyrite.<namespace>"]` registers supported package and
leaf-module attributes, each with a concrete `module:attribute` target and API
status. Existing documented scientific helpers remain advanced-stable; the
axial diagnostic remains experimental. Concrete targets may live in private
implementation modules without making those modules public. The static gate
checks that the namespace exists in source, the name appears in a literal
`__all__` where present (otherwise in its declarations), the target module exists, and
editors can discover its direct declaration or `TYPE_CHECKING` import. When a
name is also registered at the root, the status and binding must agree. Run
`pyrite-dev api check --imports` to check runtime object identity as well.

`[legacy_private_all]` separately freezes only the underscore-prefixed
`__all__` entries of `pyrite.results`, `pyrite.plots`, and `pyrite.montecarlo`.
They remain importable for existing consumers, but are **internal compatibility
names**, not stable APIs. An addition/removal requires an inventory edit and
review; the private list does *not* freeze or promote the other historical
re-exports. Do not remove or rename existing imports as part of this governance
work. No numerical, serialization, or scientific identity behavior changes.

For a new supported namespace name, document its scope, add its status and
concrete target to `package_exports`, expose it in `__all__` if used, ensure a typing
declaration exists, then run the static and runtime API checks. Don't register
a helper merely because an implementation package exports it.
