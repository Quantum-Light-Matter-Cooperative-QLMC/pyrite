# CLI output vocabulary

Every human `pyrite ... list` and `pyrite ... show` result renders through
`pyrite.console.tables`: upper-case column headers, aligned columns, long
values wrapped unless `--output wide`. `--output json` bypasses the renderer
and keeps each command's documented schema and raw spellings.

## Shapes

| Shape | Columns | Used by |
|---|---|---|
| Named list | NAME, ORIGIN, extra columns, count, DESCRIPTION | `beam`, `detector`, `filter`, `target`, `numerics`, `physics`, `profile`, `tables` lists |
| Object | `[heading]  origin: ORIGIN`, then SETTING / VALUE | `beam`, `detector`, `filter`, `target`, `material`, `tables` show |
| Settings | SETTING / VALUE / SOURCE | `profile numerics\|physics show`, preset show, `adaptive-sampling` and `cxr-grid` show, `config` |

## Value source

Where an effective setting came from, highest precedence first:

| SOURCE | Meaning |
|---|---|
| `run` | run-time override |
| `material` | per-material override inside the profile |
| `profile` | set on the profile |
| `preset:NAME` | inherited from the attached named preset |
| `option` | given on the command line (`pyrite config`) |
| `env:VAR` | environment variable (`pyrite config`) |
| `config` | persistent config store (`pyrite config`) |
| `default` | built-in value; nothing selected it |

## Object origin

| ORIGIN | Meaning |
|---|---|
| `bundled` | read-only bundled catalog |
| `user` | your `catalog.user` layer |
| `catalog` | single explicitly selected catalog |

## Sampling mode

Numerics views open the sampling group with `mode`: `fixed (REASON)` when
history counts apply, or `adaptive (SOURCE policy)` when adaptive sampling
chooses them; fixed counts then read `unused (adaptive)`. Switch with
`pyrite profile numerics adaptive-sampling enable|disable PROFILE`.

## JSON compatibility

JSON source and origin spellings are a machine contract and are not
normalized (for example `built-in`, `fidelity`, `yours`). One payload changed
in 0.32: `cxr.profile.numerics.show` carries numerics rows only and adds a
`preset` key; physics rows moved to `pyrite.profile.physics.show`. Deprecated
`pyrite numerics|physics show PROFILE` keeps the legacy combined schema.

`remote job list|status` and `remote performance list` keep the dashboard
renderer, which colours job state.
