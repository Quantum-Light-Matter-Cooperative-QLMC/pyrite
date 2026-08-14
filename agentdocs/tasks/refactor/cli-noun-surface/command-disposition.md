# CLI command disposition

Inventory date: 2026-08-14. Source: `tests/data/cli_contract.json` at
`2ebe356`. The RFC's count of 88 predates the named-beam and detector/scene
work: the current frozen tree contains 94 visible paths and 43 hidden paths
(138 nodes including the root).

## Contract rule

Every moved spelling invokes the same Click callback as its replacement.
Therefore parameters/defaults/config precedence, completion callbacks, result
stdout, diagnostic/warning/prompt stderr, exit codes, JSON envelopes, preview
and confirmation behavior, revalidation, and side effects remain owned by the
existing callback. Only the command path and its D7 warning differ. Help for a
replacement is generated from that callback; root help hides each retired
top-level noun. Contract tests and real subprocess probes must demonstrate this
rather than relying on the shared-callback construction alone.

## Visible paths

| Current path(s) | Disposition / replacement |
| --- | --- |
| `run` | Keep. |
| `setup` | Move to `pyrite config setup`; retain hidden D7 alias. |
| `app`; `app analysis`; `app analysis export`; `app analysis launch`; `app viewer`; `app viewer export`; `app viewer launch`; `app validation`; `app validation export`; `app validation launch` | Keep unchanged. |
| `checkpoint`; `checkpoint slim`; `checkpoint recompute`; `checkpoint recompute brem`; `checkpoint recompute line`; `checkpoint archive`; `checkpoint restore`; `checkpoint list`; `checkpoint merge`; `checkpoint gc`; `checkpoint rm` | Keep unchanged. Checkpoints remain results, never a cache noun. |
| `completion`; `completion install`; `completion remove` | Move below `pyrite config completion`; retain hidden D7 aliases. |
| `config`; `config get`; `config list`; `config set` | Keep; add the relocated `setup` and `completion` children. |
| `performance`; `performance analyze`; `performance list`; `performance rm` | Move unchanged to `pyrite-dev performance`; retain hidden D7 aliases. |
| `remote`; `remote gc`; `remote performance`; `remote performance list`; `remote performance pull`; `remote performance rm`; `remote profile pull`; `remote prune-jobs`; `remote pull`; `remote rm`; `remote sync` | Keep unchanged. Resource management remains under `remote`; submission stays a modifier on user workflows. |
| `job`; `job attach`; `job list`; `job logs`; `job status`; `job stop` | Keep unchanged. |
| `energy-grid` | Hide as the compatibility parent for all former paths. No replacement noun at the user root. |
| `energy-grid derive` | Move unchanged to `pyrite material energy-grid derive`. |
| `energy-grid show`; `energy-grid line`; `energy-grid line show`; `energy-grid brem`; `energy-grid brem show` | Move unchanged below `pyrite material energy-grid`. |
| `energy-grid defaults` | Move unchanged to `pyrite profile energy-grid defaults`. |
| `energy-grid add`; `energy-grid line set`; `energy-grid brem set`; `energy-grid rm`; `energy-grid verify`; `energy-grid gc` | Move unchanged below `pyrite-dev energy-grid`. |
| `energy-grid regen-golden` | Move to the existing `pyrite-dev regen-golden`. |
| `energy-grid job` | Retire to the existing top-level `pyrite job` lifecycle; the group remains hidden for D7. |
| `sweep set`; `sweep show` | Existing hidden compatibility group; no change. |
| `profile`; `profile add`; `profile create`; `profile delete`; `profile list`; `profile members add`; `profile members remove`; `profile members reset`; `profile members set`; `profile remove`; `profile rename`; `profile set`; `profile show` | Keep unchanged. Interactive mutation remains supported; see decision below. |
| `material`; `material set`; `material show`; `material blaze`; `material validate` | Keep; add the relocated `energy-grid` child. |
| `beam`; `beam create`; `beam delete`; `beam list`; `beam rename`; `beam set`; `beam show` | Keep unchanged. Beam remains a first-class physical noun. |

## Hidden paths

| Current hidden path(s) | Disposition / replacement |
| --- | --- |
| `checkpoint prune`; `checkpoint clear`; `slim`; `rebrem`; `reline`; `archive`; `restore`; `archives`; `union`; `prune` | Keep existing D7 aliases unchanged. |
| `performance prune` | Preserve under the hidden compatibility parent; canonical replacement becomes `pyrite-dev performance rm`. |
| `remote check`; `remote clear`; `remote jobs`; `remote logs`; `remote performance prune`; `remote profile`; `remote prune`; `remote reap`; `remote rebrem`; `remote reline`; `remote run`; `remote status`; `remote stop`; `remote validate` | Keep existing D7 aliases unchanged. |
| `energy-grid apply` | Preserve; replacement becomes `pyrite-dev energy-grid add`. |
| `energy-grid attach`; `energy-grid logs`; `energy-grid status`; `energy-grid stop`; `energy-grid job attach`; `energy-grid job logs`; `energy-grid job status`; `energy-grid job stop` | Preserve; replacement remains `pyrite job attach|logs|status|stop`. |
| `energy-grid line delete` | Preserve; replacement becomes `pyrite-dev energy-grid rm`. |
| `energy-grid submit` | Preserve; replacement becomes `pyrite material energy-grid derive --remote --detach`. |
| `sweep`; `profile add-material`; `profile members`; `profile remove-material` | Keep existing D7 aliases unchanged. |
| `profile analyze` | Preserve; replacement becomes `pyrite-dev performance analyze NAME`. |
| `check`; `check-config` | Keep existing D7 aliases unchanged. |

## Settled open decisions

1. Keep all interactive profile mutation flows. Removing leaf verbs does not
   reduce the top-level noun count; named-beam attachment already relies on the
   same lifecycle; the approved profile lock/unlock and create-from fixes both
   require continued mutation. Those inbox/bug items are not implemented here.
2. All B-D moves are safely landable now. This branch contains the completed
   detector-scorer and scene-object-model dependencies, including the public
   filesystem-free simulation API. Slice E therefore has no remaining gate,
   but its evidence-backed outcome is no behavior change.
3. The primary target remains exactly nine visible top-level nouns: `run`,
   `app`, `checkpoint`, `config`, `remote`, `job`, `profile`, `material`, and
   `beam`. No `cache` or `detector` noun is introduced in this milestone.
