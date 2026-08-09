# Memory Maintenance

## Discovery Model

- Build a progressively disclosed graph of memories.
- Use `mem:core` as the root; link focused memories with descriptive context.
- Group focused memories by topic when the project needs more depth.
- Write references as backticked `mem:<name>` links.
- Put read-routing guidance in the referring memory, not the target.

## Style

- Write dense agent notes: durable invariants and terse bullets.
- Omit obvious context, generic knowledge, rationale, and examples unless they prevent likely mistakes.
- Exclude task-local or volatile line-level details.

## Maintenance

- Add or update only stable, non-obvious facts that avoid expensive rediscovery.
- Rename memories through Serena so references update automatically.
- Run `serena memories check` after structural changes.
