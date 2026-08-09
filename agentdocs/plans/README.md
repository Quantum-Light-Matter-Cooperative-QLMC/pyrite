# Cross-task plans (tracked, ephemeral)

Sequencing plans that coordinate multiple branch tasks and therefore cannot
belong under one `agentdocs/tasks/<branch-name>/`. Single-task plans and
handoffs stay with their task. Superseded material moves to `agentdocs/archive/`.

For current behavior use generated and durable material in
[`../../docs/`](../../docs/). Accepted decisions live in
[`../../docs/adr/`](../../docs/adr/); long-form rationale lives in the
`docs/*-rfc.md` files.

A plan here that shipped code actively cites as an authoritative decision source
has graduated to reference material — promote it to `docs/` and give it an ADR
entry rather than leaving it in this bucket.
