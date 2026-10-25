# Initial differential overflow evidence (#522)

Array 1614; remote checkout `/home/user/pyrite-522-overflow`, outputs
`scratch/overflow/`. Source hashes and exact parameters are in `submit.sh`
and each JSON. All four tasks completed by 2026-10-20 12:10 UTC. Tasks 2–3
were retrieved after completion; do not sync over the isolated checkout until
all outputs needed from it have been retrieved.

All allocations use one CPU, 4 GiB and four hours; at most two run concurrently.
Task 0: Si(110), 5 MeV, H16/cells16/dz125 Å. Task 1: same with dz62.5 Å.
Tasks 2–3: Si(100)/Si(110), 54.5 MeV, H48/cells16/dz31.25 Å.
All use 20 µm thickness and the declared 300 eV margin. These are fixed-energy
instrument diagnostics, not full Kephart reproduction runs.

The retrieved schema-1 JSON reports use historical `*_probability` keys.
Interpret `full_loss_probability` as an integrated jump weight (expected count
of repeated jumps), not a probability bounded by one. Outside and unresolved
entries are integrated gain/loss weights per incident electron. Schema 2 of
the harness corrects these labels to `full_loss_weight`, `reference_overflow_flux`,
`resolved_outside_flux`, `unresolved_weight`, `outside_weights` and
`solver_overflow_flux`; the physical computation is unchanged. The submitted
harness hash therefore differs from the current script's schema-2 hash.

Tasks 0–1 produced valid JSON and completed with diagnostic exit status 1,
meaning their requested transfer windows failed the 1% closure gate. This is
an explicit negative scientific result, not successful tail sampling.
The reference overflow reconstructed from the accumulated source matches
solver-booked overflow to roundoff. Review `resolved_outside_probability`,
`unresolved_probability` and `inband_gain_shift` together. Remote logs retain
source-hash checks and `/usr/bin/time` exit statuses; JSON existence alone is
insufficient evidence of successful execution for remaining tasks.

Tasks 2–3 also produced valid JSON and completed with diagnostic exit status 1.
Both Si(100) and Si(110) at 54.5 MeV fail closure for transfer orders 64 and
96 (`overflow spectrum is not closed against reference loss`). This is a
negative result; neither distribution is eligible for conditional sampling.
Retrieved JSON SHA-256 values: run-2 `71c9adaad9dfd5a3a93f195ccfadfa51d0ed5c00e822c22c99056892668dcbeb`,
run-3 `037f1415b9150bfd4daf911c6f181a3add92eaced47f91e643f4c90aa306e683`.
The corresponding logs record `/usr/bin/time` exit status 1 and diagnostic
exit status 1 for both tasks.
