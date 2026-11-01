# Photon CUDA evidence (#341)

`benchmark.py` runs the fixed-physics CUDA scaling, profiling and Optuna chunk
experiments; run it only in a remote GPU allocation (`pyrite remote`). The
`*-rtx5080.json` files are pinned RTX 5080 records (first flight, production,
performance, charged re-entry) with source hashes, cited by
`docs/validation/beam-transport/photon-charged-reentry.md`. Do not delete or
regenerate them in place; they are evidence, not a physics sign-off.
