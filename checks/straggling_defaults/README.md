# Collision-straggling default qualification (#642)

This is an opt-in sensitivity pilot, with production defaults unchanged.
`protocol.json` freezes the first bounded subset: 30 keV electrons, a 5 keV
tracking threshold, four shell-covered materials and sapphire as an unsupported
shell control, at 0.05 and 0.8 of the SBETHE CSDA range down to that threshold.
That range sizes a shared slab for on/off arms; it is not an independent range
validation. Every slab has a 10 × 10 mm footprint and normal pencil incidence.

The exact transport cores use ELSEPA, corrected SBETHE, Kawrakow atomic-electron
deflection, midpoint integration, disabled transport LUT, and coupled BremsLib
loss with a 1 keV hard-photon cutoff. Shell arms resolve `shell-soft-hard` at
50 eV; the sapphire control explicitly uses continuous stopping. No secondary,
pair, positron, channeling or transition-radiation transport is selected.
PXR/CBS, line and detector scoring are absent. Hard photons above 10 keV are
creation-level analog counts; these are not the weighted production detector
estimator. The study calls the transport API with explicit resolved models,
not its different low-level defaults.

Each arm warms one identical 512-primary workload and then measures three
independent 512-primary seeds. On/off seeds are disjoint; matching seeds across
CPU/GPU or refinements do not establish pairing. Raw `.npz` files retain one
score per incident primary, including zeros. Conditional transmitted-energy
moments and quantiles additionally report their selected population size.
The ledger includes soft loss, hard inelastic and radiative debits, and the
remaining energy after a terminal hard event. Every result must contain one
terminal record per primary and pass the segment event contract.

`analyze.py` verifies raw hashes, protocol identity, seed uniqueness, table
identity and replica completeness before pooling. Pointwise normal intervals
are sensitivity diagnostics. The predeclared 2 percentage-point fraction and
2% transmitted-mean margins do not certify accuracy, rare tails, or simultaneous
equivalence. Missing replicas and zero observed variance cannot qualify a
margin. Timing is synchronized warm transport at fixed population; warmup is
reported separately and import/table setup is excluded but not timed. Three
short samples cannot establish production
overhead or time to equal achieved precision. Peak RSS is process-wide host
memory, not GPU peak allocation; `gpu.txt` records device configuration.

## Remote execution

Use the linked issue worktree and a dedicated remote checkout:

```bash
PYRITE_REMOTE_DIR='~/pyrite-642-straggling' uv run pyrite remote sync
```

On the lab host, inside that folder, install `uv sync --extra nvidia --locked`.
Submit `sbatch checks/straggling_defaults/run.sbatch silicon` as preflight,
then `hopg`, `mos2`, `wse2`, and `sapphire`. Each allocation requests one CPU,
one GPU, 24 GiB RAM and at most ten minutes; each arm has a two-minute timeout.
Failures remain in `failures.txt` and their logs. Do not overwrite evidence
with a rerun. `refinement` separately varies the on-state Si 0.8-range cap
(0.025%, 0.0125%) and collision cutoff (100, 200 eV). `cpu` repeats Si 0.8-range
on/off on one host thread. Inspect actual `energy_substeps` before interpreting
step refinement. These are bounded convergence probes, not selected defaults.

The `anchors` allocation runs the existing exact/LUT Urban and exact shell
hardware tests through `pyrite-dev`. Copy `tests/` separately: maintained remote
sync ships source/checks but not tests. Test isolation clears `PYRITE_HOME`, so
the batch script redirects its platform-data fallback to the staged checkout
inside the dedicated folder. No shared user data is modified.

Each job retains source scripts, sync identity, scheduler/CPU/GPU records and
SHA-256 manifests under `evidence/straggling/JOB-MODE/`. Pull those directories
and analyze with:

```bash
uv run python checks/straggling_defaults/analyze.py RAW_ROOT summary.json
```

The protocol lists deferred energies, layer/groove cases, line/detector
observables, rare-tail populations, independent accuracy references, and
held-out equal-precision timing. Reuse #510's comparator/scorer infrastructure
for those gates. No default/identity/model marker changes are warranted by this
pilot alone; historical omission semantics remain unchanged.
