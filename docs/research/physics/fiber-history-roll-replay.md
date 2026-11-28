# History roll streaming and trajectory replay (#1527, Phase IIc)

The experimental fixed-grid, fixed-count `roll_sampling="history"` route
uses the persisted case seed and global primary ID to reconstruct the same
orientation for every physical flight of that primary. There is no new HDF5
field or mutable RNG checkpoint state. The artifact case digest already
includes the seed and versioned roll record. Replay against a different seed
or deterministic estimator must fail the existing identity check.

The line-active population is a prefix of the transported primaries, and
can differ from the continuum population. The roll RNG must not renumber
or restrict IDs to that prefix before the ordinary kernel applies its
`elec_id < Ne_lines` mask. Both populations keep their global normalizers.

The native per-hkl route now caches only orientation-independent chi/U
coupling rows in the existing bounded streamed fiber table cache. The cache
key includes exact tabulation-grid content (SHA-256), crystal, ordered
reflections, projected thermal values, anomalous-factor choice, backend and
precision. It does not include roll, history population, mosaic or detector
orientation. Tables are read-only in the accumulation route. OOM eviction
uses the existing cache budget and eviction machinery; no RNG state is cached.
Legacy deterministic scoring is unchanged.

## Qualification

Run the CPU integration tests with exact bin means and the production
Fourier reduction. Test both unequal-population directions, HDF5 replay,
small artifact block limits and in-memory scorer rollback. Rollback is not
an end-to-end persisted checkpoint/restart test.

For a remote GPU, the executable check writes a lossless HDF5 trajectory,
JSON metrics and NPZ spectra (the `checks/` directory is shipped by remote
sync, `tests/` is not). The check compares retained, live and replayed spectra
including characteristic and BremsLib continuum, count metadata, matched
continuum record and rejected mismatched case identities. Its FP32 engineering
thresholds are not a physics-accuracy certification.

```sh
pixi run -e nvidia python checks/fiber_history_replay.py \
  --histories 64 --brem-histories 96 --blocks 17 41 --replay-rows 51 119 \
  --output scratch/fiber-roll/history-replay-smoke
```

The first validation gates are fixed-grid only. Automatic grid certification,
adaptive stopping, jointly sampled mosaicity, persisted checkpoint/restart,
and cascades/lineage renumbering remain outside this slice.
