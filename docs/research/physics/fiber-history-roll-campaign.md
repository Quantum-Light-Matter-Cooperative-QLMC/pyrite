# Experimental campaign history rolls (#1527, Phase IIb)

`FiberAverage(cutoff_invang=8, roll_sampling="history")` selects one uniform
sector angle per global incident primary. `roll_sampling="quadrature"` is the
unchanged default. Profile option: `--cxr-fiber-roll-sampling history`.
`azimuth_count` is unused in history mode and omitted from that case identity.
The versioned record carries `roll_sampling` and `roll_rng`; old records remain
byte-for-byte equivalent as canonical JSON (key order is not significant).

The case seed roots a dedicated SeedSequence spawn path `(0xF1527,)` and the
existing two-hash transport counter algorithm, draw 0, addressed by global
primary ID. A sparse vector of represented primary IDs is evaluated on the
host, and rotations and polarization stay on device. The seed object is
transient: it is never persisted as a case field or checkpoint payload.
Missing primaries remain in the normal denominator. IDs are not renumbered.
The continuum's reflection set and thermal tensor are unchanged.

Initial envelope: fixed counts, final explicit grid, ordinary HOPG transport,
incoherent bin-mean lines, deterministic mosaic quadrature only. Automatic
line-grid certification, adaptive stopping, cone split, reciprocal miscut,
truncation audit and direct domain/selection scoring are not enabled for
history rolls. Replay/streamed restart and mixed-population behavior require
validation before a production qualification claim. No defaults change.

Run the same-realization campaign scorer parity check:

```sh
pixi run -e nvidia python checks/fiber_randomized_rolls.py \
  --histories 4096 --cutoff 8 --tilt 50 --step 1 --energy-max 13000 \
  --quadratures --reference 4 --shifted --strata --orientation-seeds 101 \
  --native --native-blocks 3 --production-history-parity \
  --output scratch/fiber-roll/history-campaign
```

Compare `production_history_total` and `production_history_expected_total`.
Repeat on CPU, CUDA, and ROCm; repeat with deterministic mosaic (`--mosaic
--mosaic-route mc --mosaic-nodes 3`). The reference is the explicit-angle
native scorer with the same case seed, not a quadrature or a different RNG
realization. This tests the campaign line seam, not a complete live campaign.
Next: end-to-end fixed-N run, equal/unequal populations, retained vs streamed,
restart/replay, isolated memory and timing; only then qualify automatic grid
and adaptive precision. Per-hkl setup/table reuse and jointly sampled mosaic
are separate future optimizations.
