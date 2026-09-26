# Shell soft/hard secondary electron transport

`Validation: shell-secondary-transport` — status `unverified`. Physics:
[secondary electron transport](../../physics/beam-transport/shell-soft-hard-transport.md#secondary-electron-transport).
Ledger row: [`shell-secondary-transport`](../ledger-transport-background.md#shell-secondary-transport).

This record covers `simulate_trajectories(secondary_threshold_eV=...)` in the
opt-in shell soft/hard mode: the in-kernel secondary launch direction, the
generation-batched cascade, its track identity, random-number keying and
energy and particle balance, and the radiation scored on secondary tracks.

## Planned checks

- Primary-only compatibility: `secondary_threshold_eV=None` is bit-for-bit
  the transport before the option existed.
- Kernel parity: the in-kernel secondary direction equals
  `shell_sampling.shell_collision_world_directions` for the same draws.
- Energy and particle balance per run and per primary history.
- Determinism across batch sizes and cores, and CPU/CUDA aggregate agreement.
- Per-shell characteristic-yield invariance of the primaries' track-length
  estimator under the vacancy bookkeeping.
- Cascade spectrum: launched secondary energies against the Møller/shell $W$
  spectrum.
- Threshold convergence of backscatter, transmission, depth-dose,
  characteristic line yields and bremsstrahlung.
