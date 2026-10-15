"""Read the opt-in primary-only TestEm5 adapter's diagnostic scoring log."""

import numpy as np
from numpy.typing import NDArray


def parse_primary_log(
    text: str, n_primaries: int
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Parse primary terminals and grouped photon inputs; require full histories.

    Accept the single-worker Geant4 MT prefix and sequential output. Every
    primary must terminate once; secondaries must already have been disabled.
    """
    ends, photon_ids, energies = [], [], []
    for raw in text.splitlines():
        line = raw.removeprefix("G4WT0 > ")
        if line.startswith("PRIMARY_END "):
            _, event, terminal, energy = line.split()
            ends.append((int(event), int(terminal), float(energy)))
        elif line.startswith("EBREM "):
            fields = line.split()
            if len(fields) != 10:
                raise ValueError("invalid primary bremsstrahlung record")
            photon_ids.append(int(fields[1]))
            energies.append(float(fields[3]))
    if n_primaries < 2 or sorted(e[0] for e in ends) != list(range(n_primaries)):
        raise ValueError("incomplete or duplicate primary terminal records")
    terminals = np.asarray(ends, dtype=np.float64)
    pids = np.asarray(photon_ids, dtype=np.int64)
    k = np.asarray(energies, dtype=np.float64)
    if (
        not np.all(np.isin(terminals[:, 1], [4, 5, 6, 7]))
        or not np.all(np.isfinite(terminals[:, 2]))
        or np.any(terminals[:, 2] < 0)
        or np.any(pids < 0)
        or np.any(pids >= n_primaries)
        or not np.all(np.isfinite(k))
        or np.any(k <= 0)
    ):
        raise ValueError("invalid terminal state or photon primary/energy")
    return terminals, pids, k
