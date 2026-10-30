"""Progress events of in-process simulation phases (#776).

Defined below the Monte Carlo runner, which emits them, and re-exported by
:mod:`pyrite.phases`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SimulationProgress:
    """One progress update of a running simulation.

    ``phase="transport"`` reports electrons finished (``stage="electrons"``)
    after each transport batch; ``phase="spectrum"`` reports completed
    spectrum stages (``"lines"``, ``"characteristic"``, ``"brem"``) out of
    ``total``. Updates arrive on the calling thread between GPU launches; a
    single long kernel or spectrum stage produces none while it runs.
    """

    phase: Literal["transport", "spectrum"]
    stage: str
    done: int
    total: int

    @property
    def fraction(self) -> float:
        """``done / total`` of this phase, in ``[0, 1]``."""
        return self.done / self.total if self.total else 1.0
