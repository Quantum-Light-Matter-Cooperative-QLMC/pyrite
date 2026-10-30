"""Public, response-free temporal PXR/CBS intensity profiles."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class TemporalIntensity:
    """Paired source ``I(t)`` profiles on a common time axis.

    ``time_fs`` is the detector-arrival-time coordinate in femtoseconds.
    ``incoherent`` and optional ``coherent`` contain photons per incident
    electron, per steradian, per femtosecond. They precede any detector
    response, and coherent time bins can be locally signed by the unbiased
    cross-electron Monte Carlo estimator. Do not clip negative bins.
    """

    time_fs: np.ndarray
    incoherent: np.ndarray
    coherent: np.ndarray | None = None

    def __post_init__(self) -> None:
        time = np.asarray(self.time_fs, dtype=np.float64)
        intensity = np.asarray(self.incoherent, dtype=np.float64)
        if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
            raise ValueError("time_fs must be a finite one-dimensional grid")
        if not np.all(np.diff(time) > 0):
            raise ValueError("time_fs must be strictly increasing")
        if intensity.shape != time.shape or not np.all(np.isfinite(intensity)):
            raise ValueError("incoherent intensity must be finite and match time_fs")
        object.__setattr__(self, "time_fs", time)
        object.__setattr__(self, "incoherent", intensity)
        if self.coherent is not None:
            coherent = np.asarray(self.coherent, dtype=np.float64)
            if coherent.shape != time.shape or not np.all(np.isfinite(coherent)):
                raise ValueError("coherent intensity must be finite and match time_fs")
            object.__setattr__(self, "coherent", coherent)


__all__ = ["TemporalIntensity"]
