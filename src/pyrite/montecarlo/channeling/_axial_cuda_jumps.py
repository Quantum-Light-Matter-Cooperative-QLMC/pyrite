"""CUDA jump operations with per-electron Philox streams.

Complex128 arithmetic, statistically equivalent to CPU, not matched draws.
Validation: cr-axial-realspace-dissipator
"""

from __future__ import annotations

import cupy as cp

from ...materials.channeling.transverse import HBAR2_2M_EV_ANG2
from ._axial_cuda_rng import ElectronRNG
from .axial_mcwf import _PROPOSALS, MAX_REJECTION_TRIALS
from .axial_mcwf import _SHELL_EDGES as _HOST_SHELL_EDGES

_SHELL_EDGES = cp.asarray(_HOST_SHELL_EDGES)


def electron_rng(seed, electron):
    return ElectronRNG(seed, electron)


_CATEGORICAL = cp.RawKernel(
    r"""extern "C" __global__ void categorical(
  const double* w, const double* u, int n, long long* result) {
  double total=0.0;
  for(int i=0; i<n; ++i) total+=w[i];
  if(!(total>0.0)) { result[0]=-1; return; }
  double target=u[0]*total, cumulative=0.0;
  for(int i=0; i<n; ++i) {
    cumulative+=w[i];
    if(cumulative>target) {result[0]=i; return;}
  }
  result[0]=n-1;
}""",
    "categorical",
)


def _choose(rng, weights):
    import numpy as np

    weights = cp.ascontiguousarray(weights, dtype=cp.float64)
    result = cp.empty((), dtype=cp.int64)
    _CATEGORICAL((1,), (1,), (weights, rng.random(), np.int32(weights.size), result))
    chosen = int(result)
    if chosen < 0:
        raise ValueError("no positive weight to sample")
    return chosen


class DeviceJumps:
    """Same jump equations as AxialMCWF; only scalar control reaches Python."""

    def __init__(self, solver, dissipator):
        self.dissipator = dissipator
        self.grid = dissipator.grid
        self.gamma = solver.gamma
        self.hbar_v = solver.hbar_v
        self._rates = {k: cp.asarray(v) for k, v in solver._rates.items()}

    def _half_kinetic(self, tilt, step):
        kx = self.grid.kx[:, None] + tilt[0]
        ky = self.grid.ky[None, :] + tilt[1]
        energy = HBAR2_2M_EV_ANG2 / self.gamma * (kx**2 + ky**2)
        return cp.exp(-0.5j * step * energy / self.hbar_v)

    def _jump(self, state, tilt, rng):
        """Draw one jump: ``(kind, normalized state or None for a sink, tilt)``."""
        density = cp.abs(state) ** 2
        shares = [self._expect(density, rate) for rate in self._rates.values()]
        kind = list(self._rates)[_choose(rng, cp.stack(shares))]
        if kind == "sink":
            return (kind, None, tilt)
        if kind == "tds":
            state = self._tds_jump(state, density, rng)
        else:
            state, tilt = self._electronic_jump(state, density, tilt, rng)
        return (kind, state / cp.sqrt(cp.sum(cp.abs(state) ** 2) * self.grid.cell_area_ang2), tilt)

    def _expect(self, density: cp.ndarray, rate: cp.ndarray) -> float:
        return cp.sum(density * rate) * self.grid.cell_area_ang2

    def _tds_jump(self, phi, density, rng):
        d = self.dissipator
        weights = cp.einsum("sij,ij->s", d.tds_source_rates, density)
        source = _choose(rng, weights)
        src = d.tds_sources[source]
        sigma = cp.sqrt(src.displacement_ang2)
        edges = sigma * _SHELL_EDGES
        coefficients = d.tds_weight_spectrum(source, density)
        envelope = cp.maximum(d.source_envelope(source, coefficients, edges), 0.0)
        shell_p = -cp.diff(cp.exp(-0.5 * _SHELL_EDGES**2))
        proposal = shell_p * envelope
        for _ in range(MAX_REJECTION_TRIALS // _PROPOSALS):
            shells = cp.searchsorted(
                cp.cumsum(proposal), rng.random(_PROPOSALS) * proposal.sum(), "right"
            )
            shells = cp.minimum(shells, proposal.size - 1)
            lo = cp.exp(-0.5 * _SHELL_EDGES[shells] ** 2)
            hi = cp.exp(-0.5 * _SHELL_EDGES[shells + 1] ** 2)
            radius = sigma * cp.sqrt(-2.0 * cp.log(hi + (lo - hi) * rng.random(_PROPOSALS)))
            angle = 2.0 * cp.pi * rng.random(_PROPOSALS)
            offsets = cp.stack([radius * cp.cos(angle), radius * cp.sin(angle)], 1)
            uniforms = rng.random(_PROPOSALS)
            weights = d.tds_weight(coefficients, offsets)
            if cp.any(weights > envelope[shells] * (1.0 + 1e-12)):
                raise RuntimeError("TDS displacement weight exceeded its rejection envelope")
            accepted = cp.flatnonzero(uniforms * envelope[shells] < weights)
            if accepted.size:
                return d.tds_phase(source, offsets[accepted[0]]) * phi
        raise RuntimeError("TDS displacement rejection sampling did not converge")

    def _electronic_jump(self, phi, density, tilt, rng):
        d = self.dissipator
        weights = d.electronic_weights * cp.maximum(d.electronic_expectations(density), 0.0)
        node = _choose(rng, weights)
        q = float(d.electronic_q_inv_ang[node])
        azimuth = 2.0 * cp.pi * rng.random()
        kicked = (tilt[0] + q * cp.cos(azimuth), tilt[1] + q * cp.sin(azimuth))
        return (cp.sqrt(d.smoothed_density(q)) * phi, kicked)
