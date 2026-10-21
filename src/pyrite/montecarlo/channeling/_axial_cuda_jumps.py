"""CUDA jump operations with per-electron Philox streams.

Complex128 arithmetic, statistically equivalent to CPU, not matched draws.
Validation: cr-axial-realspace-dissipator
"""

from __future__ import annotations

import math

import cupy as cp

from ...materials.channeling.transverse import HBAR2_2M_EV_ANG2
from ._axial_cuda_rng import ElectronRNG
from .axial_mcwf import _PROPOSALS, MAX_REJECTION_TRIALS
from .axial_mcwf import _SHELL_EDGES as _HOST_SHELL_EDGES

_SHELL_EDGES = cp.asarray(_HOST_SHELL_EDGES)
_SHELL_DECAY = cp.exp(-0.5 * _SHELL_EDGES**2)
_SHELL_PROBABILITY = -cp.diff(_SHELL_DECAY)


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


_ACCEPT = cp.RawKernel(
    r"""extern "C" __global__ void acceptance(
 const double* weights, const double* envelope, const long long* shells,
 const double* uniforms, int n, long long* result) {
 long long first=-1;
 for(int i=0;i<n;++i) {
  double bound=envelope[shells[i]];
  if(weights[i]>bound*(1.0+1e-12)) { result[0]=-2;return; }
  if(first<0 && uniforms[i]*bound<weights[i]) first=i;
 }
 result[0]=first;
}""",
    "acceptance",
)


def _accepted(weights, envelope, shells, uniforms):
    import numpy as np

    result = cp.empty((), dtype=cp.int64)
    _ACCEPT(
        (1,),
        (1,),
        (
            cp.ascontiguousarray(weights),
            envelope,
            shells.astype(cp.int64, copy=False),
            uniforms,
            np.int32(weights.size),
            result,
        ),
    )
    index = int(result)
    if index == -2:
        raise RuntimeError("TDS displacement weight exceeded its rejection envelope")
    return index


_PROPOSE = cp.RawKernel(
    r"""extern "C" __global__ void proposals(
 const double* proposal, const double* decay, const double* draws, double sigma,
 int n, int count, long long* shells, double* offsets) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;
 if(i>=count)return;
 double total=0;
 for(int j=0;j<n;++j)total+=proposal[j];
 double target=draws[i]*total, cumulative=0;
 int shell=n-1;
 for(int j=0;j<n;++j) { cumulative+=proposal[j]; if(cumulative>target){shell=j;break;} }
 shells[i]=shell;
 double lo=decay[shell], hi=decay[shell+1];
 double radius=sigma*sqrt(-2.0*log(hi+(lo-hi)*draws[count+i]));
 double angle=6.283185307179586476925286766559*draws[2*count+i];
 offsets[2*i]=radius*cos(angle);offsets[2*i+1]=radius*sin(angle);
}""",
    "proposals",
)


def _proposals(proposal, draws, sigma):
    import numpy as np

    count = draws.shape[1]
    shells = cp.empty(count, dtype=cp.int64)
    offsets = cp.empty((count, 2), dtype=cp.float64)
    _PROPOSE(
        ((count + 63) // 64,),
        (64,),
        (
            proposal,
            _SHELL_DECAY,
            draws,
            np.float64(sigma),
            np.int32(proposal.size),
            np.int32(count),
            shells,
            offsets,
        ),
    )
    return shells, offsets


class DeviceJumps:
    """Same jump equations as AxialMCWF; only scalar control reaches Python."""

    def __init__(self, solver, dissipator):
        self.dissipator = dissipator
        self.grid = dissipator.grid
        self.gamma = solver.gamma
        self.hbar_v = solver.hbar_v
        self._rates = {k: cp.asarray(v) for k, v in solver._rates.items()}
        self._stacked_rates = cp.stack(list(self._rates.values()))

    def _half_kinetic(self, tilt, step):
        kx = self.grid.kx[:, None] + tilt[0]
        ky = self.grid.ky[None, :] + tilt[1]
        energy = HBAR2_2M_EV_ANG2 / self.gamma * (kx**2 + ky**2)
        return cp.exp(-0.5j * step * energy / self.hbar_v)

    def _jump(self, state, tilt, rng):
        """Draw one jump: ``(kind, normalized state or None for a sink, tilt)``."""
        density = cp.abs(state) ** 2
        shares = cp.sum(self._stacked_rates * density[None], axis=(1, 2)) * self.grid.cell_area_ang2
        kind = list(self._rates)[_choose(rng, shares)]
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
        sigma = math.sqrt(src.displacement_ang2)
        edges = sigma * _SHELL_EDGES
        coefficients = d.tds_weight_spectrum(source, density)
        envelope = cp.maximum(d.source_envelope(source, coefficients, edges), 0.0)
        proposal = _SHELL_PROBABILITY * envelope
        for _ in range(MAX_REJECTION_TRIALS // _PROPOSALS):
            draws = rng.random(4 * _PROPOSALS).reshape(4, _PROPOSALS)
            shells, offsets = _proposals(proposal, draws, sigma)
            uniforms = draws[3]
            weights = d.tds_weight(coefficients, offsets)
            accepted = _accepted(weights, envelope, shells, uniforms)
            if accepted >= 0:
                return d.tds_phase(source, offsets[accepted]) * phi
        raise RuntimeError("TDS displacement rejection sampling did not converge")

    def _electronic_jump(self, phi, density, tilt, rng):
        d = self.dissipator
        weights = d.electronic_weights * cp.maximum(d.electronic_expectations(density), 0.0)
        node = _choose(rng, weights)
        q = float(d.electronic_q_host[node])
        azimuth = 2.0 * cp.pi * rng.random()
        kicked = (tilt[0] + q * cp.cos(azimuth), tilt[1] + q * cp.sin(azimuth))
        return (cp.sqrt(d.smoothed_density(q)) * phi, kicked)
