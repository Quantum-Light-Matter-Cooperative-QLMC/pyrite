"""Counter-addressed Philox4x32-10 for axial CUDA (stream version 1).

The complete 64-bit ``mcwf_stream_key(seed, electron)`` becomes the Philox
key; a monotonically increasing 64-bit scalar-draw index supplies the counter.
Unlike a CuPy generator this allocates no per-SM curand state and never reduces
the identity key to uint32. Algorithm: Random123 Philox4x32-10 (Salmon et al.,
SC11), constants and round mapping pinned by the published zero-key vector.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from .mcwf import mcwf_stream_key

_DRAW = cp.RawKernel(
    r"""
extern "C" __global__ void philox_uniform(
    unsigned long long key, unsigned long long first, unsigned long long n,
    double* output) {
  unsigned long long i = blockDim.x * (unsigned long long)blockIdx.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long draw = first + i, block = draw / 2;
  unsigned int c0 = (unsigned int)block, c1 = (unsigned int)(block >> 32);
  unsigned int c2 = 0, c3 = 0;
  unsigned int k0 = (unsigned int)key, k1 = (unsigned int)(key >> 32);
  for (int round = 0; round < 10; ++round) {
    unsigned long long p0 = 0xD2511F53ULL * c0;
    unsigned long long p1 = 0xCD9E8D57ULL * c2;
    unsigned int next0 = (unsigned int)(p1 >> 32) ^ c1 ^ k0;
    unsigned int next2 = (unsigned int)(p0 >> 32) ^ c3 ^ k1;
    c0 = next0; c1 = (unsigned int)p1;
    c2 = next2; c3 = (unsigned int)p0;
    k0 += 0x9E3779B9U; k1 += 0xBB67AE85U;
  }
  unsigned int a = (draw & 1) ? c2 : c0;
  unsigned int b = (draw & 1) ? c3 : c1;
  output[i] = ((a >> 5) * 67108864.0 + (b >> 6)) / 9007199254740992.0;
}
""",
    "philox_uniform",
)


class ElectronRNG:
    """Small host counter; random values are constructed directly on device."""

    def __init__(self, seed, electron):
        self.key = np.uint64(mcwf_stream_key(seed, electron))
        self.counter = 0

    def random(self, size=None):
        shape = () if size is None else (size,)
        count = 1 if size is None else size
        if self.counter + count > 2**64:
            raise OverflowError("axial CUDA Philox draw counter exhausted")
        out = cp.empty(shape, dtype=cp.float64)
        _DRAW(
            ((count + 127) // 128,),
            (128,),
            (self.key, np.uint64(self.counter), np.uint64(count), out),
        )
        self.counter += count
        return out
