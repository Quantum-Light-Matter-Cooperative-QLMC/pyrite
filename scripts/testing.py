from time import perf_counter

import numpy as np

from pyrite.montecarlo.transport import _rotate_directions, _rotate_directions_numpy

rng = np.random.default_rng(0)

d = rng.normal(size=(10_000, 3))
d /= np.linalg.norm(d, axis=1)[:, None]

cos_t = rng.uniform(-1.0, 1.0, size=len(d))
phi = rng.uniform(0.0, 2 * np.pi, size=len(d))

old = _rotate_directions_numpy(d, cos_t, phi)
new = _rotate_directions(d, cos_t, phi)

print("max diff:", np.max(np.abs(old - new)))
print("norm error:", np.max(np.abs(np.linalg.norm(new, axis=1) - 1)))

n_trials = 1000
t0 = perf_counter()
for _ in range(n_trials):
    _rotate_directions_numpy(d, cos_t, phi)
dt = perf_counter() - t0
dt /= n_trials
print(f"Numpy single trial execution time (over 1000 trials): {dt * 1e6:0.1f} us")

# warm up Numba compilation
_rotate_directions(d, cos_t, phi)

t0 = perf_counter()
for _ in range(n_trials):
    _rotate_directions(d, cos_t, phi)
dt = perf_counter() - t0
dt /= n_trials
print(f"Numba single trial execution time (over 1000 trials): {dt * 1e6:0.1f} us")
