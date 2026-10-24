# Portable quantum transverse source states (#680)

`TransverseStateSpec` is an immutable, versioned description of the electron's
*entrance* transverse wavefunction or incoherent density mixture. This is
independent of the classical entrance spot and angular distribution, and of
the numerical propagation solver. Lengths are in Å, momenta in Å⁻¹, kinetic
energies in eV, and semiangles in radians. The descriptor contains no
arbitrary functions or numerical arrays, and can be serialized to JSON.

```python
import json
import pyrite as pr
from pyrite import states

plus = pr.TransverseStateSpec.recipe("laguerre_gauss", ell=1, p=0, waist_ang=80)
minus = pr.TransverseStateSpec.recipe("laguerre_gauss", ell=-1, p=0, waist_ang=80)
coherent = pr.TransverseStateSpec.superpose([(1, plus), (1j, minus)])
ensemble = pr.TransverseStateSpec.mixture([(3, plus), (1, minus)])

stored = json.loads(json.dumps(coherent.to_dict()))
assert pr.TransverseStateSpec.from_dict(stored) == coherent
print(coherent.digest)  # canonical SHA256

# For a diagnostic or a supported external solver, reuses #508 evaluators:
entry = coherent.to_runtime()
```

A mixture combines **densities**, not amplitudes; its nonnegative weights are
normalized to sum to one. A coherent superposition retains complex relative
phase and is normalized during runtime evaluation. Supported analytic kinds:
`gaussian`, `hermite_gauss`, `laguerre_gauss`, `bessel_gauss`, `airy`, and
`aperture_probe` (including abTEM-style aberration coefficients, but **not**
arbitrary phase-plate Python callables). Use `from pyrite import states` for
existing runtime factories and diagnostics; the portable spec is distinct.

Sample arrays are stored **outside** the JSON. Supply a 1D or 2D finite,
nonzero floating/complex NumPy array saved as `.npy` (not object arrays):

```python
import numpy as np

np.save("probe.npy", np.ones((16, 16), dtype=np.complex128))
spec = pr.TransverseStateSpec.from_npy(
    "probe.npy",
    spacing_ang=(0.1, 0.1),
    origin_ang=(-0.8, -0.8),
)
metadata = spec.to_dict()  # path, data digest, shape, dtype and sampling grid
wave = spec.to_runtime(base_dir=".")
```

Loading checks the on-disk dtype, shape, finiteness, and SHA-256 of contiguous
sample data. The canonical *scientific* spec digest excludes asset paths, but
includes the array digest and grid conventions. Relative paths need `base_dir`
when materialized; missing/changed files refuse. No sample bytes are embedded
in the configuration and no arbitrary file formats are deserialized.

`Beam(state=...)` stores the typed `TransverseStateSpec`. A restricted,
experimental production adapter now supports centered 1D Gaussian states
with lattice-averaged impact, when the Scene selects planar channeling.
Standalone `Beam.to_spec()` still refuses: it lacks the required crystal
configuration. Every admitted source has a distinct versioned case identity,
momentum-conditioned handoffs, and deterministic CR live/replay reconstruction.
See [conditional source validation](../validation/channeling/cr-planar-conditional-source.md).
Other source recipes remain diagnostic-only; the physical handoff still needs
independent validation and convergence tests.

#624's axial candidate resolver remains separate from the planar diagnostics.
Its continuation branch hardens physical-admission refusals; describing an
axial region still never admits a production run without #623's matching,
source and depth-budget contracts.
