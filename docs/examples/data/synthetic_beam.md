---
orphan: true
---

# Synthetic electron distribution

**SYNTHETIC DEMONSTRATION.** `synthetic_beam.gdf` was constructed for this
tutorial by `make_fixture` in `docs/examples/gpt_gdf_beams.py`. It contains no
experimental, facility, proprietary GPT or third-party particle data. The
original generator and its output are distributed under this repository's
{download}`Academic Software License <../../../LICENSE.txt>`; retain that license
when redistributing.

The GDF `creator="GPT"` header is required by PyRITE's format validator. It
identifies the expected record format here, **not** the origin of this data.
The file is generated with EasyGDF (a PyRITE dependency), without running or
requiring General Particle Tracer. A fixed zero header timestamp makes the
bytes repeatable in the recorded environment. SHA-256 is recorded in
`gpt_gdf_beams.json`; docs builds check the input as well as the example source.

One time block at 1 ns contains 96 electron records at lab z = 0.1 m. Seed 42
generates Gaussian transverse positions (20 µm standard deviation), correlated
x slope and energy, and independent y slope. Nonuniform positive `nmacro`
values represent a total absolute charge of 1 pC. The original records have
energy near 30 keV; no measured emittance or accelerator performance is implied.
Positions are meters, beta components are dimensionless velocities divided by
c, masses are kg, and signed electron charges are coulombs. `G` is the Lorentz
factor. All arrays are explicitly supplied.

From the checkout, explicitly regenerate the input, then the simulated output:

```bash
uv run python docs/examples/gpt_gdf_beams.py --make-fixture
MPLCONFIGDIR=/tmp/pyrite-matplotlib PYRITE_MC_BACKEND=cpu \
  uv run python docs/_tutorials.py --regenerate gpt_gdf_beams
```

The example uses 64 CPU histories per radiation component and two reflection
families. It illustrates import and plotting, not numerical convergence.
