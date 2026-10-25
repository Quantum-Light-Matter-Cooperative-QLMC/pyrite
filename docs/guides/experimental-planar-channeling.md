# Experimental planar quantum channeling and sources

:::{warning}
**Experimental — not rigorously validated.** The planar channeling solver,
quantum-source projection and channeling-radiation predictions are restricted
models. A successful run or numerical convergence does not establish
experimental accuracy. Use independent convergence and physical comparisons
before interpreting predicted rates or line shapes.
:::

Planar channeling evolves an electron's transverse density matrix in the Bloch
basis of a selected crystal-plane family. In supported `simulate` runs, a
channeling zone hands dechanneled electrons to ordinary Monte Carlo transport.
**Channeling radiation (CR)** is recorded separately from PXR/CBS,
characteristic radiation and bremsstrahlung; it is not Cherenkov radiation.

## Run planar channeling

Install the material tables, including SBETHE, and inspect the packaged example:

```bash
pyrite tables fetch
pyrite tables fetch sbethe
pyrite profile show channeling_demo
pyrite run channeling_demo
pyrite app analysis launch silicon
```

The example uses a 17 MeV beam through a 2 µm silicon slab with a forward
detector. Clone it to change the beam, target, detector or numerical settings:

```bash
pyrite profile create my_channeling --from channeling_demo
pyrite physics set my_channeling \
  --channeling-model planar --channeling-plane 1 -1 0 \
  --radiative-model uncoupled --emission incoherent
pyrite run my_channeling
```

The equivalent Python selection is `Physics(channeling_model="planar",
channeling_plane=(1, -1, 0), radiative_model="uncoupled")`. For example:

```python
import numpy as np
import pyrite as pr

result = pr.simulate(
    pr.Beam(energy_keV=17_000, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
    pr.Slab(
        "silicon",
        thickness_ang=20_000,
        tilt_deg=3,
        tilt_azim_deg=45,
        footprint=pr.Footprint(5, 5),
    ),
    pr.Detector(
        0,
        acceptance=pr.SensorAcceptance(
            distance_mm=1_000,
            width_mm=3,
            height_mm=3,
        ),
    ),
    physics=pr.Physics(
        channeling_model="planar",
        channeling_plane=(1, -1, 0),
        radiative_model="uncoupled",
    ),
    numerics=pr.Numerics(
        cxr_trials=200,
        brem_trials=200,
        energy_bins=pr.EnergyBins(
            cxr=np.arange(1_000, 24_000, 50),
            brem=np.arange(1_000, 17_000_000, 500),
        ),
    ),
)
cr = result.channeling  # photons / (incident electron · eV · sr), or None
combined = result.combined_cxr()  # includes CR
```

The channeling grid and line coverage matter: tune
`channeling_basis_size`, `channeling_bloch_cells`, `channeling_dz_ang` and
`channeling_zone_max_ang` with independent convergence checks. Profile runs
persist a `channeling.h5` component that can be selected in the analysis app.

## Specify and inspect quantum entrance states

`TransverseStateSpec` describes a transverse wavefunction or an **incoherent**
mixture separately from the classical beam spot and divergence. A coherent
superposition retains complex relative phases; a mixture adds *densities*, not
amplitudes. Specifications can be serialized with `to_dict()` and
`from_dict()`; sampled arrays reside separately in checked `.npy` files.
Lengths use Å, momenta Å⁻¹, energies eV and angles radians.

```python
import pyrite as pr

plus = pr.TransverseStateSpec.recipe("laguerre_gauss", ell=1, p=0, waist_ang=80)
minus = pr.TransverseStateSpec.recipe("laguerre_gauss", ell=-1, p=0, waist_ang=80)
superposition = pr.TransverseStateSpec.superpose([(1, plus), (1j, minus)])
mixture = pr.TransverseStateSpec.mixture([(3, plus), (1, minus)])
restored = pr.TransverseStateSpec.from_dict(superposition.to_dict())
```

Other analytic recipes include Gaussian, Hermite–Gaussian, Bessel–Gaussian,
Airy and aperture-probe states. Most are **diagnostic-only**, not sources for
production `simulate` runs. The restricted experimental `Beam(state=...)`
path supports a centered one-dimensional Gaussian with lattice-averaged impact
for compatible planar scenes. Do not assume a source accepted by a standalone
diagnostic is accepted by a transport scene.

To project and propagate a portable 1D source *without* running transport:

```python
from pyrite.planar_source import propagate_planar_source, score_planar_source

source = pr.TransverseStateSpec.recipe("gaussian", waist_ang=1, dimension=1)
setup = pr.QuantumSetup(
    region=pr.QuantumRegion(kind="planar", plane_hkl=(1, 1, 0)),
    dynamics=pr.QuantumDynamics(kind="channeling"),
    numerics=pr.QuantumNumerics(basis_size=8, bloch_cells=4, step_ang=50),
)
entry = setup.project_state(
    source,
    crystal="silicon",
    kinetic_energy_eV=17e6,
    incidence_rad=0,
    impact="fixed",
)
evolution = propagate_planar_source(entry, depth_ang=500)
spectrum = score_planar_source(
    entry,
    depth_ang=500,
    photon_energy_eV=np.linspace(1_000, 50_000, 500),
    theta_rad=0,
    absorption=False,
)
print(entry.retained_population, evolution.trace[-1], spectrum.spectrum.shape)
```

`incidence_rad` is the effective incidence relative to the crystal planes,
not the laboratory target tilt. `impact="fixed"` retains position-dependent
coherence; `impact="averaged"` averages over the lattice position.
`retained_population`, `free_population` and `outside_population` describe
**initial basis projection**, not physical dechanneling or escape. The
`score_planar_source` spectrum is per incident electron and is not a calibrated
detector reading. Planar reduction of 2D states does not preserve full electron
or photon orbital-angular-momentum information.

## Supported and refused paths

Planar production currently requires a supported crystal/plane, a single
non-mosaic slab, fixed-count histories, and an appropriate bound-state energy
range. It refuses adaptive precision, trajectory capture, block transport,
coherent-emission combinations, secondary/coupled radiative transport, and
unsupported geometries or beam inputs. The channeling zone and source scoring
use the CPU; subsequent native transport may use CUDA.

The source assumes a mean electron energy and simplified photon attenuation;
finite side-face escape, accumulated in-plane angular diffusion and
instrument-resolution broadening are not fully modeled. Check sensitivity to
basis size, Bloch cells, step size, beam incidence, crystal data and photon
energy grid. Treat rates and linewidths as experimental predictions requiring
independent validation.

For postprocessing, see [Working with results](working-with-results.md) and
for model/evidence limits, see the [channeling validation ledger](../validation/ledger-channeling.md).
