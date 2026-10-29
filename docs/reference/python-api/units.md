# Units and quantities

PyRITE accepts bare numbers in each field's documented canonical unit. Selected
configuration boundaries also accept compatible Pint quantities, normalized to
ordinary numeric magnitudes before simulation or persistence. Use the shared
registry rather than a new Pint registry:

```python
from pyrite.units import ureg

energy = 0.06 * ureg.MeV
```

For the full field inventory and serialization contract, see the
[physical-units reference](../../repo-design/physical-units.md).

## Physical quantity inputs

Pint is included in the core installation. Construct quantities through PyRITE's
single shared registry. To convert an existing script, follow
[Migrate scripts to physical quantities](../../guides/physical-quantities.md).
Existing numeric inputs keep the units encoded by their
field names. Supported configuration boundaries convert compatible quantities
to those same numeric values before case lowering or persistence.

```python
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.observations import Acquisition
from pyrite.detectors import PixelGrid, PlanarPose
from pyrite.units import ureg

beam = Beam(
    kinetic_energy=0.06 * ureg.MeV,
    transverse=Spot(),
    longitudinal=PointBunch(),
)
pose = PlanarPose(center=[0, 0, 10] * ureg.cm)
pixels = PixelGrid(shape=(256, 256), pitch=[55, 55] * ureg.um)
acquisition = Acquisition(
    exposure=500 * ureg.ms,
    measured_edges=[0.1, 0.2, 0.3] * ureg.keV,
)
```

The normal `import pyrite` and numeric beam paths do not load Pint. Incompatible dimensions,
non-real magnitudes, and dimensionless values supplied to angular fields fail
at configuration construction. Stored values remain ordinary Python numbers
and NumPy arrays; quantities do not enter transport kernels or checkpoints.
See the [unit boundary and field inventory](../../repo-design/physical-units.md)
for supported fields, profile string syntax, serialization policy, and measured
overhead.

Unit suffixes describe the numeric value returned by current fields, even when
assignment uses a different compatible unit: `Beam.energy_keV` always reads
back as a magnitude in keV. `Beam.kinetic_energy` returns a fresh quantity in
keV; array magnitudes are copied. Its constructor keyword requires a real,
finite positive energy quantity and conflicts with an explicit `energy_keV`.
Use `beam.with_kinetic_energy(80 * ureg.keV)` for quantity-aware replacement;
`dataclasses.replace` continues to accept the numeric `energy_keV` field.
`PlanarPose(center=...)` takes a three-component length and
`PixelGrid(pitch=...)` a two-component `(pitch_y, pitch_x)` length; they read
back as fresh mm quantities (`center`, `pitch`, `active_size`) and replace via
`with_center` / `with_pitch`. `PlanarPose.from_observation` accepts
`distance`, `polar`, `azimuth`, `roll` and `offset` quantities; angles must
carry plane-angle units. Direction vectors stay dimensionless tuples.
`Physics` takes keyword-only `inelastic_cutoff`, `secondary_threshold` and
`radiative_cutoff` energies and a three-component `transition_lattice_origin`
length; each reads back as a fresh eV/Å quantity (or `None`) and replaces via
`with_*`. `EnergyBins.cxr` / `brem` keep returning numeric eV arrays;
`cxr_energies` / `brem_energies` return fresh quantities and
`with_cxr_energies` / `with_brem_energies` replace a grid. Planar channeling
lengths use `QuantumNumerics.step_length` and `QuantumRegion.zone_max_length`;
the flat `Numerics.channeling_*` fields stay numeric.
`PlanarDetector(size=...)` and `FilterPlate(thickness=..., size=...)` take a
positive two-component `(width_x, height_y)` length and a positive scalar
thickness; they read back as fresh mm quantities and replace via `with_size` /
`with_thickness`. `PlanarDetector.with_size` keeps a legacy embedded response
and scorer; pixelated detectors still require the size to match the grid.
`Acquisition` follows the same pattern: keyword-only `exposure` (positive
scalar time), `measured_edges` (one-dimensional energy array) and
`hit_threshold` (nonnegative scalar energy) read back as fresh quantities in s
and eV, with `with_exposure`, `with_measured_edges` and `with_hit_threshold`
replacement. `Acquisition.uniform` takes `exposure`, `minimum`, `maximum`,
`bin_width` and `hit_threshold`; `Acquisition.native` and `NativeAcquisition`
take `exposure` and `hit_threshold`. Numeric `*_s` / `*_eV` spellings keep
their meaning, and acquisition identities and stored observations are
unchanged.
See the [beam-energy contract](../../repo-design/physical-units.md#beam-energy-quantity-interface)
for scalar/array, replacement and serialization details. The empty `units`
installation extra remains a compatibility spelling.

## Quantity sweep paths

`Sweep.axes` accepts semantic quantity properties whose owner provides a
`with_<name>` replacement method. Each value follows that method's unit,
shape and validation rules. Nested and indexed owner paths work as well:

```python
from pyrite import Beam, Scene, Sweep
from pyrite.beams import PointBunch, Spot
from pyrite.geometry import Footprint, Slab

scene = Scene(
    Beam(kinetic_energy=30 * ureg.keV, transverse=Spot(), longitudinal=PointBunch()),
    Slab(
        "hopg", thickness=200 * ureg.nm, footprint=Footprint(width=5 * ureg.mm, height=5 * ureg.mm)
    ),
)
sweep = Sweep(
    scene,
    axes={
        "beam.kinetic_energy": [30 * ureg.keV, 0.045 * ureg.MeV],
        "target.thickness": [200 * ureg.nm, 0.5 * ureg.um],
    },
)
```

Expansion produces numeric scene storage through the owning replacement
methods; it leaves the base scene unchanged. Axis labels include the supplied
quantity's unit. Legacy numeric paths such as `beam.energy_keV` keep their
existing numeric units and validation. Optimization domains can target the same
semantic properties; see [quantity parameter paths](../optimization-studies.md#quantity-parameter-paths).

## Shared registry and quantity constructor

`ureg` owns the shared unit definitions; `Q_` constructs quantities bound to
that registry. Both are advanced-stable exports. Do not mutate the registry's
definitions after import. Their values are omitted from generated reference to
avoid expanding registry internals.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :template: autosummary/public-data.rst
   :nosignatures:

   pyrite.units.ureg
   pyrite.units.Q_
```
