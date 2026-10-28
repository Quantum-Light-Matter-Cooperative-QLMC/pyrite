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
single shared registry. Existing numeric inputs keep the units encoded by their
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
pose = PlanarPose(center_mm=(0, 0, 10 * ureg.cm))
pixels = PixelGrid(shape=(256, 256), pitch_mm=(55 * ureg.um, 55 * ureg.um))
acquisition = Acquisition(
    exposure_s=500 * ureg.ms,
    measured_edges_eV=(0.1 * ureg.keV, 0.2 * ureg.keV, 0.3 * ureg.keV),
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
See the [beam-energy contract](../../repo-design/physical-units.md#beam-energy-quantity-interface)
for scalar/array, replacement and serialization details. The empty `units`
installation extra remains a compatibility spelling.

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
