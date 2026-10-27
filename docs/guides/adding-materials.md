# Add a material

Build silica from a formula and a measured bulk density, inspect its elemental
number densities, then register it for later runs. Inspection needs no reference
tables and writes no files.

(amorphous-media)=
(python-build-inspect-then-save)=
## Build and inspect

Run the downloadable Python example with `pyrite-mc` installed:

```{literalinclude} ../examples/adding_materials.py
:language: python
```

Actual output from the composition conversion (not a transport simulation):

```{literalinclude} ../examples/data/adding_materials.txt
:language: text
```

There are twice as many oxygen atoms as silicon atoms. The supplied **2.2 g/cm³**
sets the absolute number density; the formula alone cannot determine it.
Download the {download}`source <../examples/adding_materials.py>` and
{download}`checked output <../examples/data/adding_materials.json>`.

(crystalline-materials)=
## Register for later use

Choose unused names before running these commands. The first creates an
amorphous medium; the second reuses the catalog's silicon crystal:

```bash
pyrite material create amorphous lab-silica --formula SiO2 --density-g-cm3 2.2
pyrite material show lab-silica
pyrite material create crystal lab-silicon --crystal silicon
pyrite material show lab-silicon
```

With the bundled catalog selected, these commands write to your user catalog
layer and refuse to overwrite existing names. An explicitly selected complete
catalog is edited in place. Python callers can persist the inspected definition
with `pyrite.materials.save_material`; see the
[build/save contract](../reference/material-authoring.md#python-build-inspect-then-save).

(tables-remote-runs-and-maintainers)=
## Use the definition

Amorphous media can supply layers, substrates, filters and electron transport;
registration does not give them crystal PXR/CBS physics. A crystal definition
specifies structure, while geometry and campaign membership are configured
separately. Registration neither generates interaction tables nor validates a
physical model.

For a new CIF, mixtures, density checks, managed assets and maintainer commands,
use [material authoring contracts](../reference/material-authoring.md).
[Configure a campaign](configuration-cookbook.md) to select materials and
[prepare reference tables](getting-started.md#install-and-fetch-reference-data)
before transport.
