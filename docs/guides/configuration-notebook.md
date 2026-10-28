# Edit a campaign profile interactively

Use the configuration app to duplicate a catalog profile, change its beam,
target, detector and numerical settings, then save it for a CLI run. The editor
manages **campaign profiles**, not arbitrary Python `Scene` objects. For
in-memory scene construction, see the
[beam and target tutorial](../notebooks/designing/04-beams-crystals-target-geometry.md).

## Launch the editor

Install `pyrite-mc[apps]`, then launch:

```bash
pyrite app configuration launch
```

To edit an independent catalog, select it explicitly:

```bash
pyrite --catalog /path/to/catalog app configuration launch
```

Bundled entries are read-only. Changes are **staged** until you select
**Save catalog changes**; changing controls never starts a simulation.

## Create an HOPG profile

1. In **Overview**, duplicate `quickstart` as `hopg-configured`. Do not edit
   the bundled profile in place.
2. In **Scene and Materials**, keep `hopg` and set the energy to **30 keV**,
   thickness to **20,000 Å** (2 µm), incidence to **45°** and azimuth to
   **180°**. Choose one thickness representation, not both layers and Å.
3. In **Beam**, load `example`. If you need changes, create a private named
   copy; editing a shared preset would affect other profiles.
4. In **Detectors**, retain the scalar `reference` detector or attach a
   profile-local copy. Keep at least one explicit detector.
5. In **Physics**, select `incoherent` emission; in **Numerics**, set
   CXR trials to **300** and bremsstrahlung trials to **150**. Leave unrelated
   settings inherited.
6. In **Review and Save**, inspect the resolved settings, validation messages
   and TOML diff, then choose **Save catalog changes**.

The editor stages multiple profiles together. Close it only after saving any
changes you want to keep. An override checkbox controls whether a value is
written explicitly; an unchecked suggestion does not alter the simulation.

## Run and inspect the result

From the same workspace and catalog used by the editor, run:

```bash
pyrite run hopg-configured
pyrite checkpoint list
```

Use **Outputs and Exports** to select a saved checkpoint by its exact stem and
download stored source arrays as CSV or NPZ. Exporting arrays does not modify
what the simulation computed or retained. To record electron histories, enable
trajectory capture before running; cached simulations need an explicit
`--recompute` to acquire new histories.

The editor also offers a TOML snapshot download; relative CIF and GDF assets
must accompany it when transferred. For complete field descriptions, overrides,
detector-ID migration, saving conflicts and export semantics, consult the
[configuration app reference](../reference/configuration-app.md) and
[campaign configuration contract](../reference/campaign-configuration.md).
