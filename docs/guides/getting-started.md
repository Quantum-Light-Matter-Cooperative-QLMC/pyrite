# Getting started

Install the released package, run one small CPU simulation, and locate its output.
No source checkout is required. PyRITE needs Python 3.13 or newer and
[uv](https://docs.astral.sh/uv/); the [README installation matrix](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite#install)
covers library installs and accelerator extras.

## Install and fetch reference data

```bash
uv tool install pyrite-mc
pyrite --version
pyrite --help
pyrite tables fetch
```

If `pyrite` is not on your `PATH`, run `uv tool update-shell` and restart the
shell. The base package is CPU-only. `tables fetch` installs pinned EEDL, EADL,
EPDL, SBETHE, ELSEPA, and BremsLib datasets (about 100 MB) from public GitHub
Releases; no GitHub token is required. Data go to the selected workspace;
see [output-home configuration](configuration-cookbook.md#choose-an-output-home).
For an offline archive or individual dataset, inspect `pyrite tables fetch --help`.

`pyrite config setup` is optional: it detects hardware and saves a backend choice.
Use the [development setup](../repo-design/development-workspace.md#install-from-source)
only when contributing or running from a checkout. For persistent tab completion,
see [shell completion](shell-completion.md).

## Run the quickstart profile

Choose a working directory where you want results, then inspect and run the
bundled one-case HOPG profile:

```bash
pyrite profile show quickstart
pyrite run quickstart
```

The demo uses a 30 keV beam and a 2 µm HOPG target at 45° tilt. It scores
incoherent PXR/CBS lines plus bremsstrahlung with a scalar detector. Allow roughly
a minute on CPU, with extra time for first-use compilation. A case-local line
grid is resolved automatically; no manual energy-grid derivation is needed.

Full `standard` sweeps cover much larger grids. Use the
[cluster guide](running-on-a-cluster.md) for heavy GPU or SLURM work.

## Locate and inspect the result

```bash
pyrite checkpoint list
```

The inventory lists active checkpoint datasets. Find the HOPG dataset produced
by the run; its component files live beneath
`pyrite-output/checkpoints/<dataset>/` in the effective workspace. Modified
profiles can have identity-qualified directory names. Repeating the same run
resumes compatible stored cases rather than discarding them.

To plot the saved spectra, install the optional apps extra in the same tool
environment, then launch the analysis app:

```bash
uv tool install --upgrade "pyrite-mc[apps]"
pyrite app analysis launch hopg
```

Select the quickstart dataset in the app's **Checkpoint** picker. If it opens on
**choose a checkpoint**, pick the dataset explicitly. The **Spectra** view shows
line and continuum components; it reads stored output without rerunning transport.
See [Working with results](working-with-results.md) for component meanings,
normalization, exports, and cleanup.

## Choose the next workflow

- [How PyRITE works](how-pyrite-works.md): follow the simulation stages.
- [Sweep profiles](sweep-profiles.md) and [configuration recipes](configuration-cookbook.md): change beam, material, detector, and sampling settings.
- [Python API workflow](python-api-workflow.md): simulate one scene in memory.
- [Run reference](../repo-design/cli/cli-reference-run.md) and [checkpoint reference](../repo-design/cli/cli-reference-checkpoint.md): complete options and output contracts.
- [Capabilities and limitations](../physics/capabilities.md) and [validation status](../validation/status-summary.md): check the evidence before scientific use.

A completed simulation establishes that the workflow ran; it does not establish
accuracy for a material, energy, or observable outside the documented evidence.
