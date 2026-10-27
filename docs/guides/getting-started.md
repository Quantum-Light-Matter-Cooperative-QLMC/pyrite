# Getting started

Install PyRITE, run one CPU simulation, and open its saved spectrum.
You need Python **3.13 or newer** and [uv](https://docs.astral.sh/uv/).
For a library install or GPU dependencies, use the
[README installation matrix](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite#install).

## Install and fetch reference data

Install the CLI with its analysis app:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
uv tool install "pyrite-mc[apps]"
uv tool update-shell
```

Open a new shell so `pyrite` is on `PATH`, then choose a workspace and fetch
PyRITE's pinned cross-section and atomic reference data:

```bash
mkdir -p pyrite-demo
cd pyrite-demo
export PYRITE_HOME="$PWD"
pyrite --version
pyrite tables fetch
```

Keep this shell for the remaining commands. The fetch verifies the downloads
and installs them in the selected workspace. For offline archives or individual
datasets, see the [tables reference](../repo-design/cli/cli-reference-tables.md).

## Run the quickstart profile

```bash
pyrite profile show quickstart
PYRITE_MC_BACKEND=cpu pyrite run quickstart
```

This runs **30 keV electrons through 2 µm (20,000 Å) of HOPG at 45° tilt**,
with a scalar detector at **90°** to the beam. The profile scores incoherent
PXR/CBS, characteristic radiation and bremsstrahlung. It resolves a line-energy
grid automatically and uses adaptive sampling. Allow roughly a minute on CPU,
plus first-use compilation; the captured run below took 36 seconds.

## Locate and inspect the result

```bash
pyrite checkpoint list
```

Actual inventory from a completed CPU quickstart:

```{literalinclude} ../examples/data/getting_started.txt
:language: text
```

One record means the single profile case was saved. Its component files,
including `line.h5`, `brem.h5` and `characteristic.h5`, are beneath
`pyrite-output/checkpoints/<dataset>/` in your workspace. The dataset's identity
suffix can vary with resolved inputs and software. Repeating the run resumes
compatible saved cases. The {download}`captured output and run metadata <../examples/data/getting_started.json>`
and {download}`capture script <../examples/getting_started.py>` make this example
reviewable; the script reuses installed tables and isolates checkpoint writes.

Open the saved spectra:

```bash
pyrite app analysis launch hopg
```

Select the **quickstart** dataset in the app's **Checkpoint** picker, then use
**Spectra** to inspect the line and continuum components. The app reads saved
output without rerunning transport. This is simulated emission, not measured
data; display scaling and detector response affect the plotted observable.
[Working with results](working-with-results.md) explains the plots, units and
normalization. For an inline source-spectrum figure and a runnable in-memory
simulation, follow [Your first Python simulation](python-api-workflow.md).

(install-completion-optional)=
(backend-setup-optional)=

## Choose the next workflow

- [How PyRITE works](../physics/how-pyrite-works.md): simulation stages and observables.
- [Configure a campaign](configuration-cookbook.md): change inputs or the output home.
- [Completion](../repo-design/cli/cli-reference-config.md#pyrite-config-completion-install) and [backend setup](../repo-design/cli/cli-reference-config.md#pyrite-config-setup): optional machine setup. NVIDIA CUDA is supported; Intel and ROCm remain untested.
- [Running on a cluster](running-on-a-cluster.md): larger sweeps and GPU jobs.

A successful run demonstrates the workflow. Check the
[capabilities and limitations](../physics/capabilities.md) and
[validation status](../validation/status-summary.md) before quantitative use.
