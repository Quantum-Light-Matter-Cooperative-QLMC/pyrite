# Getting started

This guide takes a new user from installation to a small local run. The [project README](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite#readme) owns the complete installation matrix, scientific scope, and current validation caveats.

## Install and inspect the CLI

PyRITE requires Python 3.13 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite.git
cd pyrite
uv sync
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pyrite --help
```

Every `pyrite ...` command in these guides assumes `pyrite` is on your `PATH`: either activate the project environment as above in each new shell, or install it once as a tool (`uv tool install .`, see the [shell-completion guide](shell-completion.md)).

The base installation is CPU-only. Accelerator extras and contributor setup are documented in the [project README](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite#install) and [development workspace guide](../repo-design/development-workspace.md). To expose `pyrite` as a persistent user command and enable tab-completion without activating `.venv`, follow the [shell-completion guide](shell-completion.md).

## Discover profiles and materials

Profiles define reusable campaign settings and material membership. Inspect the available configuration before starting compute:

```bash
pyrite profile list
pyrite profile show standard
pyrite material show hopg
pyrite config list
```

Use the generated [CLI reference](../repo-design/cli/cli-reference.md) when a command's complete option and output contract matters. The [sweep-profile guide](sweep-profiles.md) explains profiles, fidelity, dataset identity, and named beams.

## Run a small smoke test

Start with one material on the tiny `--quick` grid rather than a production sweep:

```bash
pyrite run standard -m hopg --quick
```

Bundled profiles resolve a conservative case-local line grid automatically; no derivation step is required. Full sweeps are heavy; use the [cluster guide](running-on-a-cluster.md) for GPU or SLURM work.

## Inspect the result

Runs write identity-qualified component checkpoints beneath `pyrite-output/checkpoints/`. Continue with [Working with results](working-with-results.md) for checkpoint, analysis-app, export, and cleanup workflows.

Before using output for scientific claims, inspect the [validation ledger](../validation/physics-validation-ledger.md). A successful run does not imply that every model or instrument input is signed off.
