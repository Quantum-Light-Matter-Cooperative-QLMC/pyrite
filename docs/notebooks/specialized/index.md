# Specialized workflows

Import a correlated electron distribution, compare transition radiation
from thin films, optimize a source, tune CUDA execution on your GPU, and
explore classical radiation at MeV beam energies. Each notebook can be run
independently after the introductory
[Fundamentals](../fundamentals/index.md).

```{toctree}
:maxdepth: 1

09-importing-electron-beams
10-transition-radiation
11-optimizing-xray-source
15-tuning-cuda-execution
16-classical-mev-radiation
```

## Classical coherent phase tracking (source preview)

The {download}`coherent phase-tracking Marimo source <17-classical-coherent-phase-tracking.py>`
uses the experimental public microtrain and temporal-profile interfaces.
This is **source only** until CPU timing, a reproducible execution capture, and
strict docs validation are completed. It is deliberately not yet in the
published tutorial toctree (issue #777).
