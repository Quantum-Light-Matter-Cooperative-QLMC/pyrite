# Interactive Tutorials

Learn PyRITE through short, executable scientific notebooks. Start with
Fundamentals: verify your environment, simulate a spectrum, interpret its
results and predict line energies. Each notebook stands alone and uses the
public Python API.

The docs build executes each notebook and embeds its **static output**, including
code and figures from the same
canonical `.py` file used by Molab and local marimo. Static pages need no Python
kernel or GPU. Recompute and edit in Molab or a local marimo session; static
controls cannot rerun simulations. The HTML viewer loads marimo frontend assets
from a CDN; the canonical source downloads remain available if it cannot load.
Generated HTML is a build artifact and is not committed to the repository.

```{toctree}
:maxdepth: 2

Fundamentals <fundamentals/index>
Designing simulations <designing/index>
```

For specific tasks, use the [guides](../guides/index.md); for exact interfaces,
consult [Python API](../reference/python-api/index.md). The [physics](../physics/index.md) and
[validation](../validation/index.md) sections explain assumptions and evidence.
