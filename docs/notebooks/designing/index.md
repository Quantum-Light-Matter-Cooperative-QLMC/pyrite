# Designing simulations

Start by separating beam phase space, finite target geometry and crystal lattice
orientation. The first notebook combines these choices in one small CPU example. The next
shows how the emitted PXR/CBS spectrum depends on observation direction.
Then turn emitted photons into expected and acquired pixel counts.
Compare filtered and unfiltered spectra with a realistic detector response to
assess signal against background.
Next study history-count convergence and adaptive stopping, then map a
two-parameter response surface. Finally, learn to persist and resume
longer parameter studies using the same sweep concepts.

New to the API? Begin with [Fundamentals](../fundamentals/index.md).
For task recipes beyond these examples, use [How-to Guides](../../guides/index.md);
for exact inputs and result contracts, use the [Python API reference](../../reference/python-api/index.md).

```{toctree}
:maxdepth: 1

04-beams-crystals-target-geometry
05-angular-radiation-distributions
06-detector-counts
15-pxr-signal-background
07-sampling-convergence-performance
08-parameter-sweeps
18-persisted-parameter-studies
```
