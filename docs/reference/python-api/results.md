# Results and analysis

Inspect computed spectra, provenance, acquired counts and analysis reductions. Historical root aliases remain supported while canonical ownership moves into `pyrite.results`.

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.PixelMetadata
   pyrite.PixelRayMap
   pyrite.SpectralFactors
   pyrite.SpatialResult
   pyrite.Analysis
   pyrite.Result
   pyrite.Provenance
   pyrite.CxrSpectra
   pyrite.SpectralDensity
   pyrite.results.Settings
   pyrite.results.records
   pyrite.results.filter_results
   pyrite.results.select_results
   pyrite.results.best_azimuth
   pyrite.results.cxr_metrics
   pyrite.results.radiation.radiation_band_metrics
   pyrite.results.summary_table
   pyrite.results.directional.DirectionalSpectrum
```

## Result analysis

Single simulations return `Result`; resumable campaigns store result mappings
in [component checkpoints](../../repo-design/storage/dataset-identity-and-storage.md).
`Settings` is the compatibility surface for checkpoint analysis. `Analysis`
holds presentation controls and legacy conversions; `simulate` does not
consume it.
