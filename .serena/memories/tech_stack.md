# Technology stack

- Python >=3.13; uv project; Hatchling build; `src/` layout.
- Click CLI; NumPy/SciPy, xraydb, crystals; marimo plus Matplotlib/Altair/Plotly/pandas.
- pytest with xdist/timeout/cov; Ruff lint/format; ty type checking; Sphinx/MyST/Furo docs.
- Accelerator extras are mutually exclusive: NVIDIA CuPy, AMD CuPy/ROCm, or Intel dpnp/dpctl.
- Serena owns symbol navigation. Headroom shapes agent/tool output; it is not a shell wrapper or code index. Tokensave and RTK are retired.
