# TODO — Zhai supplementary coherent-emission figures

Extend the Zhai validation dashboard with the requested 200 keV supplementary
coherent-emission-only studies: WSe₂ (42, 55, 75 nm), MoSe₂ (47, 112, 147 nm),
and h-BN (921 nm), all at polar tilts −10°, −15°, −17.5°, and −20°. The
implementation belongs in `checks/anchor_figures.py` with the marimo dashboard
as a thin cached driver: TMDs render 2×2 tilt panels over 800–1200 eV, while
h-BN overlays the four tilts over 600–1200 eV.
