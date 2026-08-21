# Developer physics checks

Standalone, potentially slow validation anchors. These are not application
entry points and are not part of the fast unit-test suites. The validation
ledger remains authoritative; a successful check is evidence, not human
sign-off.

| Artifact | Validation ledger id(s) | Purpose |
|---|---|---|
| `dans_diffraction_oracle.py` | `dans-diffraction-oracle` | Pinned external lattice, reciprocal-geometry, and structure-factor comparison. |
| `detector_solid_angle_check.py` | `detector-eaglexo`, `detector-line-broadening` | Solid-angle integration and analytic aperture-width comparison; the integrated-spectrum route is not separately ledgered. |
| `energy_loss_straggling_observables.py` | `energy-loss-straggling` | Paired-seed phase, terminal-fraction, stopped-range, bremsstrahlung, and coherent-line response to Urban straggling. |
| `feranchuk_check_script.py` | `closed-form-flux`, `pxr-amplitude`, `cbs-amplitude` | Legacy Feranchuk–Spence LiF absolute-flux anchor. |
| `feranchuk_vs_zhai_check.py` | `closed-form-flux`, `coherent-line-spectrum` | Analytic-versus-transport comparison through the maintained Zhai anchor pipeline. |
| `kinematic_validity_check.py` | `coherent-line-spectrum`, `line-energy-dispersion` | Kinematic-approximation audit; its validity parameters are supporting diagnostics, not separate ledger claims. |
| `mosaic_mc_check.py` | `mosaic-analytic`, `mosaic-mc` | Perfect-crystal limit, quadrature convergence, broadening, and yield checks. |
| `multilayer_check.py` | `multilayer-stack`, `self-absorption` | First-slice stack transport and front/back escape checks. |
| `multilayer_slice3_check.py` | `multilayer-stack`, `self-absorption` | Per-layer crystalline-radiation and incoherent-sum checks. |
| `multilayer_validation_check.py` | `multilayer-stack`, `self-absorption` | Closed-form stack attenuation and depth-range scaling anchors. |
| `cxr_analysis_feranchuk.ipynb`, `cxr_analysis_feranchuk.md` | `coherent-line-spectrum`, `closed-form-flux` | Legacy paired notebook retained for historical Feranchuk analysis; keep output-free. |

Run commands are documented in each check. Heavy Monte Carlo or GPU work must
use `cxr remote`; do not launch it locally from this directory.
