# Developer physics checks

Standalone, potentially slow validation anchors. These are not application
entry points and are not part of the fast unit-test suites. The validation
ledger remains authoritative; a successful check is evidence, not human
sign-off.

The canonical check-to-ledger mapping is `pyrite.validation.check_records.CHECK_LEDGER_IDS`.
Use the collector to preserve normal output and write dated JSONL evidence:

```bash
uv run pyrite-dev validation-records -- checks/dans_diffraction_oracle.py
```

On remote workers, use the collector's `--output-dir`; copy reviewed records into
`docs/validation/check-records/` before regenerating the status summary. The collector
marks exit code 0 as pass, 2 as skip, and any other nonzero exit as fail.

| Artifact | Purpose |
|---|---|---|
| `dans_diffraction_oracle.py` | Pinned external lattice, reciprocal-geometry, and structure-factor comparison. |
| `detector_solid_angle_check.py` | Solid-angle integration and analytic aperture-width comparison; the integrated-spectrum route is not separately ledgered. |
| `energy_loss_straggling_observables.py` | Paired-seed phase, terminal-fraction, stopped-range, bremsstrahlung, and coherent-line response to Urban straggling. |
| `feranchuk_check_script.py` | Legacy Feranchuk–Spence LiF absolute-flux anchor. |
| `feranchuk_vs_zhai_check.py` | Analytic-versus-transport comparison through the maintained Zhai anchor pipeline. |
| `kinematic_validity_check.py` | Kinematic-approximation audit; its validity parameters are supporting diagnostics, not separate ledger claims. |
| `mosaic_mc_check.py` | Perfect-crystal limit, quadrature convergence, broadening, and yield checks. |
| `multilayer_check.py` | First-slice stack transport and front/back escape checks. |
| `multilayer_slice3_check.py` | Per-layer crystalline-radiation and incoherent-sum checks. |
| `multilayer_validation_check.py` | Closed-form stack attenuation and depth-range scaling anchors. |

Run commands are documented in each check. Heavy Monte Carlo or GPU work must
use `pyrite remote`; do not launch it locally from this directory.
