# Crystal mosaicity

Real crystals are **mosaic**: an incoherent ensemble of small, slightly misoriented perfect crystallites whose orientations follow a Gaussian distribution about the mean. The width is the **mosaic spread** η, quoted as a rocking-curve FWHM (e.g. HOPG grades ZYA 0.4° / ZYB 0.8° / ZYH 3.5°; good bulk TMDs ~0.2–0.5°; CVD/exfoliated flakes up to a few degrees). Because the PXR/Bragg coherence angular width is far smaller than η, the crystallites radiate **incoherently**, and the observable is the single-crystal spectral-angular distribution **averaged over the orientation distribution** (i.e. over the spread of the reciprocal vector **g**).

This is the standard kinematical mosaic-crystal treatment (Darwin mosaic-block model; in PXR specifically: Nasonov; Feranchuk–Ivashin; and the 2026 *Rad. Phys. Chem.* paper "The Effect of Crystalline Mosaicity on the Spectral-Angular Distribution of Parametric X-ray Radiation").

Two implementations are available: **analytic detector broadening** and an explicit **Monte-Carlo mosaic average.**

---

## Analytic broadening

### What it does

A mosaic tilt rotates **g**, and only the numerator `v·g` of the resonance `E_res = ħc·(v·g)/(1 − v·n̂)` depends on g. To first order a tilt δ shifts the line by `dE/E = −tan(ψ)·δ`, ψ = ∠(v, g). A Gaussian mosaic of rocking-curve FWHM η therefore broadens the line by a Gaussian of energy width

```{math}
\mathrm{FWHM}_\mathrm{mosaic} = E \cdot |\tan \psi| \cdot \eta
```

added **in quadrature** with the energy-dispersive spectrometer (EDS) resolution and detector-aperture widths and applied through the same Gaussian convolution path. The **unbroadened source** spectrum is untouched — mosaicity enters only the detector-convolution FWHM, so a record computed with `mosaic=False` can be re-broadened at plot time.

### Where it lives

- `montecarlo.mosaic_fwhm_eV(E, ψ, η_rad)` — the formula above.
- `montecarlo.mosaic_psi_rad(case, E_pk)` — $\psi$ for the reflection whose nominal (unscattered-beam) resonance is nearest the peak; uses the shared `montecarlo._orientation_R` (extracted from `mc_spectrum`).
- `results.store_result` — adds the term in quadrature, gated on `case["mosaic_fwhm_rad"]`, capped at `E_pk`.
- `data/catalog/crystals/<key>.toml` — optional per-crystal `mosaic_fwhm_deg`; the immutable `materials.CATALOG` and `materials.crystal.load_crystals` surface it.
- `Scene(target=Slab(…, mosaic=True), …)` → public case lowering → `case["mosaic_fwhm_rad"]`. `mosaic` is target state; the width comes from the catalog crystal. Route and quadrature controls live under `Numerics.convergence`.
- `plots.plot_mosaic_comparison` — overlay grades from one record.
- Tests: `tests/montecarlo/test_mosaic.py`.

### Model Limitations

- **Energy-shift only.** It holds the amplitudes (polarization projection `g·e`, the PXR detuning `|k+g|²−ω²`, the CBS braced products) fixed across the mosaic cone — these all depend on g *direction*. Good while the line is narrow; approximate once it is broad.
- **Diverges at grazing.** `tan ψ → ∞` as $\psi$ → 90° (g grazing v), so the term is capped at `E_pk` and is unreliable at steep tilts near grazing.
- **Often sub-dominant.** The electron multiple-scattering Doppler spread already broadens the line in thick/bulk crystals, so mosaic broadening is mostly visible in thin / near-perfect samples and for the high-mosaic ZYH grade.
- **Single representative ψ.** It uses the nominal beam direction (like `aperture_fwhm_eV` uses nominal θ_obs), not the per-segment scattered velocities.
- **No comparison to measurement.** The broadening has not been compared with measured HOPG rocking-curve or line widths.

---

## Monte-Carlo mosaic average

### What it does

`mc_spectrum(..., mosaic_fwhm_rad=<rad>, mosaic_nodes=<n>)` replaces the post-hoc Gaussian with a true incoherent average **inside** the spectrum: it sums the per-reflection block over a set of crystallite orientations drawn from the mosaic distribution, recomputing the polarization pair `e_s/e_p`, `E_res`, the amplitudes `A_PXR/A_CBS` and the sinc lineshape at each orientation, weighted by the mosaic PDF. The structure-factor tabulations (`chi_g`/`U_g`) depend on hkl + energy, **not** orientation, so they are computed once per reflection and reused; the self-absorption geometry is mosaic-independent (only its `μ(E_res)` is re-evaluated, cheaply).

`mosaic_fwhm_rad=None` **or** `mosaic_nodes ≤ 1` is the **perfect-crystal fast path**, which returns the single-orientation result bit-for-bit.

### Orientation quadrature

The mosaic average is a 2-D integral of a *smooth* integrand (spectrum vs crystallite tilt) against a Gaussian weight — textbook **Gauss-Hermite**. The implementation (`montecarlo._mosaic_quadrature`) uses a 2-D product Gauss-Hermite rule over the tilt of the crystallite normal: per-axis σ = η_FWHM / 2.3548 (the rocking curve is the 1-D projection), `mosaic_nodes` nodes per axis, **K = mosaic_nodes² orientations**, weights summing to 1. Each node's tilt rotates **g** by a Rodrigues rotation (`_small_tilt_R`) at the existing `_orientation_R` hook.

Deterministic quadrature is used instead of randomly drawn orientations: it **converges in far fewer evaluations** for a smooth integrand and needs **no RNG sub-stream** (so it never couples to the transport `seed`). The single-node (K=1) rule sits exactly at zero tilt, which makes the perfect-crystal path bit-for-bit.

### Convergence: moments vs lineshape

The **moments converge fast.** Integrated yield, mean energy and the second-moment width are converged by `mosaic_nodes ~ 5` (nodes 5 vs 9 agree to a few × 10⁻⁶ on the HOPG ZYH line integral). For a width-vs-rocking-curve comparison this is enough.

The **detailed lineshape converges slowly when the mosaic spread ≫ the unbroadened source-line width.** Each node contributes a shifted copy of the narrow base line; for HOPG (00l) the energy shift is essentially 1-D in the in-plane tilt, so a handful of nodes gives a handful of discrete copies and the summed line is *lumpy* until the node spacing in energy drops below the unbroadened core width. A smooth broad lineshape needs `mosaic_nodes` scaling with (mosaic width / unbroadened source width): HOPG ZYH (3.5°) on a thin film needs ~30–40 nodes/axis for a smooth core; ZYA/ZYB (0.4/0.8°) are smooth by ~9–13. Cost is K = `mosaic_nodes²` evaluations of the line hot loop, **serial under CuPy**, so the broad-lineshape regime is genuinely expensive. The `Convergence` default (`mosaic_nodes=5`) targets converged moments; raise it for a publication-quality broad lineshape.

### Where it lives

- `montecarlo._mosaic_quadrature(fwhm_rad, nodes)` / `_small_tilt_R(dx, dy)` — the quadrature and the tilt rotation.
- `montecarlo.mc_spectrum(..., mosaic_fwhm_rad, mosaic_nodes)` — the orientation loop.
- `montecarlo.run_case` / `_spectrum_case` — read `case["mosaic_mc_fwhm_rad"]` / `case["mosaic_mc_nodes"]`.
- `Scene(target=Slab(…, mosaic=True), …)` with `Numerics(convergence=Convergence(mosaic_route="mc", mosaic_nodes=…))` sets the `mosaic_mc_*` case keys **and turns the analyti `store_result` term off** — the two routes are mutually exclusive (applying both double-counts the broadening).
- Checks: `checks/mosaic_mc_check.py`; synthetic unit tests (quadrature + wiring): `tests/montecarlo/test_mosaic_mc.py`.

```python
import pyrite as pr

beam = pr.Beam(energy_keV=30.0)
target = pr.Slab("hopg", thickness_ang=10_000.0, mosaic=True)
detector = pr.Detector()
numerics = pr.Numerics(convergence=pr.Convergence(mosaic_route="mc", mosaic_nodes=35))
result = pr.simulate(beam, target, detector, numerics=numerics)
```

The supported high-level API uses the catalog's `mosaic_fwhm_deg`. Campaign compatibility helpers still permit explicit run-level width overrides, but they are not part of the supported scene API.

### Behavior (`checks/mosaic_mc_check.py`; HOPG (002), 30 keV, θ_obs 90°)

- K=1 / `fwhm=None` reproduce the perfect crystal bit-for-bit; η→0 converges to it.
- The line broadens monotonically with grade (core FWHM 10 → 12 → 17 → 53 eV for perfect / ZYA / ZYB / ZYH) and the peak drops.
- The added width matches the analytic `E·|tan ψ|·η` to within ~10% in the small-η limit, where the analytic route is valid.
- The integrated yield is nearly conserved (ZYH/perfect = 0.999) for this fixed-detector, whole-line observable; broadening, not a yield change, dominates. Yield changes reported in the literature arise in other geometries and observables (fixed narrow window, divergent beam). The energy-shift-only analytic route cannot reproduce any yield change.

### Relation to the detector solid-angle integral

Mosaic averaging shares the "incoherently sum over a distribution of a direction" structure with the [detector solid-angle integral](../detectors/detector-solid-angle.md): **g** for mosaic, n̂ for the aperture. Combining them multiplies the cost (K × N_dir).
