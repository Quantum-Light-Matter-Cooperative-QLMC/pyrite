# xray-refractive-index

## Independent derivation

**Cited source**: Maxwell dispersion relation in a homogeneous, linear, isotropic dielectric, `k² = (1 + χ₀) ω²`. **Claim**: complex crystal refractive index `n(E) = √(1 + χ₀(E)) ≈ 1 − δ − iβ`. **Signature**: `materials/crystal.py::refractive_index(crystal, photon_E_eV, use_henke=True)`.

### From Maxwell's equations to the dispersion relation

In a source-free, linear, homogeneous, isotropic medium with polarization $\mathbf P=\varepsilon_0\chi\mathbf E$, Maxwell's equations give the wave equation

$$
\nabla\times(\nabla\times\mathbf E)=-\mu_0\varepsilon_0(1+\chi)\,
\partial_t^2\mathbf E .
$$

A plane-wave ansatz $\mathbf E\propto\exp[i(\mathbf k\cdot\mathbf r-\omega t)]$ (or, with the sign convention adopted below, $\exp[i(\omega t-\mathbf k\cdot\mathbf r)]$ — the dispersion relation itself does not depend on which sign is chosen for the spatial/temporal cross term) turns this into

$$
k^2=(1+\chi)\,\frac{\omega^2}{c^2}.
$$

In the natural units used throughout this module ($c=1$, lengths in Angstrom, $\omega$ in Å⁻¹, i.e. `E/ħc`), and specializing to the forward-scattering ($g=0$) susceptibility $\chi_0$ of the crystal,

$$
k^2=(1+\chi_0)\,\omega^2 .
$$

### Definition of the refractive index

By definition, the phase velocity in the medium is $\omega/k$, and the refractive index is $n\equiv c\,k/\omega = k/\omega$ (with $c=1$). Taking the (principal-branch) square root of the dispersion relation,

$$
\boxed{\,n(\omega)=\sqrt{1+\chi_0(\omega)}\,}.
$$

This is exact — no smallness assumption has been used yet. Everything past this point is Taylor expansion, not new physics.

### Linearization and the δ, β convention

Write $\chi_0=\chi_0'+i\chi_0''$. Because $|\chi_0|\ll1$ in the X-ray regime, expand the square root about 1:

$$
n=\sqrt{1+\chi_0}=1+\tfrac12\chi_0-\tfrac18\chi_0^2+\tfrac{1}{16}\chi_0^3-\cdots
$$

Define $\delta,\beta$ by $n\equiv1-\delta-i\beta$ (the sign convention is fixed below). To first order,

$$
1-\delta-i\beta \;\approx\; 1+\tfrac12(\chi_0'+i\chi_0'')
\quad\Longrightarrow\quad
\delta=-\tfrac12\chi_0',\qquad \beta=-\tfrac12\chi_0''.
$$

For a passive medium (`xray-chi-zero`: $\chi_0'<0$, $\chi_0''<0$ off resonance, since $\chi_0=-\frac{r_e\lambda^2}{\pi V}\sum_i(f_{1,i}+if_{2,i})$ with $f_1\approx Z>0$, $f_2>0$), this gives $\delta>0$, $\beta>0$, matching the textbook X-ray convention (Als-Nielsen & McMorrow; Attwood & Sakdinawat) and the sibling claim `grazing-optical-constants`.

**Exact remainder.** Substituting $\chi_0=-2\delta_0-2i\beta_0$ (using the first-order δ₀, β₀ as the expansion parameter) into the $-\chi_0^2/8$ term:

$$
\chi_0^2=4\delta_0^2-4\beta_0^2+8i\delta_0\beta_0,
\qquad
-\tfrac18\chi_0^2=-\tfrac12\delta_0^2+\tfrac12\beta_0^2-i\delta_0\beta_0 .
$$

So, to next order,

$$
n=\bigl(1-\delta_0-\tfrac12\delta_0^2+\tfrac12\beta_0^2\bigr)
-i\bigl(\beta_0+\delta_0\beta_0\bigr)+O(\chi_0^3).
$$

For an off-edge medium ($\delta_0\gg\beta_0$), the fractional correction to the real part is

$$
\frac{\Delta\mathrm{Re}(n)}{\delta_0}\approx\frac{\delta_0}{2},
$$

and to the imaginary part,

$$
\frac{\Delta\mathrm{Im}(n)}{\beta_0}\approx\delta_0 .
$$

These are the two closed-form residuals the ledger cites ("measured residuals δ/2 and δ").

### Sign / time-factor convention check

Adopt the module-wide time factor $\exp(+i\omega t)$ with propagation phase $\exp\{i[\omega t-\mathbf k\cdot\mathbf r]\}$ (stated in the ledger row and matched by the implementation docstring). With $n=1-\delta-i\beta$ ($\delta,\beta>0$) and $\mathbf k=n\omega\hat{\mathbf n}$,

$$
k=(1-\delta)\omega-i\beta\omega .
$$

Writing $z=\hat{\mathbf n}\cdot\mathbf r$,

$$
-ikz=-i\bigl[(1-\delta)\omega-i\beta\omega\bigr]z
=-\beta\omega z-i(1-\delta)\omega z,
$$

so

$$
\exp\{i[\omega t-kz]\}=\exp(i\omega t)\,\exp(-i(1-\delta)\omega z)\,
\exp(-\beta\omega z).
$$

The amplitude decays as $\exp(-\beta\omega z)$ with increasing propagation distance $z$ (β>0), which is the physically required behavior for a passive (absorbing) medium under this sign convention. This is self-consistent with `absorption-length`'s independently derived $\mu=2\beta k\approx2\beta\omega$
(intensity ∝ |amplitude|² ⇒ $\exp(-2\beta\omega z)=\exp(-\mu z)$), so the
δ, β sign convention used here is the same one that produces a positive, physical absorption coefficient elsewhere in the module — internally consistent, not merely asserted.

### Summary of the independent result

$$
n(E)=\sqrt{1+\chi_0(E)}\ \ \text{(exact, principal branch)},\qquad
n(E)\approx1-\delta(E)-i\beta(E)\ \ \text{(linearized, error }O(\chi_0^2)\text{)},
$$
with $\delta=-\chi_0'/2$, $\beta=-\chi_0''/2$ at leading order, residuals $\sim\delta/2$ (real part, fractional) and $\sim\delta$ (imaginary part, fractional), and $n\to1$ as $\chi_0\to0$ (vacuum / far off any edge).

## Units, conventions, and limiting cases

- **Units**: $\chi_0$ is dimensionless (`xray-chi-zero`: $r_e\lambda^2/V$ is dimensionless), so $n$ is dimensionless. Pass.
- **Sign**: off resonance, $\delta>0$, $\beta>0$, $\delta\gg\beta$ (since $f_1\sim Z\gg f_2$ away from an edge). Pass.
- **Time-factor / propagation-phase convention**: $\exp(+i\omega t)$ with $n=1-\delta-i\beta$ gives amplitude decay $\exp(-\beta\omega z)$ along the propagation direction — physical for a passive medium, and consistent with the independently derived `absorption-length` coefficient $\mu=2\beta\omega$ (in these natural units). Pass.
- **Limiting case, χ₀→0**: $n\to\sqrt{1}=1$ exactly (vacuum / asymptotically high energy, no expansion needed — this limit holds for the exact square root, not just the linearization). Pass.
- **Limiting case, exact vs. linearized**: the two forms agree to $O(\chi_0^2)$; the leading fractional residuals are $\delta/2$ (real part) and $\delta$ (imaginary part), derived above from a plain Taylor expansion of the square root — no implementation input was used to obtain these coefficients.

## Implementation comparison

`src/pyrite/materials/crystal.py::refractive_index`:

```python
def refractive_index(crystal, photon_E_eV, use_henke=True):
    return np.sqrt(1.0 + chi_0(crystal, photon_E_eV, use_henke))
```

This is `n = √(1+χ₀)` literally, with `chi_0` supplying the already-validated (`xray-chi-zero`, `filtered`) forward susceptibility. `np.sqrt` on a complex array takes the principal branch (`Re(√z) ≥ 0`), and for `z = 1+χ₀` near 1 with small negative real and imaginary parts (passive medium, off edge) the principal branch places the result near `1` with `Im(n) < 0`, i.e. `n ≈ 1 − δ − iβ` with `δ, β > 0` — exactly the branch selection the sign analysis above requires. No branch ambiguity issue.

The docstring states the same time-factor/propagation-phase convention used in the derivation (`exp(+i ω t)`, `exp{i[ω t − k·r]}`) and explicitly notes the square root is kept exact — matching the "exact, principal branch" result above rather than the linearization.

### Sibling-function consistency (χ₀, δ/β from `optical_constants`)

`chi_0` (already `filtered`) is built from `f1 = Z + f'`, `f2 = f''` in the Henke convention with the *same* `r_eλ²/(2πV)` normalization used by `optical_constants`'s δ, β and by `absorption_length_ang`'s β — by construction, `χ₀ = −2(δ_lin + iβ_lin)` exactly, not merely to leading order. This means the "linearized" comparison in the ledger checks is not an independent formula colliding with the exact one by coincidence; it is the same microscopic `f1, f2` input fed through two different final operations (a first-order Taylor step vs. an exact square root), so a residual of `O(χ₀²)` is exactly what the derivation above predicts, and nothing else.

### Downstream-consumer consistency (the claim's stated open item)

Grepping all call sites of `refractive_index` in `src/pyrite`:

```
src/pyrite/montecarlo/spectrum/lines.py:913:  refractive_index(crystal, E_tab, use_henke).real            # -> n_re_tab_g, used as Re n(E) in k = Re n(ω) ω n̂ (xray-in-medium-resonance)
src/pyrite/montecarlo/spectrum/lines.py:971:  1.0 - refractive_index(crystal, E_grid_eV, use_henke).real   # -> δ(E), used in delta_omega_grid = δ(E)ω(E) (xray-in-medium-propagation-phase)
```

Both consumers call the *same* exact-square-root function and take `.real`; the propagation-phase δ is computed as `1 − Re(n_exact)`, not by a separate call to the linearized `optical_constants`. The CUDA kernels (`coherent_stream_jit_kernel.py`) receive `n_re_tab` and `delta_omega` as precomputed arrays from these same two call sites — they do not recompute a refractive index on-device with a different (e.g. linearized) formula, so the GPU path cannot diverge from the CPU path on this choice either.

The only other production call site of the *linearized* formula, `optical_constants`, is `detectors/grating.py:186` (grazing-incidence reflectivity, `Grating.reflectivity`), a Fresnel-reflectivity calculation outside the crystal-diffraction/CXR line-kinematics code path entirely — it is not competing with or substituting for `refractive_index` anywhere. So within the code path this claim gates (`xray_dispersion="refractive"` line kinematics and coherent propagation phase), the exact square root is used exclusively and consistently; the exact-vs-linearized choice does not create an internal inconsistency. This resolves the claim's own stated open item at the "which form is used" level (the separate, still-open question of how much the δ-order phase *accumulates* over a trajectory is the domain of `xray-in-medium-propagation-phase`, not this claim).

## Numerical spot check

Evaluated the installed implementation directly (not a `tests/` helper) for silicon at 1.5 keV, an off-edge point cited in the ledger row:

```
chi_0(Si, 1500 eV)      = -3.6795352231e-04 - 1.4761822587e-05j
n = refractive_index(...) = 9.9981600634e-01 - 7.3822695844e-06j
delta_exact = 1 - Re(n)  = 1.8399366074e-04
beta_exact  = -Im(n)     = 7.3822695844e-06

delta_lin = -Re(chi_0)/2 = 1.8397676116e-04
beta_lin  = -Im(chi_0)/2 = 7.3809112936e-06

(delta_exact - delta_lin)/delta_lin = 9.1857e-05   predicted delta_lin/2 = 9.1988e-05
(beta_exact  - beta_lin )/beta_lin  = 1.8403e-04   predicted delta_lin   = 1.8398e-04
```

Both measured residuals match the closed-form predictions ($\Delta\mathrm{Re}(n)/\delta\approx\delta/2$, $\Delta\mathrm{Im}(n)/\beta\approx\delta$) derived independently above to better than 1%, and reproduce the ledger's cited figures ("9.2e-5 and 1.8e-4 rel at 1.5 keV in Si") to 4 significant figures. `n → 1` was also confirmed in the `χ₀ → 0` high-energy limit (already anchored by `tests/materials/test_crystallography.py::test_chi_0_vanishes_and_index_tends_to_vacuum_at_high_energy`, which checks `abs(refractive_index("silicon", 20000.0) - 1) < 1e-5`).

## Adjudication

**match**

`n = √(1+χ₀)` is exactly the Maxwell dispersion relation's square root, taken on the correct (principal) branch, with no unexplained factor. The linearization `1 − δ − iβ` and its `O(χ₀²)` residual (`δ/2` real, `δ` imaginary) follow from a plain Taylor expansion and reproduce the implementation's own claimed figures to 4 significant figures. The `exp(+iωt)`/`exp{i[ωt−k·r]}` sign convention gives physical (decaying) absorption, consistent with the independently derived `absorption-length` coefficient. Both production call sites of `refractive_index` (the in-medium resonance kinematics and the coherent propagation phase in `lines.py`, and the CUDA kernels fed from them) exclusively use the exact square root; the separately-computed linearized `optical_constants` is used only in the unrelated grazing-incidence `Grating.reflectivity` path. No inconsistency between exact and linearized forms was found in the code paths this claim covers.

Suggested ledger disposition: `rederived`.

## Addendum 2026-08-19: the vacuum-dispersion switch was removed

`xray_dispersion` no longer exists, so the code path named above as `xray_dispersion="refractive"` is simply *the* line-kinematics and coherent propagation path. Nothing in this claim changes: `refractive_index` and its exact square root are untouched, and the consumers named here are the same ones, now reached unconditionally.
