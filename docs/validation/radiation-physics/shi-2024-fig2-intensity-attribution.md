# Shi 2024 Fig. 2 intensity attribution (#1103)

Validation: `shi-2024-fig2-intensity-attribution`.

## Scope and independence

This is a fresh-context verification at `3445bd19a` (branch
`issue-1103-shi-pxr-intensity`, clean tree). The order of reading was:

1. The ledger row and the sources: Shi, Wong, Huang, Wong & Kaminer,
   Nat. Commun. 15, 7803 (2024), main-text Eqs. (1)–(3), (5)–(8), Methods and
   Fig. 2, and SI Eqs. (S1)–(S17).
2. §1–§4 below, derived and computed with the validator's own scripts
   (`scratch/validator/`, not committed). The scripts are an absolute
   per-reflection model normalized from Shi Eq. (1), plus a pixel read of a
   600 dpi render of Fig. 2 and independent Waasmaier–Kirfel, Kissel $f'$ and
   Debye–Waller inputs.
3. Only then: the research note
   `docs/research/physics/shi-2024-intensity-1103.md`, the checks
   `checks/shi_2024_matrix_elements_1103.py` and
   `checks/digitize_shi_2024_fig2_1103.py`, the test
   `tests/montecarlo/spectrum/test_shi_2024_fig2_intensity.py`, the `A_PXR` and
   `A_CBS` assembly in `montecarlo/spectrum/cxr/_per_hkl.py` and `_batched.py`,
   `materials/crystal.py::{chi_g,U_g}`, and the hook
   `montecarlo/spectrum/cxr/_recoil.py::_RECOIL_HBARC_EV_ANG`.

The claim is measurement-only. The validator edited nothing except this file.

## 1. Per-reflection elements

### Geometry and conventions

The electron moves along $\hat{\mathbf z}$ with velocity $\mathbf v=\beta c\,\hat{\mathbf z}$
through a slab of thickness $L=30$ nm, which it crosses in time $T=L/v$. The
photon direction is $\hat{\mathbf n}=(\sin\theta,0,\cos\theta)$ with
$\theta=100^\circ$. The polarizations are
$\mathbf e_1=\hat{\mathbf y}$ and $\mathbf e_2=(\cos\theta,0,-\sin\theta)$.

Shi's momentum balance is $\mathbf q_i+\mathbf g=\mathbf q_f+\mathbf k$ (main
Eq. (3), SI Eq. (S14)), with $\mathbf g=(n_x,n_y,n_z)\,2\pi/a$. Since the
electron loses longitudinal momentum, the classical root is

$$
\omega_c=\frac{-v g_z}{1-\beta\cos\theta},
$$

so $n_z<0$. For $n_z=-2$ this is 1418.4 / 3124.2 / 3876.0 eV at 30 keV /
300 keV / 5 MeV, and for $n_z=-3$ it is 2127.6 / 4686.3 / 5814.0 eV.

Let the crystal be $\chi(\mathbf r)=\sum_{\mathbf g}\chi_{\mathbf g}e^{i\mathbf g\cdot\mathbf r}$
and $\phi(\mathbf r)=\sum_{\mathbf g}\phi_{\mathbf g}e^{i\mathbf g\cdot\mathbf r}$.
The emission mode of SI Eq. (S9) is the complex conjugate of the absorption
mode, so for a real $\chi(\mathbf r)$ its $\mathbf g$ component carries
$\chi_{-\mathbf g}^{*}=\chi_{\mathbf g}$. That is the same $\mathbf g$ as the
Coulomb vertex $\phi_{\mathbf g}$ in SI Eq. (S13). Both share the structure
factor $S_{\mathbf g}=\sum_j e^{-i\mathbf g\cdot\mathbf R_j}$, so
$\chi_{\mathbf g}/\phi_{\mathbf g}$ is real. With SI Eq. (S3) and
$e<0$, the electron potential energy is

$$
U_{\mathbf g}\equiv e\phi_{\mathbf g}
=-\frac{4\pi e^2}{4\pi\varepsilon_0 V_{\rm cell}}\,\frac{Z-f(g)}{g^2}\,S_{\mathbf g}<0,
\qquad
\chi_{\mathbf g}=-\frac{r_e\lambda^2}{\pi V_{\rm cell}}\,f(g)\,S_{\mathbf g},
$$

where Shi's Salvat–Yukawa form factor is
$f(g)=Z\sum_k C_k\tilde\mu_k^2/(\tilde\mu_k^2+g^2)$ with
$\tilde\mu_k=\mu_k/a_0$. Note that $(Z-f)/g^2=Z\sum_k C_k/(\tilde\mu_k^2+g^2)$,
which is the Yukawa Fourier transform, as SI Eq. (S3) states.

### Classical amplitudes from Eq. (1)

Main-text Eq. (1) with the vacuum field $\mathbf e_\sigma e^{-i\mathbf k\cdot\mathbf r}$
reproduces Jackson's photon spectrum
$d^2N/d\omega d\Omega=(\alpha\omega/4\pi^2c^2)\,\lvert\int \mathbf v\cdot\mathbf e_\sigma e^{i(\omega t-\mathbf k\cdot\mathbf r)}dt\rvert^2$.
This fixes the absolute normalization. Write each reflection's amplitude as
$\int dt\,e^{i(\omega-(\mathbf k-\mathbf g)\cdot\mathbf v)t}\,a_{\mathbf g\sigma}$.

**PXR** (the field term of Eq. (S9), on a straight path). Here
$\mathbf K=\mathbf k-\mathbf g$ and $\mathbf K\cdot\mathbf e=-\mathbf g\cdot\mathbf e$:

$$
a^{\rm PXR}_{\mathbf g\sigma}
=\chi_{\mathbf g}\,
\frac{k^2(\mathbf v\cdot\mathbf e_\sigma)+(\mathbf v\cdot\mathbf K)(\mathbf g\cdot\mathbf e_\sigma)}{K^2-k^2}.
$$

**CB** (the $\delta\mathbf v$ term that Shi drops). The force from
$U_{\mathbf g}e^{i\mathbf g\cdot\mathbf r}$ along $\mathbf r=\mathbf vt$ gives
$\delta\mathbf p=-U_{\mathbf g}\mathbf g\,e^{i\mathbf g\cdot\mathbf vt}/(\mathbf g\cdot\mathbf v)$.
Then $\delta\mathbf v=\mathbf w\,\delta p/\hbar g$ with
$\mathbf w=\hbar(\mathbf g-\boldsymbol\beta(\boldsymbol\beta\cdot\mathbf g))/\gamma m$,
the relativistic inverse-mass tensor $\partial^2E/\partial\mathbf p^2$. Keep both
$\delta\mathbf v\cdot\mathbf e$ and the phase term
$-i(\mathbf k\cdot\delta\mathbf r)(\mathbf v\cdot\mathbf e)$:

$$
a^{\rm CB}_{\mathbf g\sigma}
=-\frac{U_{\mathbf g}}{\hbar\,\mathbf g\cdot\mathbf v}
\Bigl[\mathbf w\cdot\mathbf e_\sigma-(\mathbf v\cdot\mathbf e_\sigma)\frac{\mathbf k\cdot\mathbf w}{\mathbf g\cdot\mathbf v}\Bigr].
$$

**Quantum check of the same element.** SI Eq. (S15) has two orderings,
scatter-then-emit (intermediate $\mathbf q_1=\mathbf q_i+\mathbf g$) and
emit-then-scatter ($\mathbf q_2=\mathbf q_i-\mathbf k$). Its amplitude is
$U_{\mathbf g}[\mathbf j_{f1}\cdot\mathbf e/(E_i-E_1)+\mathbf j_{2i}\cdot\mathbf e/(E_f-E_2)]$
in the same velocity units. Expanding $E(\mathbf q)$ to second order gives

$$
\frac{1}{E_i-E_1}+\frac{1}{E_f-E_2}=\frac{\hbar^2\,\mathbf k\cdot M^{-1}\mathbf g}{(\hbar\mathbf v\cdot\mathbf g)^2},
\qquad
\mathbf j_{f1}-\mathbf j_{2i}=\hbar M^{-1}\mathbf g+O(\hbar^2).
$$

In the limit $\hbar\to0$ the quantum element therefore reproduces
$a^{\rm CB}$ exactly, including its sign relative to $a^{\rm PXR}$. With Shi's
stated $u_{\mathbf q}\approx u_{\mathbf q_i}$, every current is $\mathbf v_i$.
The $\mathbf w\cdot\mathbf e$ (transverse-acceleration) term then cancels
between the two orderings, and only the dispersion remainder survives. That
remainder has no $\sigma$-polarized ($\hat{\mathbf y}$) part.

### Rates

Different $\mathbf g_\perp$ add incoherently after the average over the
entrance position $\mathbf r_0$ that Shi states under Eq. (1). With
$E=\hbar\omega$, each row contributes a peak height and an area:

$$
\frac{d^2N}{dE\,d\Omega}\Big|_{\rm peak}=\frac{\alpha E L^2}{4\pi^2\beta^2(\hbar c)^2}\sum_\sigma\Bigl\lvert\frac{a_{\mathbf g\sigma}}{c}\Bigr\rvert^2,
\qquad
\frac{dN}{d\Omega}=\frac{\alpha E L}{2\pi\,\hbar c\,\beta\,(1-\boldsymbol\beta_f\cdot\hat{\mathbf n})}\sum_\sigma\Bigl\lvert\frac{a_{\mathbf g\sigma}}{c}\Bigr\rvert^2.
$$

In the area, $\boldsymbol\beta_f\to\boldsymbol\beta$ at the classical root,
and the final-state velocity carries the recoil Jacobian at the exact root.

### Cheap filters

| Filter | Result |
|---|---|
| Units | $a^{\rm PXR}/c$ and $a^{\rm CB}/c$ are dimensionless; peak in eV$^{-1}$ sr$^{-1}$, area in sr$^{-1}$ — pass |
| Limit $\hbar\to0$ | quantum CB element equals classical $a^{\rm CB}$ (algebra above; numerically 1.01 at 5 MeV) — pass |
| Correspondence | at 5 MeV exact/classical area ratio with identical elements on both sides is 1.003 / 1.000 — pass |
| Sign | PXR–CB interference is constructive for every row at 30 keV, 300 keV and 5 MeV — pass |
| Shi's approximation | all-$\mathbf v_i$ currents cancel the $\mathbf w\cdot\mathbf e$ term — pass (consistent with the row's Notes) |

## 2. Shi's CL and QM from Shi's inputs

Inputs: Salvat Yukawa ($C=0.5160,0.4840$; $\mu=5.8492,1.1732$), $a=5.431$ Å,
no $f'$, no Debye–Waller, diamond-allowed $\lvert n_x\rvert,\lvert n_y\rvert\le 8$.
The normalization is absolute from Eq. (1); nothing is calibrated against
PyRITE.

### Reference data spot-check

I rendered PDF page 5 at 600 dpi (sha256 `50ebf2ab…` matches the pinned
value). The axis ticks and the topmost black (CL) pixel of each peak were
located by pixel scan.

| Panel, $n_z$ | own read | pinned JSON | axis used |
|---|---|---|---|
| f, −2 | $1.160\times10^{-9}$ | $1.165\times10^{-9}$ | left |
| f, −3 | $1.213\times10^{-9}$ | $1.211\times10^{-9}$ | right |
| d, −2 | $5.88\times10^{-11}$ | $5.96\times10^{-11}$ | left |
| d, −3 | $5.89\times10^{-11}$ | $5.82\times10^{-11}$ | right |
| b, −2 / −3 (visual) | $4.5$ / $3.8\times10^{-11}$ | $4.53$ / $3.80\times10^{-11}$ | shared scale |

The left/right scale ratios are 3.33 (panel d: 6e-11 against 2e-10 over the
same height) and 3.51 (panel f: 1e-9 over 383 px against 4e-9 over 437 px).
The digitized CL peak positions sit within 0.15 % of my classical roots, and
the QM peaks sit within 3.2 eV (30 keV) and 1.2 eV (300 keV) of the
$(0,\pm2,-2)$ exact root.

### Peaks and area ratios

Here "q" means quantum second-order CB with exact energies and transition
current $c^2(\mathbf p_a+\mathbf p_b)/2\sqrt{E_aE_b}$; "cl" means the classical
element. The current variants $(\mathbf v_a+\mathbf v_b)/2$ and
$(\mathbf p_a+\mathbf p_b)/(E_a+E_b)$ give identical medians.

| $T$, $n_z$ | Shi CL / model PXR | Shi CL / model PXR+CB (cl) | Shi CL / Σ rows PXR+CB (q, exact roots) | Shi QM/CL area | model (PXR+CB q, exact)/(PXR, cl) | model with CB (cl) |
|---|---|---|---|---|---|---|
| 30 keV, −2 | **1.08** | 0.33 | 0.27 | 3.88 | **4.04** | 3.54 |
| 300 keV, −2 | **0.73** | 0.34 | 0.32 | 2.91 | **2.24** | 2.16 |
| 5 MeV, −2 | **1.16** | 1.02 | 1.02 | 1.13 | **1.14** | 1.14 |
| 30 keV, −3 | 2.56 (right axis) | 0.69 | 0.59 | 5.83 | 4.31 | 3.93 |
| 300 keV, −3 | 1.58 (right axis) | 0.73 | 0.70 | 4.31 | 2.25 | 2.19 |
| 5 MeV, −3 | 1.04 | 0.97 | 0.98 | 1.06 | 1.07 | 1.07 |

On the axis-unambiguous $n_z=-2$ group, Shi's CL height is the PXR-only rate
(1.08 / 0.73 / 1.16), while PXR+CB overshoots it 3–4× below 1 MeV. That holds
whether CB is taken classically or quantum. Shi's QM/CL area excess is
(PXR+CB)/PXR-only: 4.04 against 3.88 at 30 keV, and 1.14 against 1.13 at
5 MeV. The 300 keV ratio (2.24 against 2.91, 0.77×) and the $n_z=-3$ ratios
(0.74×, 0.52×) are low in my model as well. With the classical element the
30 keV area ratio is 3.54, still within 10 % of Shi's 3.88.

The tallest QM line on the degenerate $(0,\pm2,-2)$ pair is
$1.15\times10^{-9}$ against Shi's $1.116\times10^{-9}$ at 30 keV. I evaluated
it from my Shi-input row heights, with quantum CB summed over the two
degenerate members.

### The $n_z=-3$ axis

On the $n_z=-3$ group my model's tallest QM line is
$2.92\times10^{-10}$ at 2072.7 eV (30 keV) and $2.07\times10^{-11}$ at
4655 eV (300 keV), with sinc² lines on the exact roots. Shi's peak read on
the left axis is $2.57\times10^{-10}$ (1.14×) and $1.58\times10^{-11}$
(1.31×). Read on the right axis it is $9.03\times10^{-10}$ and
$5.27\times10^{-11}$ (0.32×, 0.39×).

The CL readings point the other way at 300 keV. Model PXR-only over Shi is
1.37 / 2.1 on the left axis and 0.39 / 0.63 on the right. Neither axis fits
both curves. The ambiguity is real, and the row is right to exclude
$n_z=-3$ absolute heights. The axes are 3.3–3.5× apart, not just "about 3×".

### Shi's stated approximation

With all currents $\mathbf v_i$, the quantum CB intensity relative to
classical CB is 0.09 / 0.06 (30 keV, $n_z=-2$ / $-3$), 0.29 / 0.27 (300 keV)
and 0.35 / 0.34 (5 MeV). Those are amplitude factors of 0.25–0.59. The
resulting $n_z=-2$ QM/CL area ratios are 1.29 / 1.13 / 1.03 at 30 keV /
300 keV / 5 MeV, and cannot reach Shi's 3.88 / 2.91. The row's
inference stands: Shi's curves need momentum-dependent currents. The Notes'
"0.3–0.5×" should say whether it is amplitude or intensity, and against which
reference.

## 3. The PXR-only remainder: crystal potential

The independent inputs are Waasmaier–Kirfel Si coefficients
(Acta Cryst. A51, 416 (1995)), Kissel $f'$, $f''$ from xraylib and
$B=0.4632$ Å². The CB potential is Mott–Bethe $(Z-f_0)/g^2$ with the same
Debye–Waller factor and no $f'$.

| $T$, $n_z$ | WK $f_0$ / Yukawa (PXR) | + $f'$ | + DW | model PXR / Shi CL | model PXR+CB (cl) / Shi CL |
|---|---|---|---|---|---|
| 30 keV, −2 | 0.979 | 0.570 | 0.512 | **0.473** | **1.83** |
| 300 keV, −2 | 0.979 | 1.041 | 0.911 | 1.25 | **2.43** |
| 5 MeV, −2 | 0.980 | 1.085 | 0.938 | 0.81 | 0.91 |

The $(022)$ reflection at 1418.4 eV has $f_0^{\rm WK}=8.717$ and Kissel
$f'=-1.79$, $f''=0.59$. That gives
$\lvert f_0+f'+if''\rvert^2/f_0^2=0.635$ from $f'$ alone, and
$0.635\,e^{-2B s^2}=0.597$ with Debye–Waller.

PyRITE's own catalogue gives $f'=-1.78$, $\lvert F/f_0\rvert^2=0.6365$, and
0.5977 with the Debye–Waller factor. The ledger's "0.60" is therefore
$f'$ **and** Debye–Waller, not $f'$ alone. The research note's
"$f'\approx-1.5$" is wrong; it is −1.78. WK against Yukawa is a 2 % effect.
Below the K edge the remainder is dominated by $f'$ (0.57×), then
Debye–Waller (0.90×).

## 4. Implementation comparison

### Element by element

PyRITE labels rows $\mathbf g_{\rm P}=-\mathbf g_{\rm Shi}$, so
`detuning` $=g^2+2\mathbf k\cdot\mathbf g_{\rm P}=K^2-k^2$. `chi_g` builds
$-r_e\lambda^2 S/(\pi V)$ with $S=\sum F e^{+i\mathbf g_{\rm P}\cdot\mathbf R}e^{-W}$,
which is Shi's $S_{\mathbf g}$ at $\mathbf g_{\rm P}=-\mathbf g$. `U_g` returns
$+\lvert e\rvert\phi$, which is minus my $U_{\mathbf g}$: numerically
$4.459$ eV against my Yukawa $-4.411$ eV for $(022)$.

Substituting, the code's

```text
A_PXR = chi / detuning * (v_dot_kg * (g_dot_e + k_dot_e) - k_mag**2 * v_dot_e)
A_CBS = -eUg_over_m / (gamma * vdg) * (braced_ge + v_dot_e * braced_kg / vdg)
```

equals $-a^{\rm PXR}/c$ and $-a^{\rm CB}/c$ term by term in vacuum
($\delta=0$). It is the same in `_per_hkl.py` (coherent branch) and
`_batched.py`. The overall sign is common, so $\lvert A\rvert^2$ and the
interference sign agree.

### Absolute numerical comparison

I compared PyRITE's classical-hook `mc_cxr_spectrum(..., components=True)`
peak per row (single straight 30 nm segment, $B=0.4632$,
$\lvert n_x\rvert,\lvert n_y\rvert\le4$) with my absolute elements, fed
PyRITE's $\chi_{\mathbf g}$ and $U_{\mathbf g}$.

| $T$ | model/PyRITE PXR | CBS | interference |
|---|---|---|---|
| 30 keV | 1.002–1.008 | 1.002–1.007 | 1.002–1.007 |
| 300 keV | 1.000–1.001 | 1.000–1.001 | 0.999–1.001 |
| 5 MeV | 1.000–1.008 | 1.000 | 1.000–1.004 |

This is stronger than the row's "within 2.3 % at 5 MeV". The classical
elements, the absolute normalization and the constructive sign agree at all
three energies. The check's sub-MeV spreads (1.19 PXR scale, 1.36 CB/PXR) come
from its model keeping quantum currents and denominators, not from PyRITE.

### PyRITE production on Shi's window

These use the `_RECOIL_HBARC_EV_ANG = 0` hook for classical, with
$\lvert n_x\rvert,\lvert n_y\rvert\le8$.

| $T$, $n_z$ | classical total / Shi CL | PXR-only / Shi CL | exact total / classical PXR | exact total / classical total |
|---|---|---|---|---|
| 30 keV, −2 | **1.83** | **0.474** | 4.12 | 1.071 |
| 30 keV, −3 | 1.01 | 0.30 | 3.00 | **0.891** |
| 300 keV, −2 | **2.48** | 1.31 | 1.93 | 1.020 |
| 300 keV, −3 | 1.15 | 0.59 | 1.99 | 1.022 |
| 5 MeV, −2 | 0.93 | 0.83 | 1.12 | 1.003 |
| 5 MeV, −3 | 0.95 | 0.90 | 1.06 | 1.001 |

The 30 keV $n_z=-2$ peak shares are 26 % PXR, 31 % CBS and 43 % interference,
as in the note. The 0.891 is outside the row's "1.00 ± 0.08 (recoil Jacobian
only)", for three reasons:

- It does not depend on the window: 0.889 on 200–2248 eV, and no exact root
  falls below the window.
- My model with fixed couplings and the Jacobian gives +3 to +6 % per row,
  whereas PyRITE gives −1 to −8 % per row. The difference grows as the root
  moves toward the Si K edge.
- PyRITE re-evaluates $\chi_{\mathbf g}$ at each shifted root, where $f'$ and
  $f''\approx3.5$–3.8 change. For example, $\lvert F\rvert^2$ for $(333)$ falls
  16 % between 2128 and 1985 eV, and the complex $\chi$ changes the PXR–CBS
  interference.

The test asserts `abs=0.12`, so the anchor is consistent. The row's number and
cause are not. In the check, the "CB on both sides" column (1.00–1.04) is the
independent model, not PyRITE.

### Runs

| Command | Result |
|---|---|
| `uv run python checks/shi_2024_matrix_elements_1103.py` | rc 0, PASS. Prints 1.06/0.73/1.16, 0.27/0.33/1.02, 4.11/2.25/1.14 against 3.88/2.91/1.13; 5 MeV spread 1.021/1.011, CB/PXR and interference ≤ 1.023 |
| `uv run pyrite-dev test tests/montecarlo/spectrum/test_shi_2024_fig2_intensity.py` | 17 passed, 0.86 s |

The check's 4.11 against my 4.04 reflects two differences. The check sums row
heights at the exact roots without the $(1-\boldsymbol\beta_f\cdot\hat{\mathbf n})^{-1}$
width Jacobian, and it uses a current on the PXR vertex too. Both are
percent-level. The check's "PXR+CB" CL column evaluates quantum CB at the
off-shell classical root. My on-shell per-row sum gives the same 0.27 / 0.32 /
1.02, and the classical element gives 0.33 / 0.34 / 1.02. All three variants
are far below 1, so the attribution is robust.

## 5. Findings

1. **Attribution: rederived.** Shi's CL is PXR-only (Eq. (1) without
   $\delta\mathbf v$; Shi's own text says this "obtain[s] the emission rate of
   PXR"). The QM excess is CB plus interference. PyRITE's classical line is
   PXR + CBS + constructive interference. Every Claim number is reproduced:
   1.8× / 2.5×, 0.47, 3.9 / 2.9 / 5.8 / 4.3 / ≈ 1.1, 1.06 / 0.73 / 1.16,
   0.27 / 0.33 / 1.02, 4.11 / 2.25 / 1.14, and ≤ 2.3 %.
2. **Wording: $f'$ factor.** "Si K-edge f′ (|χ₀₂₂|² × 0.60 at 1.42 keV)" puts
   the combined $f'$ × Debye–Waller factor on $f'$. With $f'=-1.78$ the factor
   is 0.64 alone and 0.60 with Debye–Waller. The research note's
   "$f'\approx-1.5$" should read −1.78.
3. **Number: CBS on both sides.** The Checks field says "collapses every QM/CL
   area ratio to 1.00 ± 0.08 (recoil Jacobian only)". PyRITE production gives
   0.89–1.07; the 30 keV $n_z=-3$ value is 0.891, partly from energy-dependent
   K-edge couplings at the shifted roots. The independent model gives
   1.00–1.04. Research note §4 repeats the PyRITE "1.00 ± 0.08".
4. **Minor wording.**
   - "about 3× apart" is 3.3× (d) and 3.5× (f).
   - "raise sub-100 keV CB by a median 1.36× at 30 keV" is the $n_z=-2$ CB/PXR
     row ratio against PyRITE ($n_z=-3$: 1.19). My $\lvert{\rm CB_q}\rvert^2/\lvert{\rm CB_{cl}}\rvert^2$
     medians are 1.55 / 1.30 (pooled 1.34). The claim is "not claimed", but
     the quantity should be named.
   - The Notes' "0.3–0.5×" needs a basis: intensity 0.06–0.35 against classical
     CB, amplitude 0.25–0.59.
5. **Strengthening (optional).** The classical-limit agreement holds
   absolutely at 30 keV and 300 keV as well (≤ 0.8 %). The check could pin it
   without the PyRITE-derived calibration.

## 6. Verdict

`discrepancy`, on wording and one Checks-field number. The physics is
rederived: the element derivation, the attribution, the sign and every Claim
number match. Findings 2–3 must be corrected before `rederived` and
`anchored`. Finding 4 is recommended.

Suggested ledger edits (the owner applies them; never `signed-off`):

- Claim: replace "Si K-edge f′ (|χ₀₂₂|² × 0.60 at 1.42 keV), Debye–Waller and
  Waasmaier–Kirfel f₀" with "Si K-edge f′ = −1.78 (|χ₀₂₂|² × 0.64 at
  1.42 keV; × 0.60 with Debye–Waller), Debye–Waller and, at the 2 % level,
  Waasmaier–Kirfel f₀".
- Claim: "about 3× apart" → "3.3–3.5× apart". Write "raise sub-100 keV CB by a
  median 1.36× at 30 keV" as "raise CB relative to PXR by a median 1.36×
  (n_z = −2) and 1.19× (n_z = −3) at 30 keV".
- Checks: replace "CBS on both sides collapses every QM/CL area ratio to
  1.00 ± 0.08 (recoil Jacobian only)" with "CBS on both sides collapses every
  QM/CL area ratio to 1.00–1.04 in the independent model and 0.89–1.07 in
  production (recoil Jacobian plus energy-dependent K-edge couplings at the
  shifted roots; test tolerance ±0.12)".
- Notes: state the basis of "0.3–0.5×". Optionally record that the absolute
  classical elements agree with production within 0.8 % at 30 keV, 300 keV and
  5 MeV.
- Research note: "f′ ≈ −1.5" → "f′ ≈ −1.78" (§0), and correct the §4
  "1.00 ± 0.08".

### Re-check after owner corrections (2026-11-12)

The owner applied the findings in the uncommitted worktree on `3445bd19a`.
That covers the ledger row, the research note, the test docstring (no
assertion changed) and the toctree entry. Each corrected statement was
re-checked against §2–§4 above.

| Corrected wording | Own result | Status |
|---|---|---|
| Claim: f′ = −1.78, $\lvert\chi_{022}\rvert^2\times0.64$; × 0.60 with Debye–Waller; WK against Yukawa ≈ 2 % | PyRITE $f'=-1.78$, 0.6365, 0.5977; WK/Yukawa PXR 0.979–0.980 | pass |
| Claim: axis scales differ by 3.3× (d) and 3.5× (f) | 3.33 and 3.51 | pass |
| Claim: CB relative to PXR, median 1.36× ($n_z=-2$) and 1.19× ($n_z=-3$) at 30 keV | check output 1.358 / 1.189 | pass |
| Checks: CBS on both sides 1.00–1.04 model, 0.89–1.07 production, Jacobian plus K-edge couplings | model 1.00–1.04 (check); production 0.891–1.071; window-independent; $\chi_{\mathbf g}$ changes across the shifted roots | pass |
| Notes: $u_q\approx u_{q_i}$ gives 0.06–0.35× intensity, 0.25–0.59× amplitude, area ratios 1.29 / 1.13 / 1.03 | 0.06–0.35, 0.25–0.59, 1.288 / 1.125 / 1.031 | pass |
| Notes: absolute match within 0.8 % at 30 keV, 300 keV, 5 MeV | §4 table | pass |
| Research note: $f'=-1.78$; §4 0.89–1.07 with K-edge cause; §1 300 keV CL favours neither axis | §3, §4, §2 above | pass |
| Test docstring: 0.89–1.07 | matches production; assertion `abs=0.12` unchanged | pass |

Non-blocking nits (no effect on the verdict):

- **Notes, double "so".** The Notes sentence reads "…cannot produce their QM,
  so their curves imply…" with two consecutive "so" clauses.
- **Research note §2, n_z split.** "1.08× at 300 keV and 1.01× at 5 MeV" are
  the $n_z=-2$ medians; for $n_z=-3$ the values are 1.05 and 1.01.
- **Research note §1, axis caveat.** The model value 2.61e-10 is the check's.
  My sinc-convolved Shi-input model gives $2.92\times10^{-10}$ against the
  left-axis $2.57\times10^{-10}$. Either way the left axis is the closer read.
- **Test docstring line length.** The edited line is 136 characters against
  the configured line length of 100. Ruff passes because E501 is not enforced.

Re-check verdict: **`rederived`**. The physics matched in the first pass, and
every wording and number finding is now corrected consistently across the
ledger row, the note and the test docstring. The owner may apply `rederived`,
then `anchored` with the named passing anchors. Human sign-off is pending.
