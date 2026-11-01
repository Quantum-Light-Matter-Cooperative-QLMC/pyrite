# `optical-planar-tr`

## Claim and source

- Claim: optical TR of transported planar-stack crossings is the exact planar Maxwell (Ginzburg–Frank) crossing amplitude with an explicit measured scalar permittivity per layer, summed per history with crossing phases $e^{i(\omega t-\mathbf k_\parallel\cdot\mathbf r_\parallel)/\hbar c}$; matched media and vanishing films give zero, and the conductor limit is the Ginzburg–Frank form. Locally straight formation is bounded from transported straight runs; unbounded use is refused.
- Code: `montecarlo/spectrum/optical_transition.py::optical_transition_radiation`; `montecarlo/runner/transition_emission.py::case_optical_transition`; `montecarlo/spectrum/optical_transition_validity.py::formation_error_bound`; `montecarlo/spectrum/transition_radiation.py::interface_transition_radiation_amplitude`; `materials/optical.py::gold_johnson_christy_response`.
- Sources: Ginzburg and Tsytovich (1990) Ch. 2; Coenen, Vesseur and Polman, arXiv:1107.3632v1 Fig. 3 and Fig. 4 (withdrawn from arXiv; APL 99, 143103); Johnson and Christy, PRB 6, 4370 (1972).
- Units: Gaussian, $\hbar=c=1$ inside the solver; $E$ in eV, $\omega=E/\hbar c$ in Å$^{-1}$; output photons eV$^{-1}$ sr$^{-1}$ per electron.
- Frozen revision: commit `3caced48` (fresh-context verifier, 2026-10-22).

The derivation was written from the ledger row, the docstrings and the physics page. The implementation bodies were read afterwards.

## Independent derivation

### Single interface, normal incidence

A charge $e$ moves along $+z$ with speed $\beta$ from vacuum ($z<0$) into a medium $\varepsilon$ ($z>0$). In the plane of observation $(x,z)$ only $H_y$ and $E_x,E_z$ appear. With $e^{-i\omega t}$, $k_0=\omega$, $k_x=\omega\sin\theta$, $q=\omega/\beta$, the Lorenz-gauge own field in medium $j$ is $A_z\propto e^{iqz}/D_j$, $D_j=k_x^2+q^2-\varepsilon_j\omega^2$, so $H_y^{(j)}=h_je^{iqz}$, $h_j\propto 1/D_j$. Ampère's law without $J_x$ gives $E_x=(\partial_zH_y)/(i\omega\varepsilon)$: $E_x=qh_j/(\omega\varepsilon_j)$ for the own field and $\pm k_{zj}/(\omega\varepsilon_j)$ times the amplitude for free waves $e^{\pm ik_{zj}z}$, $k_{z1}=\omega\cos\theta$, $k_{z2}=\sqrt{\varepsilon\omega^2-k_x^2}$ ($\operatorname{Im}\ge0$). Continuity of $H_y$ and $E_x$ at $z=0$ gives the backward (reflected) amplitude

$$
R=\frac{(\varepsilon-1)\,(q^2+qk_{z2}-\omega^2)}{D_1\,(q+k_{z2})\,(\varepsilon k_{z1}+k_{z2})}
\;\propto\;\frac{\beta\,(\varepsilon-1)\bigl(1-\beta^2+\beta\sqrt{\varepsilon-\sin^2\theta}\bigr)}{(1-\beta^2\cos^2\theta)\bigl(1+\beta\sqrt{\varepsilon-\sin^2\theta}\bigr)\bigl(\varepsilon\cos\theta+\sqrt{\varepsilon-\sin^2\theta}\bigr)} .
$$

For $\varepsilon\to\infty$ (tangential $E$ vanishes), $R_{\rm PC}=qh_1/k_{z1}$, and the far field ($\propto k_{z1}^2|R|^2$) reproduces the textbook conductor result. Converting energy per $d\omega\,d\Omega$ into photons per $d(\hbar\omega)\,d\Omega$:

$$
\frac{d^2N_{\rm PC}}{dE\,d\Omega}=\frac{\alpha\beta^2\sin^2\theta}{\pi^2E\,(1-\beta^2\cos^2\theta)^2},
\qquad
\frac{d^2N}{dE\,d\Omega}=\frac{d^2N_{\rm PC}}{dE\,d\Omega}\,\Bigl|\frac{R}{R_{\rm PC}}\Bigr|^2 .
$$

The bracket in $R$ has no Cherenkov pole for backward emission ($1+\beta\sqrt{\cdot}$), and $R=0$ for $\varepsilon=1$.

### Crossing superposition and phase

The planar solve is linear in its source, the jump $-(\mathbf P_{i+1}-\mathbf P_i)$ of the own field at interface $i$. For a charge through $(t_c,\mathbf r_c)$, the source carries $e^{i(\omega t_c-\mathbf k_\parallel\cdot\mathbf r_{\parallel,c})}$. A straight trajectory's crossings sum to the full stack amplitude. Splitting the jump into its below/above own-field terms is an exact linear decomposition.

### Formation length and error measure

Each own-field term varies as $e^{iqz}$ with $q=(\omega-\mathbf k_\parallel\cdot\mathbf v_\parallel)/v_z$; free waves in the medium vary as $e^{\pm i\lambda z}$, $\lambda=\sqrt{\varepsilon\omega^2-k_\parallel^2}$. The source–wave mismatch builds over $1/|q\mp\lambda|$, so $L=\max(1/|q-\lambda|,1/|q+\lambda|)\,\hbar c$ is the longer (conservative) formation length. In a metal $|q\pm\lambda|\ge\operatorname{Im}\lambda$, so $L$ is at most about the skin depth. If the straight run on side $s$ is shorter than $L$, replacing that side's term $\mathbf A_s$ by one of no larger magnitude changes it by at most $2|\mathbf A_s|$. Then $|\delta\mathbf S_e|\le\sum_c2\sum_sd_s|\mathbf A_s|$ and $|\mathbf S+\delta\mathbf S|^2-|\mathbf S|^2\le2|\delta\mathbf S||\mathbf S|+|\delta\mathbf S|^2$. The premise (no larger magnitude) makes this a heuristic sensitivity measure, not a proven bound.

## Cheap filters

| Check | Result |
|---|---|
| Units | $\alpha/E$ gives eV$^{-1}$ sr$^{-1}$; $E\,n_z^2|\sum\mathbf a|^2/(16\pi^4)$ reproduces it numerically. Pass. |
| Limits | $\varepsilon=1\Rightarrow0$; $\varepsilon\to\infty\Rightarrow$ conductor form; vanishing film $\Rightarrow0$. Pass. |
| Conventions | $e^{-i\omega t}$, $\operatorname{Im}\varepsilon\ge0$, $\operatorname{Im}\lambda\ge0$; translation by $(\delta t,\delta\mathbf r)$ multiplies by $e^{i(\omega\delta t-\mathbf k_\parallel\cdot\delta\mathbf r)/\hbar c}$ for both faces. Pass. |

## Comparison with the implementation

- `interface_transition_radiation_amplitude`: phase `exp(1j*(omega*t - k_perp·r_perp)/HBARC_EV_ANG)` matches. `own_field_sides` column 0 scales the medium-below term (sign $+1$) and column 1 the medium-above term (sign $-1$); the default $(1,1)$ is the physical jump.
- `optical_transition_radiation`: prefactor `energy * n_z**2 / (16 pi^4)` matches.
- `boundary_crossings` / `_straight_runs`: a run is a maximal same-direction row sequence of one track; a direction change above $10^{-12}$ ends it. Entry: run before $\infty$ (vacuum), run after the first row's forward run. A face at a row end: that row's backward run before; after an inner boundary the next row's forward run (0 if the track ends, conservative); after an exit $\infty$. Runs crossing layers are correct because the planar solution is exact for any straight multi-layer passage.
- `formation_error_bound`: the solver-to-slab layer map is correct for both faces; $L$ uses the minimum mismatch (maximum length); accumulation and self-power ratio match.
- `case_optical_transition`: band support, explicit per-layer responses, charge agreement and history range, then formation and energy-budget refusals. Matches the documented contract.

### Independent numerical checks

1. `planar_transition_radiation` against the closed form above for $\varepsilon\in\{-10+1.5i,\,2.25+0.01i,\,-1.3+3i,\,5,\,-10^6+10^3i\}$, $\beta\in\{0.1,0.33,0.9\}$, 9 polar angles: relative difference $6.1\times10^{-10}$ (constant; CODATA α-value difference).
2. Fig. 3 re-digitized at 400 dpi (frame plus 400–800 nm and 1.2–2.0 ticks; calibration residual below 0.003): all 16 anchor values within 0.004 (in $10^{-7}$ nm$^{-1}$); dip 484 nm, peak 642 nm.
3. Hemisphere-integrated backward TR per nm, 30 keV ($\beta=0.32838$), Johnson–Christy n,k from the pinned record (results insensitive to ε, n,k-in-E or n,k-in-λ interpolation; 200-point Gauss–Legendre). Integral over 350–950 nm: $1.54\times10^{-4}$ photons per electron.

   | λ [nm] | 375 | 400 | 450 | 485 | 500 | 550 | 600 | 650 | 700 | 800 | 900 |
   |---|---|---|---|---|---|---|---|---|---|---|---|
   | ours [$10^{-7}$/nm] | 2.351 | 2.245 | 1.981 | 1.839 | 1.910 | 2.598 | 2.877 | 2.953 | 2.917 | 2.765 | 2.603 |
   | published/ours | 0.677 | 0.679 | 0.677 | 0.661 | 0.672 | 0.692 | 0.696 | 0.696 | 0.697 | 0.696 | 0.697 |

4. Hypotheses for the 0.696 factor that change the shape and are therefore excluded: polar caps at 60–85°; $\cos\theta$ weighting; non-relativistic $\beta$ (ratio 0.637); 15–25 keV beams (ratio drifts by 5% across the band). The paper's own band integral (about $10^{-4}$) and its measured photon rate (3.9e6/s against about 4e6/s expected) are consistent with its lower scale, within loose mirror/reflection/QE estimates.
5. Angular peaks: per steradian 54.8° (400 nm) and 65.8° (900 nm); $\sin\theta$-weighted 59.6° and 68.7°. A rough re-digitization of the measured Fig. 4c lobes gives about 58.5° and 65.5°, consistent with neither convention at both wavelengths.

`tests/montecarlo/spectrum/test_transition_radiation_optical.py` and `test_transition_radiation_optical_anchor.py`: 23 passed at the frozen revision.

## Adjudication

No divergent factor, sign, exponent, unit or convention. The planar amplitude, crossing phase, linear own-field split, normalization and straight-run bookkeeping match the independent derivation; the general-ε closed form agrees with the code to $6\times10^{-10}$. Verdict: **rederived**.

What is anchored:

- Absolute normalization, via the closed-form conductor limit and (owner follow-up) the finite-ε Ginzburg–Frank closed form for a charge entering bulk gold at 1.4, 2.5 and 3.3 eV (`test_finite_gold_permittivity_matches_closed_form_ginzburg_frank`, $10^{-9}$).
- The matched-medium and vanishing-film zeros.
- The normalized spectral shape of the published Coenen Fig. 3 *calculation* with Johnson–Christy optics.

What is not anchored:

- The absolute Coenen scale: unresolved constant 0.696, reproduced by the independent implementation, so not a code defect.
- The measured Fig. 4c angular patterns; and any measurement (Fig. 3 is theory).
- Scattered or multi-crossing histories beyond the heuristic formation measure. The APL version was not consulted; the source is a withdrawn preprint.

### Findings and owner disposition

1. Low, anchor robustness: add a general complex-ε closed-form regression. Applied (finite-gold test above).
2. Low, anchor interpretation: the 375–550 nm residual is systematic (+0.018 to +0.032 normalized, 3–5% relative, largest at the dip); over 375–450 nm node spacing does not explain it. The 0.04 tolerance passes with about 0.008 margin. Recorded in the ledger notes.
3. Low, wording: the formation measure is heuristic and covers the self power only. Applied to the module docstring, physics page and ledger row.

### Caveats

- Vacuum runs are taken as infinite (straight upstream beam optics and free exit).
- Energy loss along a run does not end it; negligible over optical $L$.
- No finite radiator, near field, optics or detector response.
