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

## Fresh-context review of `coherent_formation_error_bound` (2026-11-08)

The propagation of the per-history formation bound through the self, pair and total estimators was re-derived and compared at `b0860fbfa`. The verdict is **rederived**: the algebra matches term by term, a 20 000-trial randomized perturbation test shows no violation, and the self bound is tight. The write-up is in [`optical-ctr-population`](optical-ctr-population.md#fresh-context-review-of-the-615-follow-up-2026-11-08). The factor-2 premise for $b_h$ remains a documented heuristic.

## Owner follow-up: APL version (2026-11-08)

The published APL letter (99, 143103) was read. It states that Fig. 3 is the TR probability "for the entire upper hemisphere" calculated with García de Abajo, RMP 82, 209 and Johnson–Christy constants, and that Fig. 4c is the measured pattern at 400 and 900 nm (40 nm band-pass) "integrated over azimuthal angle", excluding φ = 150–210° and θ > 85°. The flux was first converted to photons per solid angle.

- **Angular pattern.** Digitized Fig. 4c measured curves (`test_coenen_measured_angular_pattern_is_the_azimuth_integrated_flux`) agree with the band-averaged $\sin\theta\,dN/d\Omega$ shape to RMS 0.023 (400 nm) and 0.05 (900 nm) over 24–84°. Peaks: measured 61° and 66–68°, model 59.5° and 68.5°. The per-sr shape (peaks 55°, 66°) is excluded at RMS 0.14. The plotted quantity is therefore $dN/d\theta$ (azimuth integral over $d\Omega$). The 900 nm residual is largest at 72–80° (measured low; mirror edge) and below 0.3 where the curves overlap in the raster.
- **Absolute factor 0.696.** Not a mirror-acceptance factor (the text says full hemisphere). Not an effective beam energy: the published/ours ratio spread over 375–900 nm is 5.5 % at 30 keV and grows to 6.7–9.5 % at 25–15 keV. The paper's absolute check (8×10⁶ s⁻¹ generated at 12 nA, ~4×10⁶ s⁻¹ expected after acceptance/reflection/QE, 3.9×10⁶ s⁻¹ measured) assumes an unitemized ~50 % efficiency; our scale would need ~34 %. The measurement cannot discriminate. Remaining step: compare with García de Abajo RMP 82, 209 Eq. for TR probability directly.

## Fresh-context review of the APL Fig. 4c and RMP-scale anchors (2026-11-09)

Scope: `test_coenen_measured_angular_pattern_is_the_azimuth_integrated_flux` and `test_cited_theory_reproduces_our_absolute_scale_not_the_published_curve` at `6e12e4de0`. Sources read: Coenen, Vesseur and Polman, APL 99, 143103 (2011), pp. 3–4; García de Abajo, arXiv:0903.1669 (preprint of RMP 82, 209), Sec. IV.B–C, pp. 30–32. The model below was written from these sources with the Johnson–Christy n,k read directly from the pinned `Au-Johnson.yml` record. No PyRITE solver function was used. Both tests pass (6 passed with the Schmidt module).

### Claim A: measured Fig. 4c angular pattern

**What the text says.** The CCD image is converted "to photon flux per unit of solid angle" (Fig. 2(c) map). The data are "interpolated to yield points that are equally spaced in $\theta$ and $\varphi$". For Fig. 4c, "the emission patterns were integrated over azimuthal angle $\varphi$", with $\varphi=150$–$210^\circ$ dropped. The caption reads "Emission pattern integrated over $\varphi$". The text does not say whether the $\varphi$ integral carries the solid-angle Jacobian. Read literally, a uniform-grid sum over $\varphi$ of a per-sr quantity is $\propto dN/d\Omega$. The measured curves exclude that reading (below). Read as an integral over $d\Omega=\sin\theta\,d\theta\,d\varphi$ at fixed $\theta$, it gives

$$
\frac{dN}{d\theta}=\int d\varphi\,\sin\theta\,\frac{dN}{d\Omega}\propto\sin\theta\,\frac{dN}{d\Omega},
$$

and the data match this. The operator is therefore inferred from the measurement, not stated by the paper. Two hypotheses with no free parameters are compared, and the margin is a factor of about 6 in RMS, so the inference is decisive. It should still be recorded as an inference.

**Independent re-digitization.** Raster: `pdfimages -f 4 -l 4 -png`, image 1 (752×1485 px). The polar centre is the baseline midpoint, $(376, 1470.5)$ px in the full image ($y$ offset 1000 gives the owner's $(376,470)$). The unit circle is the solid outer arc. A radial histogram of dark pixels peaks at $330\pm1$ px. The dotted grid circles sit at 66, 131, 197 and 264 px on the baseline, i.e. 0.2–0.8. Colour masks: red $r>170,\,g,b<100$; green $g-r>40$. Each value is the median radius in a $\pm1^\circ$ bin on each side. Left and right lobes agree to about 0.02 above $40^\circ$. The bins at $36$–$40^\circ$ (400 nm) and $56^\circ$ (900 nm) on the right are contaminated by the curve labels, and only the left side is used there.

| $\theta$ [deg] | 24 | 32 | 40 | 48 | 56 | 60 | 64 | 68 | 72 | 76 | 80 | 84 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 400 nm, this review (L/R mean) | 0.148 | 0.362 | 0.642 | 0.843 | 0.965 | 0.986 | 0.942 | 0.898 | 0.764 | 0.583 | 0.344 | 0.209 |
| 400 nm, owner | 0.163 | 0.351 | 0.662 | 0.831 | 0.961 | 0.984 | 0.942 | 0.904 | 0.773 | 0.584 | 0.354 | 0.204 |
| 900 nm, this review | 0.070 | 0.152 | 0.444 | 0.674 | 0.867 | 0.941 | 0.972 | 0.972 | 0.931 | 0.800 | 0.624 | 0.369 |
| 900 nm, owner | 0.093 | 0.156 | 0.446 | 0.672 | 0.848 | 0.941 | 0.967 | 0.970 | 0.934 | 0.800 | 0.630 | 0.411 |

Agreement is within 0.02 above $30^\circ$, except at 900 nm and $84^\circ$. There, 1° bins give 0.366/0.372 against the owner's 0.411. The curve falls by about 0.065 per degree there, so the difference is an angle offset of about $0.6^\circ$. Below $30^\circ$ the red, green and dashed curves overlap near the origin, and values differ by up to 0.05.

**Independent model.** Backward Ginzburg–Frank emission of a charge entering gold ($\beta=0.328376$ at 30 keV), photons per eV per sr:

$$
\frac{d^2N}{dE\,d\Omega}=\frac{\alpha\beta^2\sin^2\theta\cos^2\theta}{\pi^2E}
\left|\frac{(\varepsilon-1)\bigl(1-\beta^2+\beta\sqrt{\varepsilon-\sin^2\theta}\bigr)}
{(1-\beta^2\cos^2\theta)\bigl(1+\beta\sqrt{\varepsilon-\sin^2\theta}\bigr)\bigl(\varepsilon\cos\theta+\sqrt{\varepsilon-\sin^2\theta}\bigr)}\right|^2 ,
$$

converted to per nm by $E/\lambda$ and averaged over a flat 40 nm band (41 points). Against this review's digitization over 24–84°:

| Model shape | 400 nm peak / RMS | 900 nm peak / RMS |
|---|---|---|
| $\sin\theta\,dN/d\Omega$, band-averaged | 59.8° / 0.025 | 68.8° / 0.051 |
| $\sin\theta\,dN/d\Omega$, monochromatic | 59.8° / 0.025 | 68.8° / 0.051 |
| $dN/d\Omega$ (per sr) | 54.8° / 0.146 | 65.8° / 0.143 |
| $\sin^2\theta\,dN/d\Omega$ | 62.8° / 0.101 | 70.5° / 0.101 |
| exiting-charge sign ($\mp\beta\sqrt{\cdot}$) | 59.8° / 0.025 | 68.8° / 0.051 |
| perfect conductor, $\sin\theta\,dN/d\Omega$ | 89.8° / 0.36 | 89.8° / 0.23 |

The owner's numbers (0.023/0.05; per sr 0.14) reproduce. The 900 nm residual is structured: the model is high by 0.05–0.10 at 24–36° and by 0.05–0.09 at 72–80°. The measured lobe is narrower than the model. This is consistent with, but not proven to be, mirror acceptance. The open paraboloid removes part of the $\varphi$ range at large $\theta$ (Fig. 4a), and the text does not say this was compensated. That loss would suppress the large-$\theta$ tail. It would move a per-sr pattern further from the data, so it does not weaken the exclusion of the per-sr reading.

**What the anchor constrains.**

- The normalized polar shape at 400 and 900 nm, 30 keV, normal incidence. This shape tests the finite-ε gold response: a perfect conductor is excluded at RMS 0.23–0.36.
- The collection operator ($dN/d\theta$), inferred from the data as described above.

**What it does not constrain.**

- Absolute yield, and therefore the 0.696 question.
- The band-pass model: monochromatic and 40 nm averages agree to $<10^{-3}$.
- Entering versus exiting sign: the two give identical shapes to three decimals for gold at $\beta=0.33$.
- $\varphi$ dependence (azimuthal symmetry is assumed), polarization, angles below about $30^\circ$ where the curves overlap, and other energies or materials.

The paper's own dashed theory curve also follows the data, so this is a measured anchor in normalized shape only. It is legitimate at that scope. The 0.035/0.065 RMS limits pass with margin 0.010/0.014 on this independent digitization.

### Claim C: RMP Sec. IV.C absolute scale

**Transcription.** The preprint uses medium 1 = vacuum above ($z>0$) and medium 2 = metal ($z<0$).

$$
\mu_1=\frac{-(\omega/v)\epsilon_2+q_{z2}\epsilon_1}{q^2-k^2\epsilon_1}-\frac{-(\omega/v)\epsilon_1+q_{z2}\epsilon_1}{q^2-k^2\epsilon_2},\qquad
D=\frac{2ie\,q_\parallel/c}{q_{z1}\epsilon_2+q_{z2}\epsilon_1},
$$

$q^2=q_\parallel^2+(\omega/v)^2$, $q_{zj}=\sqrt{k^2\epsilon_j-q_\parallel^2}$, $\mathbf f_H=ik\cos\theta\,D\,\mu_1\hat\varphi$ at $q_\parallel=k\sin\theta$, and $\Gamma_{\rm CL}=|\mathbf f|^2/(4\pi^2\hbar k)$ per sr per unit $\omega$. The test's `mu1`, `d` and `per_sr` match term by term with $\epsilon_1=1$, $\epsilon_2=\varepsilon$, $e=\sqrt\alpha$ ($\hbar=c=1$, eV), $\omega/v=+k/\beta$, and $\epsilon_1$ multiplying $q_{z2}$ in both numerators. In the test $|\mathbf f|=|\mathbf f_H|$ because the far field is transverse in vacuum. With $\hbar=1$ and energies in eV, per unit $\omega$ is per eV.

**Sign of $v$.** Using $q^2-k^2\epsilon_2=w^2-q_{z2}^2$ with $w=\omega/v$, Eq. (52) reduces exactly to

$$
\mu_1=\frac{(\varepsilon-1)\,(k^2-w^2-w\,q_{z2})}{(w^2-q_{z1}^2)\,(w+q_{z2})}.
$$

For $w=+k/\beta$, the factor $(1-\beta^2+\beta\sqrt{\varepsilon-\sin^2\theta})/(1+\beta\sqrt{\varepsilon-\sin^2\theta})$ appears. This is the pole-free form for backward emission by a charge entering the metal, the Coenen geometry. For $w=-k/\beta$, the Cherenkov denominator $1-\beta\sqrt{\cdot}$ appears, i.e. the exiting/forward form. The preprint's $e^{i\omega z/v}$ with the metal at $z<0$ does not fix the sign of $v$ for an impinging electron unambiguously. The owner's choice is the physically correct one. Numerically the hemisphere yields of the two signs differ by only $-3.3\%$ to $+0.3\%$ over 375–900 nm, so the choice cannot explain 0.696.

**Agreement.** With $w=+k/\beta$, $\Gamma_{\rm CL}$ equals the closed form above to $10^{-6}$ at every wavelength (the algebra makes them identical). The test's $10^{-3}$ is therefore a quadrature tolerance. This "cited-theory" check is the same Ginzburg–Frank physics, written in another formalism. It is an independent transcription, not independent physics.

**$\sin\theta$ in Eq. (56).** The preprint prints $\Gamma_{\rm TR}=(2\pi\hbar k)^{-1}\int_0^{\pi/2}d\theta\,|\mathbf f|^2$. Integrating the per-sr $\Gamma_{\rm CL}$ over $d\Omega=2\pi\sin\theta\,d\theta$ gives $(2\pi\hbar k)^{-1}\int\sin\theta\,d\theta\,|\mathbf f|^2$, so the printed form lacks $\sin\theta$. This is confirmed for the arXiv preprint only; the journal version was not checked. Without $\sin\theta$ the hemisphere value is 1.264–1.363 times ours over 375–900 nm. That is the wrong direction for 0.696: published/no-$\sin\theta$ is 0.49–0.55.

**Published/ours** (this review, owner's Fig. 3 values; the APL raster image 0 was compared visually and agrees): 0.677 (375), 0.679 (400), 0.660 (485), 0.692 (550), 0.697 (600–900 nm, constant to 0.2 %).

**Conclusion check.** The statement that a faithful implementation of the cited RMP formalism reproduces our hemisphere integral, not the published curve, is supported. The tested variants were: sign of $v$ (≤3 %), the omitted $\sin\theta$ (×1.3, wrong direction), monochromatic versus band (none) and $n,k$ interpolation (insensitive). None of them produces a constant 0.70. "Discrepancy of the published curve with its cited theory" is a fair label if it means unexplained by that theory. It is not evidence of an error in the paper. The paper may have used inputs or a code (e.g. BEM) other than Eq. (56), and its cause is unknown. The record should keep "cause unknown" and cite the preprint (not the journal) for the Eq. (56) $\sin\theta$ observation.

### Verdict

- Claim A: **anchored scope confirmed** for the normalized measured shape. Record that the $\sin\theta$ operator is inferred from the data (text ambiguous), and that the band-pass and entering/exiting sign are not constrained.
- Claim C: **rederived** (transcription, sign choice and $\sin\theta$ statement verified). No divergent factor in the implementation.
- Ledger status of `optical-planar-tr`: keep `anchored`.

### Owner disposition (2026-11-09)

`anchored` kept. The ledger notes now record that the Fig. 4c comparison was reviewed, that the sinθ reading is inferred from the data, and what the comparison leaves unconstrained. They also record that the RMP check is a transcription of the same physics, that Eq. (56) omits sinθ in the arXiv preprint, and that the cause of 0.696 is unknown. The 900 nm value at 84° (owner 0.411, reviewer 0.369) is left as digitized; the test tolerance (RMS < 0.065) covers it.

## Phase formation policy (fresh-context review, 2026-11-09)

Scope: the opt-in `kink_phase_tol` / `formation_kink_phase` policy of `formation_error_bound` and `coherent_formation_error_bound`, reviewed at `d882faf32`. The derivation below was written before the implementation was opened.

### Independent derivation

**Geometry.** Let $s$ be the path length from the crossing and $\hat{\mathbf u}(s)$ the piecewise-constant unit direction, $\hat{\mathbf u}_0$ at the crossing, with kinks $\Delta\hat{\mathbf u}_i=\hat{\mathbf u}_i-\hat{\mathbf u}_{i-1}$ at $s=d_i$. Then

$$
\mathbf d(s)\equiv\mathbf r(s)-\hat{\mathbf u}_0 s=\int_0^s\bigl(\hat{\mathbf u}(s')-\hat{\mathbf u}_0\bigr)\,ds'=\sum_i (s-d_i)_+\,\Delta\hat{\mathbf u}_i ,
$$

and the triangle inequality gives, for every $s\le L$,

$$
|\mathbf d(s)|\le\sum_i(s-d_i)_+|\Delta\hat{\mathbf u}_i|\le D(L)\equiv\sum_{d_i<L}(L-d_i)\,|\Delta\hat{\mathbf u}_i| ,\qquad |\Delta\hat{\mathbf u}_i|=2\sin(\theta_i/2).
$$

$D$ is non-decreasing in $L$. It is an upper bound, not the displacement: it is attained for a single kink and for coplanar same-sense kinks, and it stays positive when later kinks restore the original direction or position. The chord $2\sin(\theta_i/2)$ is the correct weight; $\sin\theta_i$ would not bound the displacement.

**Phase.** The own-field source phase is $\omega t-\mathbf K\cdot\mathbf r$. At constant speed $t=s/v$ on the bent and on the straight path alike, so the $\omega/v$ term cancels at equal $s$ and the phase difference is $-\mathbf K\cdot\mathbf d(s)$. The wave vector is that of the free wave in the side medium, $\mathbf K=(\mathbf k_\parallel,\pm\lambda)$, with real $\mathbf k_\parallel$ and complex $\lambda=\sqrt{\varepsilon\omega^2-k_\parallel^2}$. For real $\mathbf d$, Cauchy–Schwarz gives

$$
|\mathbf K\cdot\mathbf d|\le\|\mathbf K\|\,|\mathbf d|,\qquad \|\mathbf K\|^2=k_\parallel^2+|\lambda|^2=\omega^2\bigl(\sin^2\theta+|\varepsilon-\sin^2\theta|\bigr)\ge|\varepsilon|\omega^2 .
$$

This Hermitian norm is the one that makes the bound rigorous in an absorbing medium; it bounds the modulus of the complex phase change (real phase and attenuation together), with $|e^{-i\mathbf K\cdot\mathbf d}-1|\le e^{\|\mathbf K\|D}-1$. $|n|\omega$ or $\operatorname{Re}n\,\omega$ would be too small. Energy loss inside $L$ (a change of $v$) is not covered.

**Path length of the formation zone.** The mismatch phase is $(q\mp\lambda)z=(\omega-\mathbf K\cdot\mathbf v)\,s/\beta$, so $L=\hbar c/|q\mp\lambda|$ is a depth along the normal. The path length over which the same phase accumulates is

$$
\boxed{L_s=\frac{L}{|\hat u_{0z}|}} .
$$

Distances $d_i$ are path lengths, so the displacement bound that covers the formation zone is $D(L_s)$, not $D(L)$.

**Tolerance 0.** With $|\Delta\hat{\mathbf u}_i|>0$ for every listed kink and $\|\mathbf K\|>0$, $\|\mathbf K\|D(L)>0$ holds exactly when some kink has $d_i<L$, so "path ends inside $L$ or $\|\mathbf K\|D(L)>0$" equals "straight run $<L$". A kink at $d_i=L$ gives zero in both. A zero-angle kink would break the equality only if it ended a default run.

**What the phase test does not bound.** The source is the current $\hat{\mathbf u}(s)e^{i(\omega s/\beta-\mathbf K\cdot\mathbf r)}$. A kink also changes the vector factor, by a relative amount of order $D(L)/L$ over the zone. This exceeds $\|\mathbf K\|D(L)$ whenever $\|\mathbf K\|L<1$.

### Comparison with the implementation

| Term | Independent | Code | Result |
|---|---|---|---|
| Kink weight | $2\sin(\theta_i/2)$ | `jump = norm(v_hat[1:] - v_hat[:-1])` in `_kink_lists` | match |
| Displacement | $\sum_{d_i<L}(L-d_i)|\Delta\hat{\mathbf u}_i|$, upper bound | `sum(clip(length - distance, 0) * bend)` in `_phase_short` | match |
| Wave-vector norm | $\sqrt{k_\parallel^2+|\lambda|^2}$ | `sqrt(k2 + abs(lam)**2) / HBARC_EV_ANG` | match |
| Units | Å, 1/Å, rad | `length` [Å], `wavenumber` [1/Å], `kink_phase_tol` [rad] | match |
| Time lag | cancels at equal $s$ (constant speed) | not present; docstring "Arc length fixes the time" | match |
| Path end, slot overflow | short if the usable path ends inside $L$ | `path < length`; 16th kink becomes the path end | match, conservative |
| Zero-angle kinks | must not be listed | same `> 1e-12` component test as `_straight_runs` | match |
| Formation distance along the path | $L/|\hat u_{0z}|$ | `length` (depth) compared with path distances | **differs by $1/|\hat u_{0z}|$** |
| Current direction term $D/L$ | not bounded by the phase test | not tested | undocumented limit of the heuristic |

### Numerical evidence

Scripts: `scratch/phase_policy_review.py`, `scratch/phase_policy_scene.py` (not committed).

- 20 000 random three-dimensional polylines (1–11 kinks, turn scale $10^{-3}$ to isotropic, random $L$): $\max_{s\le L}|\mathbf d(s)|/D(L)\le1.000000$, no violation. With random complex $\varepsilon$ ($\operatorname{Re}\in[-30,10]$, $\operatorname{Im}\in[0,30]$) and both signs of $\lambda$ (40 000 cases): $\max|\mathbf K\cdot\mathbf d|/(\|\mathbf K\|D)=0.9966$, no violation. Replacing the norm by $|n|\omega$ gives 130 violations, worst ratio 2.04.
- `_kink_lists` and `_phase_short` against an independent row walk and against `_straight_runs`, on 300 random multi-track scenes: 0 mismatches in 245 221 comparisons (tolerances 0, 0.01, 0.1, 1; tolerance 0 against the default). The short set shrinks monotonically with the tolerance.
- The first-kink distance and the default run are computed by different floating-point expressions and differ by up to $1.2\times10^{-10}$ relative in these scenes. With $L$ placed within one ulp of the run, tolerance 0 and the default disagree in 34 610 of 144 000 cases. The equality is therefore exact mathematically and up to rounding of the two distances; it is not bit-exact for $L$ inside that window.
- `_formation_length` for gold-like $\varepsilon=-2.55+3.6i$ at 2.5 eV, $\theta=0.9$ rad, $\beta=0.3284$: path formation length over the coded `length` is 1.000, 2.000 and 5.759 at 0°, 60° and 80° from the normal, equal to $1/|\hat u_{0z}|$. $\|\mathbf K\|L$ is 0.79, 0.54 and 0.19.
- Test scenes, self bound at the four test energies, as coded and with `length` divided by $|\hat u_{0z}|$:

| Scene | Policy | As coded | With $L/|\hat u_{0z}|$ |
|---|---|---|---|
| HOPG transport, Au optics | default / tolerance 0 | 0.018, 0.056, 0.171, 0.138 | unchanged |
| HOPG transport, Au optics | 0.1 rad | 0 | 0 |
| Gold | default / tolerance 0 | 0.178, 0.460, 1.467, 1.117 | 0.246, 0.617, 1.835, 1.364 |
| Gold | 0.1 rad | 0.057, 0.111, 0.567, 0.227 | 0.216, 0.483, 1.507, 1.173 |

  The documented 0.171, 1.47 and 0.57 reproduce. The gold scene has crossings down to $|\hat u_{0z}|=0.13$ (obliquely exiting backscatter).
- One kink at $L/2$, real $n=1.5$, relative change of $\int_0^L\hat{\mathbf u}\,e^{i(\omega s/\beta-\mathbf K\cdot\mathbf r)}ds$: at $\beta=0.33$, turn 0.1 rad, $\|\mathbf K\|D=0.049$ and the change is 0.052; at $\beta=0.05$, turn 2 rad, $\|\mathbf K\|D=0.068$ and the change is 0.87; at $\beta=0.02$, full reversal, $\|\mathbf K\|D=0.031$ and the change is 1.03. The change follows $D/L$, not $\|\mathbf K\|D$.

The focused tests (`-k "kink or formation"`) pass: 4 passed.

### Findings

1. **Medium, discrepancy.** `optical_transition_validity.py:257` returns the depth $L=\hbar c/|q\mp\lambda|$; `:237` (default) and `:273`–`:276` (phase policy) compare it with path lengths. The first divergent factor is $1/|\hat u_{0z}|$. Both policies are anti-conservative for oblique crossings, and the phase policy more so because $D$ grows with the reach. Gold scene at 0.1 rad and 2.5 eV: 0.567 as coded, 1.507 with the path formation length. The default policy inherits the same factor (1.467 against 1.835); the earlier reviews did not check it. Normal-incidence entry crossings are unaffected, so the HOPG numbers stand.
2. **Low to medium, wording and scope of the heuristic.** The phase test does not bound the change of the current direction, of relative size $D(L)/L$. For $\|\mathbf K\|L<1$ this term is the larger one, and for slow charges ($\|\mathbf K\|L\approx\beta|n|$) even a reversal inside $L$ passes a 0.1 rad tolerance. The module docstring (`:52`–`:66`) and the physics page (`optical-transition-radiation.md:177`–`:192`) name only "sub-tolerance phase is straight" as the heuristic. A test on $\max(\|\mathbf K\|,1/L)\,D(L)$ would cover both terms.
3. **Informational.** "Tolerance 0 reproduces the default exactly" is true mathematically and in every randomized comparison, but the two distances come from different floating-point expressions (`transition_radiation_crossings.py:339`–`:346` against `:372`–`:373`), so it is not bit-exact when $L$ falls within rounding of a run. The `assert_array_equal` in the test is safe in practice.
4. **Informational.** Constant speed inside $L$ is assumed and not stated as an assumption; the bound uses the crossing's kinetic energy.

### Wording

"Exact displacement bound, heuristic acceptance" is accurate for the inequality $|\mathbf r(s)-\mathbf r_0(s)|\le D(L)$ on $s\le L$ and for $\|\mathbf K\|$. It is not accurate as a statement about the formation zone until $L$ is converted to a path length (finding 1), and "exact geometry" should not be read as covering the current direction (finding 2).

### Verdict

**discrepancy** for the phase-policy sub-claim: the factor $1/|\hat u_{0z}|$ between the coded formation depth and the path distances it is compared with. The displacement bound, the wave-vector norm, the units, the tolerance-0 reduction, the `None` default and the monotonicity are rederived and match. Suggested ledger note: record the phase policy as reviewed with this discrepancy open; the status change belongs to the task owner.

### Owner disposition (2026-11-09)

- Finding 1 fixed: `_formation_length` returns the path length $L/|\hat u_{0z}|$, used by both policies. Pinned by `test_formation_length_is_a_path_length_for_oblique_crossings` against the independent per-path mismatch rate $|\omega/\beta-\mathbf K\cdot\hat{\mathbf u}|$. The scene values equal the reviewer's right-hand column: HOPG 0.171 unchanged; gold default 1.835.
- Finding 2 adopted: the phase policy tests $\max(\|\mathbf K\|,1/L)\,D(L)$. Gold at 0.1 rad is 0.216, 0.509, 1.507, 1.173; HOPG stays 0. The acceptance step remains a heuristic.
- Findings 3 and 4 recorded in the module docstring and the physics page (rounding of the two distances; constant speed inside $L$).
- The corrected code was re-reviewed below (2026-11-09 re-review). Scored spectra are unaffected; only the validity bound and refusals change. Ledger status unchanged.


## Formation path length (re-review, 2026-11-09)

Scope: commit `79cdd48e7`, which changes `_formation_length`, `_crossing_parts`, `_phase_short`, the module docstring, and adds `test_formation_length_is_a_path_length_for_oblique_crossings`. The derivation was written before the diff was opened.

### Independent derivation

A charge moves with speed $\beta c$ along the unit direction $\hat{\mathbf u}$, so $\mathbf r=s\hat{\mathbf u}$ and $t=s/(\beta c)$ at path length $s$. Its field at frequency $\omega$ is compared with a free wave $\mathbf K=(\mathbf k_\parallel,\pm\lambda)$ of the side medium, with $\lambda=\sqrt{\varepsilon\omega^2-k_\parallel^2}$ complex (units $\hbar=c=1$, as in the code). The mismatch phase along the path is

$$
\omega t-\mathbf K\cdot\mathbf r=s\Bigl(\frac{\omega}{\beta}-\mathbf k_\parallel\cdot\hat{\mathbf u}_\parallel\mp\lambda\hat u_z\Bigr),
\qquad
L_s^{\pm}=\frac{1}{\bigl|\omega/\beta-\mathbf k_\parallel\cdot\hat{\mathbf u}_\parallel\mp\lambda\hat u_z\bigr|}.
$$

The modulus includes attenuation, as in the earlier reviews. The interface wave number of the particle field is $q=(\omega-\mathbf k_\parallel\cdot\mathbf v_\parallel)/v_z$ with $\mathbf v=\beta\hat{\mathbf u}$. Multiplying gives

$$
\hat u_z(q\mp\lambda)=\frac{\omega}{\beta}-\mathbf k_\parallel\cdot\hat{\mathbf u}_\parallel\mp\lambda\hat u_z
\quad\Rightarrow\quad
L_s^{\pm}=\frac{1}{|\hat u_z|\,|q\mp\lambda|}=\frac{L_z^{\pm}}{|\hat u_z|}.
$$

This holds identically for either sign of $\hat u_z$ and for either branch of $\lambda$, because the set $\{q-\lambda,\ q+\lambda\}$ is invariant under $\lambda\to-\lambda$. Taking $\max_\pm L_s^\pm$ commutes with the division. Any frame map that is a rotation or reflection preserving the $z$ axis up to sign (such as $\pi$ about $x$) leaves $|\hat u_z|=|v_z|/|\mathbf v|$, $\mathbf k_\parallel\cdot\mathbf v_\parallel$ and $q\,v_z$ unchanged when $\mathbf n$, $\mathbf v$ and $q$ are transformed together. Both sides of the interface share the crossing velocity, so the conversion is the same for the before and after sides.

### Comparison

- `_formation_length` returns `HBARC_EV_ANG / min(|q - lam|, |q + lam|) / (|v_z| / |v|)`, which equals $\max_\pm L_s^\pm$. `velocity` is `frame.velocity` $=\beta\hat{\mathbf u}\cdot$`_FLIP` (diag $(1,-1,-1)$ for backward emission), and `q` is built from the same flipped `v` and `k_par`, so $|v_z|/|\mathbf v|=|\hat u_z|$ in both hemispheres. **Match.**
- `_crossing_parts`: the default policy compares the path length with `_side_runs` (path lengths), and the phase policy feeds it to `_phase_short`. Both now use the path length. **Match.**
- `_phase_short`: `reach = clip(L - d_i)` uses path distances on both sides, consistent with $D(L_s)$. `rate = max(||K||, 1/L)` covers the phase term $\|\mathbf K\|D$ and the direction term $D/L$, in 1/Å matching `wavenumber`. At tolerance 0, every listed kink with $d_i<L$ still gives a positive product, so the default is reproduced. **Match** (the acceptance step remains a heuristic).
- Docstring (lines 52–73): consistent with the code. Line 53 ("departs … by more than that phase") predates the $D/L$ term, but the following sentences state it.

### Numerical evidence

Script `scratch/v2_formation_path.py` (not committed). It draws 2000 random cases of 6 crossings and 3 energies, with $\beta\in[0.02,0.999]$, isotropic $\hat{\mathbf u}$ (both signs of $\hat u_z$), complex $\varepsilon$ with $\operatorname{Re}\in[-30,10]$ and $\operatorname{Im}\in[0,30]$, and emission into both hemispheres. Each is evaluated in three frames: identity, $\operatorname{diag}(1,1,-1)$, and $\operatorname{diag}(-1,1,-1)$. The code's length times my per-path rate equals $\hbar c$ to a worst relative error of $3.5\times10^{-14}$. The focused tests (`-k "kink or formation"`) pass: 5 passed.

### Test independence

The reference $|\omega/\beta-\mathbf k_\parallel\cdot\hat{\mathbf u}_\parallel\mp\lambda\hat u_z|$ is a different expression from the code's $|q\mp\lambda|/|\hat u_z|$, so it pins the $1/|\hat u_z|$ factor. It is not fully independent: the test builds `q` with the same convention it feeds to the function, and it covers only $\hat u_z>0$, one side and the forward frame. The algebra above shows that the other cases are equivalent. Nothing pins the new $D/L$ term in `_phase_short`: no test is in the regime $1/L>\|\mathbf K\|$ with a sub-$L$ kink. Exact $v_z=0$ would give $0/0$, but a crossing cannot have $v_z=0$ (informational).

### Verdict

- **Claim**: `optical-planar-tr` — `optical_transition_validity.py::{_formation_length, _crossing_parts, _phase_short}` — own-field formation path length.
- **Filters**: units pass (Å, 1/Å, rad); limits pass (normal incidence unchanged; tolerance 0 equals the default); signs/conventions pass (both $\hat u_z$ signs, both $\lambda$ branches, both hemispheres).
- **Re-derivation**: `matches`. The earlier discrepancy (factor $1/|\hat u_{0z}|$) is resolved in both policies.
- **Verdict**: `rederived` for the corrected formation-validity code.
- **Suggested ledger change**: replace "The corrected validity code is not yet re-reviewed" with "re-reviewed `rederived` at `405d97a44` (path length $L_z/|\hat u_z|$ in both policies, both hemispheres; $D/L$ term not separately pinned)". Keep the status `anchored`. Optionally, add a slow-charge test of the $D/L$ term.
