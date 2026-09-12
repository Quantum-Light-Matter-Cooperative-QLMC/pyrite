# Electron Transport Physics Recommendations

## Scope

This document summarizes recommended upgrades to PyRITE's electron-transport physics for bulk and crystalline materials while **excluding explicit channeling physics**. The intended energy range is approximately

\[
1\text{–}2~\mathrm{keV}
\lesssim E
\lesssim
100~\mathrm{MeV},
\]

with particular emphasis on preserving the detailed electron trajectories required by coherent radiation calculations such as parametric X-ray radiation (PXR) and coherent bremsstrahlung (CBS).

The central design constraint is that PyRITE is not merely interested in endpoint transport observables such as range, dose, or exit angle. Its radiation solver consumes the intermediate electron trajectory itself. The transport engine must therefore preserve a physically meaningful sequence of approximately straight-line trajectory segments between direction-changing interactions.

```{important}
For PXR/CBS, the detailed trajectory is part of the radiation calculation.

A transport approximation that reproduces only the final angular distribution, projected range, or mean energy loss is not automatically sufficient if it alters the intermediate path and therefore the coherent radiation phase.
```

---

## Current transport architecture

The current transport model can be summarized approximately as

\[
\text{elastic collision}
\rightarrow
\text{straight segment}
\rightarrow
\text{elastic collision}
\rightarrow
\cdots
\]

with inelastic energy loss applied continuously or stochastically along the segments rather than represented as explicit microscopic inelastic interactions.

The currently implemented physics is broadly:

| Process | Current treatment |
| --- | --- |
| Elastic collision rate | Browning-type fit to Mott total cross sections |
| Elastic angular distribution | Screened Rutherford form calibrated to a Mott transport moment where data are available |
| Low-energy collisional stopping | Joy–Luo |
| Higher-energy collisional stopping | ICRU-37 / Berger–Seltzer style Bethe stopping |
| Energy-loss straggling | Urban/Bichsel-inspired compound-Poisson treatment |
| Hard inelastic collisions | Not explicitly generated |
| Secondary electrons | Not transported |
| Inner-shell ionization | Not explicitly generated as a transport event |
| Atomic relaxation | Not coupled to electron transport |
| Bremsstrahlung spectrum | Born-like electron–nucleus model with Elwert correction |
| Bremsstrahlung angle | Effectively isotropic |
| Bremsstrahlung feedback on trajectory | Radiation treated as post-processing rather than a transport event |

For the current thin-target, tens-to-hundreds-of-keV regime, this is a defensible primary-electron model. The most significant shortcomings are not the existence of the Joy–Luo/Bethe stopping splice itself, but rather the approximate elastic angular model, the lack of explicit hard inelastic events, and the present bremsstrahlung treatment.

---

## Recommended target architecture

The recommended long-term transport stack is

\[
\boxed{
\begin{aligned}
\text{Elastic scattering}
&\rightarrow
\text{full ELSEPA / NIST differential cross sections}
\\[1mm]
\text{Soft collisional loss}
&\rightarrow
\text{restricted GOS or dielectric-response model}
\\[1mm]
\text{Hard electron--electron collisions}
&\rightarrow
\text{explicit Møller / GOS events}
\\[1mm]
\text{Inner-shell ionization}
&\rightarrow
\text{Bote--Salvat}
\\[1mm]
\text{Atomic relaxation}
&\rightarrow
\text{EADL or equivalent}
\\[1mm]
\text{Bremsstrahlung}
&\rightarrow
\text{Seltzer--Berger}
\\[1mm]
\text{Mean stopping validation}
&\rightarrow
\text{SBETHE + ESTAR}
\end{aligned}
}
\]

The crucial architectural recommendation is to **adopt improved interaction physics without automatically adopting conventional condensed-history transport**.

---

## 1. Preserve explicit trajectory segments as the canonical transport representation

### Why this matters

A generic condensed-history transport scheme replaces many small physical interactions with one statistically equivalent effective step. For ordinary electron transport this is often an excellent approximation.

For PXR and CBS, however, the radiation amplitude depends on the intermediate trajectory:

\[
\mathcal A(\omega,\mathbf n)
\sim
\int
F[\mathbf r(t),\mathbf v(t)]
e^{i\Phi(t)}
\,dt.
\]

After discretization, this becomes approximately

\[
\mathcal A
\simeq
\sum_j
\mathcal A_j
\left(
\mathbf r_j,
\mathbf v_j,
L_j,
E_j
\right).
\]

Two trajectories may have the same entry state, exit state, and total angular variance while producing different coherent radiation amplitudes.

For example, the physical path

\[
\mathbf v_1
\rightarrow
\mathbf v_2
\rightarrow
\mathbf v_3
\rightarrow
\mathbf v_4
\]

is not generally equivalent, for coherent radiation, to a condensed-history replacement

\[
\mathbf v_1
\rightarrow
\mathbf v_4
\]

at one effective hinge.

### Recommendation

Retain an explicit event-by-event trajectory mode in which every sufficiently important direction-changing interaction creates a new physical segment.

The canonical segment boundaries should eventually include

\[
\boxed{
\text{elastic collisions}
+
\text{hard inelastic collisions}
+
\text{hard bremsstrahlung events}.
}
\]

Soft energy loss can remain condensed along those segments.

```{note}
This does not require reproducing the PENELOPE transport algorithm. PyRITE can adopt PENELOPE-quality interaction models while retaining its own event-by-event trajectory representation.
```

---

## 2. Replace the effective Mott surrogate with full ELSEPA differential cross sections

### Current limitation

The present "Mott" angular treatment is effectively a screened-Rutherford distribution

\[
\frac{d\sigma}{d\Omega}
\propto
\frac{1}{
\left(
1-\cos\theta+2\alpha
\right)^2
},
\]

with the screening parameter chosen so that a low-order angular transport moment agrees with available Mott data.

This preserves approximately the correct mean angular diffusion, but it does not preserve the full physical differential cross section

\[
\frac{d\sigma_{\mathrm{Mott}}}{d\Omega},
\]

including its large-angle tail and higher angular moments.

For an event-by-event transport engine, this is a significant unnecessary approximation.

### Recommended replacement

For the current \(1\text{–}300~\mathrm{keV}\) domain, use the full NIST/ELSEPA elastic data:

\[
\boxed{
\sigma_{\mathrm{el}}(E,Z),
\qquad
\frac{d\sigma_{\mathrm{el}}}{d\Omega}(E,Z,\theta).
}
\]

The free-flight distance remains

\[
s
=
-\lambda_{\mathrm{el}}\ln\xi,
\qquad
\lambda_{\mathrm{el}}
=
\frac{1}{
n\sigma_{\mathrm{el}}
}.
\]

The polar scattering angle should be sampled directly from the cumulative differential cross section, with a uniform azimuth for an isotropic amorphous-equivalent atomic environment.

For \(E>300~\mathrm{keV}\), generate or tabulate the equivalent extended ELSEPA data rather than reverting to a screened-Rutherford approximation.

### Priority

**Highest priority for current-domain trajectory accuracy.**

---

## 3. Replace the Joy–Luo / ICRU stopping splice with SBETHE

### Current model

The present collisional stopping treatment uses a material- or element-dependent crossover:

\[
S_{\mathrm{col}}
=
\begin{cases}
S_{\mathrm{JL}}, & E<E_{\mathrm{cross}},
\\
S_{\mathrm{ICRU37}}, & E>E_{\mathrm{cross}}.
\end{cases}
\]

This is a reasonable practical solution in the current energy range.

Joy–Luo regularizes the low-energy breakdown of the uncorrected Bethe expression, while the higher-energy branch uses a conventional relativistic stopping-power model.

### Recommended replacement

Use SBETHE as the primary mean collisional stopping model:

\[
\boxed{
S_{\mathrm{col}}(E,\mathrm{material})
=
S_{\mathrm{SBETHE}}.
}
\]

This removes the empirical low-energy splice and incorporates a more systematic shell correction and density-effect treatment.

Benefits include:

- smoother physics from the low-keV regime into relativistic energies;
- explicit shell corrections;
- explicit density-effect corrections;
- a single reference model from approximately \(1~\mathrm{keV}\) upward;
- better suitability for eventual multi-MeV transport.

### Role of ESTAR

NIST ESTAR should remain an important independent validation reference, especially above roughly tens of keV.

It should not be treated as the low-keV gold standard because the conventional ESTAR collisional stopping treatment is less complete there than a modern shell-corrected formulation.

### Priority

**Medium priority for the current thin-target domain; high priority before extending far above 300 keV.**

---

## 4. Separate mean stopping power from microscopic inelastic transport

A stopping-power model determines

\[
S(E)
=
-\left\langle\frac{dE}{dx}\right\rangle.
\]

It does not determine the microscopic transfer distribution

\[
\frac{d^2\sigma_{\mathrm{in}}}{dW\,dQ}.
\]

Those are distinct levels of modeling.

The current PyRITE architecture primarily uses the former. A general-purpose transport model should eventually support both.

### Recommended inelastic architecture

Partition energy transfers at a configurable threshold \(W_c\):

\[
W<W_c
\quad\Rightarrow\quad
\text{soft condensed loss},
\]

\[
W>W_c
\quad\Rightarrow\quad
\text{explicit hard inelastic event}.
\]

The corresponding stopping power is

\[
S(E)
=
n
\int
W
\frac{d\sigma(E)}{dW}
\,dW.
\]

### Soft inelastic interactions

Use a generalized oscillator strength or dielectric-response model for

- valence excitation,
- plasmon losses,
- soft ionization,
- material-dependent electronic response.

A PENELOPE-like generalized oscillator strength model is a strong general default.

For high-accuracy low-energy solid-state work, an optional material-specific dielectric backend based on

\[
L(q,\omega)
=
\operatorname{Im}
\left[
-\frac{1}{\epsilon(q,\omega)}
\right]
\]

would be a meaningful upgrade.

### Hard inelastic interactions

Generate explicit Møller-like electron–electron collisions above \(W_c\), producing

- a discrete primary-electron energy decrement;
- a physical recoil angle;
- a secondary electron when the transferred energy is above the tracking threshold.

These events should break the PXR/CBS trajectory into new segments.

### Priority

**High priority once secondary electrons, ionization cascades, or multi-MeV transport become first-class goals.**

For present thin-film primary-electron trajectory calculations, this is less urgent than improving elastic scattering.

---

## 5. Add explicit secondary-electron transport

The present architecture does not generate explicit \(\delta\) electrons.

A modern mixed inelastic model should eventually create secondary electrons above a configurable production threshold.

For a hard electron–electron collision,

\[
e^-(E)
\rightarrow
e^-(E-W)
+
e^-(W),
\]

both outgoing particles can be transported if their kinetic energy exceeds the configured cutoff.

This is required for quantitatively reliable modeling of

- electron cascades,
- local energy deposition,
- secondary-electron spectra,
- thick-target response,
- secondary-induced radiation.

It is less important if PyRITE remains narrowly focused on primary-electron trajectories through thin foils.

---

## 6. Treat inner-shell ionization separately from generic inelastic loss

A generic oscillator-strength model is not the preferred tool for precise K/L/M shell vacancy production near threshold.

Use Bote–Salvat cross sections for inner-shell ionization:

\[
\boxed{
\sigma_{nl}^{\mathrm{ion}}
(E,Z)
\rightarrow
\text{Bote--Salvat DWBA/PWBA}.
}
\]

These events should create a shell vacancy explicitly.

The vacancy can then feed an atomic-relaxation model based on EADL or another vetted relaxation database, producing

\[
\text{characteristic X rays}
+
\text{Auger electrons}.
\]

This provides a physically consistent route to electron-induced characteristic radiation rather than attaching XRF yields indirectly to mean stopping power.

### Priority

**High if characteristic radiation is a desired first-class observable; otherwise medium.**

---

## 7. Replace the present bremsstrahlung model with Seltzer–Berger

### Current limitations

The existing bremsstrahlung treatment is based on an approximate Born-like electron–nucleus spectrum with Elwert correction.

The main limitations are:

- incomplete screening treatment;
- approximate low-energy Coulomb physics;
- omission of electron–electron bremsstrahlung;
- simplified angular distribution;
- no recoil or energy-loss feedback into the transported electron.

### Recommended spectrum

Use Seltzer–Berger differential bremsstrahlung data:

\[
\boxed{
\frac{d\sigma_{\mathrm{brem}}}{dk}
=
\left(
\frac{d\sigma}{dk}
\right)_{\mathrm{nuclear}}
+
\left(
\frac{d\sigma}{dk}
\right)_{\mathrm{electron}}.
}
\]

This is the natural production model over the intended \(1~\mathrm{keV}\) to \(100~\mathrm{MeV}\) range.

The electron–electron contribution is especially relevant in low-\(Z\) materials because it does not scale identically to the dominant nuclear term.

### Recommended angular model

Do not assume isotropic emission when predicting detector-resolved quantities

\[
\frac{d^2N}{dE\,d\Omega}.
\]

Use a standard bremsstrahlung angular distribution such as a Koch–Motz 2BS/2BN-type model or another validated equivalent.

### Priority

**Very high.**

This is probably the largest current weakness in the radiation-background model.

---

## 8. Couple hard bremsstrahlung events back into transport

At sufficiently high energy, bremsstrahlung cannot remain a passive post-processing calculation.

A hard emitted photon must change the electron state:

\[
e^-(E,\mathbf p)
\rightarrow
e^-(E-k,\mathbf p')
+
\gamma(k,\mathbf k).
\]

This changes

- electron energy;
- electron momentum;
- subsequent stopping;
- subsequent elastic scattering;
- the trajectory seen by the PXR/CBS solver.

At tens of MeV in high-\(Z\) materials, and certainly by \(100~\mathrm{MeV}\), radiative losses can rival or dominate collisional stopping.

Therefore:

```{warning}
The current "transport first, generate bremsstrahlung afterward" architecture must not be declared valid up to \(100~\mathrm{MeV}\).

Before entering that regime, bremsstrahlung must become a coupled transport process.
```

A practical implementation can divide bremsstrahlung into

\[
k<k_c
\quad\Rightarrow\quad
\text{continuous soft radiative loss},
\]

\[
k>k_c
\quad\Rightarrow\quad
\text{explicit hard photon event}.
\]

Hard events create both a photon and a new electron trajectory segment.

---

## 9. Density effect becomes mandatory in the relativistic regime

The current omission of the density-effect correction is small enough to be acceptable over much of the present \(\lesssim300~\mathrm{keV}\) domain.

That argument does not extend indefinitely.

At relativistic energies, dielectric polarization suppresses long-range contributions to collisional stopping, and the density-effect correction becomes an essential component of the model.

Using SBETHE or an equivalent modern stopping-power treatment resolves this systematically.

---

## 10. Retain event-by-event scattering as the high-accuracy reference mode

At high energies or in thick materials, explicitly simulating every tiny elastic deflection may become computationally expensive.

Do not solve this by replacing the canonical model outright.

Instead, maintain two transport modes.

### Detailed mode

Every physical elastic collision is sampled from the full differential cross section.

This remains the reference trajectory model:

\[
\text{collision}
\rightarrow
\text{straight segment}
\rightarrow
\text{collision}.
\]

It may be computationally expensive, but it is valuable for validation.

### Radiation-aware mixed mode

If performance eventually demands condensation, divide the elastic cross section at an angular threshold \(\theta_c\):

\[
\frac{d\sigma}{d\Omega}
=
\left(
\frac{d\sigma}{d\Omega}
\right)_{\theta<\theta_c}
+
\left(
\frac{d\sigma}{d\Omega}
\right)_{\theta>\theta_c}.
\]

Then

\[
\theta>\theta_c
\quad\Rightarrow\quad
\text{explicit physical collision},
\]

while

\[
\theta<\theta_c
\quad\Rightarrow\quad
\text{stochastic angular diffusion}.
\]

The soft-scattering trajectory must still be sampled finely enough for coherent-radiation convergence.

### Radiation-aware convergence criterion

The appropriate step-size criterion is not merely

\[
\langle\theta^2\rangle_{\mathrm{step}}\ll1.
\]

The approximation must also preserve the radiation phase.

Require approximately

\[
|\Delta\Phi_{\mathrm{step}}|
\ll1,
\]

and keep the trajectory substep shorter than the relevant formation/coherence scale,

\[
L_{\mathrm{step}}
\ll
L_f,
\]

where appropriate.

The correct validation observable is therefore not only the electron exit distribution but also the converged PXR/CBS spectrum.

---

## 11. Validation strategy

Each physics component should be validated against an independent reference appropriate to that layer.

| Quantity | Primary validation reference |
| --- | --- |
| Elastic total cross section | ELSEPA / NIST SRD 64 |
| Elastic differential cross section | ELSEPA / NIST SRD 64 |
| Elastic angular moments | Derived independently from the full DCS |
| Mean collisional stopping | SBETHE |
| Conventional stopping reference | NIST ESTAR |
| Low-energy empirical regression | Joy–Luo |
| Inelastic mean free path / loss spectrum | PENELOPE GOS and/or dielectric data |
| Inner-shell ionization | Bote–Salvat / NIST SRD 164 |
| Atomic relaxation | EADL or equivalent |
| Bremsstrahlung spectrum | Seltzer–Berger |
| Bremsstrahlung angular distribution | validated 2BS/2BN or equivalent |
| Full-track transport | PENELOPE / Geant4 comparison where equivalent settings exist |
| Coherent radiation | internal detailed-vs-approximate trajectory convergence |

### Important distinction

Matching stopping power does not validate microscopic event distributions.

It is entirely possible for a model to satisfy

\[
S_{\mathrm{MC}}
\approx
S_{\mathrm{SBETHE}}
\]

while having incorrect

\[
\lambda_{\mathrm{in}},
\qquad
P(W),
\qquad
P(\theta),
\qquad
\sigma_K,
\qquad
\sigma_L.
\]

Validation should therefore be process-specific.

---

## Recommended implementation order

### Phase 1 — improve the current \(1\text{–}300~\mathrm{keV}\) model

1. **Replace the effective screened-Rutherford/Mott surrogate with the full ELSEPA differential elastic cross section.**
2. **Replace the current bremsstrahlung spectrum with Seltzer–Berger.**
3. **Add a physically justified bremsstrahlung angular distribution.**
4. **Replace the Joy–Luo/ICRU stopping splice with SBETHE.**
5. Revalidate the current PXR/CBS spectra after each change.

These changes improve the current physics without forcing a redesign of the trajectory engine.

### Phase 2 — add microscopic inelastic transport

1. Introduce a soft/hard energy-transfer cutoff \(W_c\).
2. Use GOS/dielectric physics for soft losses.
3. Generate explicit Møller/GOS hard collisions.
4. Transport secondary electrons above a configurable threshold.
5. Add Bote–Salvat inner-shell ionization.
6. Add atomic relaxation.

At this point, physical trajectory segments should be bounded by both elastic and hard inelastic direction-changing events.

### Phase 3 — extend safely into the multi-MeV regime

Before declaring multi-MeV or \(100~\mathrm{MeV}\) validity:

1. include density-effect corrections;
2. extend elastic scattering beyond the NIST 300-keV tables using ELSEPA or another vetted source;
3. make hard bremsstrahlung an explicit transport event;
4. include radiative energy loss and recoil;
5. transport emitted photons when secondary photon interactions matter;
6. consider a radiation-aware mixed elastic mode for performance;
7. validate radiation spectra against the fully detailed transport mode.

---

## Recommended production/reference stack

The proposed long-term division of responsibilities is

\[
\boxed{
\begin{aligned}
\textbf{Elastic trajectory physics}
&:
\text{ELSEPA}
\\[1mm]
\textbf{Soft inelastic response}
&:
\text{PENELOPE-like GOS}
\\[1mm]
\textbf{Optional low-energy solid-state response}
&:
\epsilon(q,\omega)\text{ / Penn-like dielectric model}
\\[1mm]
\textbf{Hard electron--electron collisions}
&:
\text{Møller / GOS event sampling}
\\[1mm]
\textbf{Inner-shell vacancy production}
&:
\text{Bote--Salvat}
\\[1mm]
\textbf{Atomic relaxation}
&:
\text{EADL or equivalent}
\\[1mm]
\textbf{Bremsstrahlung spectrum}
&:
\text{Seltzer--Berger}
\\[1mm]
\textbf{Bremsstrahlung angle}
&:
\text{2BS/2BN or equivalent}
\\[1mm]
\textbf{Mean collisional stopping}
&:
\text{SBETHE}
\\[1mm]
\textbf{Independent stopping validation}
&:
\text{ESTAR}
\\[1mm]
\textbf{Independent atomic-data validation}
&:
\text{EEDL / EPICS}.
\end{aligned}
}
\]

---

## Final recommendation

PyRITE's present event-by-event trajectory architecture should be preserved.

The design is unusually well suited to PXR/CBS because the radiation solver depends on the intermediate trajectory rather than only on macroscopic transport observables.

The best near-term strategy is therefore **not** to replace the current engine with a conventional condensed-history algorithm. Instead:

\[
\boxed{
\text{preserve the trajectory representation}
+
\text{upgrade the interaction physics}.
}
\]

The most important immediate changes are

1. full ELSEPA differential elastic scattering;
2. Seltzer–Berger bremsstrahlung;
3. a physical bremsstrahlung angular distribution;
4. SBETHE stopping.

Only after those are established should the code move toward explicit hard inelastic collisions, secondaries, shell ionization, and coupled electron–photon shower transport.

For any later mixed/condensed implementation, the acceptance criterion should be **convergence of the calculated PXR/CBS radiation**, not merely agreement in electron range, exit angle, or stopping power.
