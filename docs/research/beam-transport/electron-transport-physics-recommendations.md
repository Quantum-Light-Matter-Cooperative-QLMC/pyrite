# Electron Transport Physics Recommendations

## Scope

This document summarizes recommended upgrades to PyRITE's electron-transport and
incoherent-radiation physics for bulk and crystalline materials while
**excluding explicit channeling physics**. The intended eventual energy range is
approximately

$$
1\text{–}2~\mathrm{keV}
\lesssim E
\lesssim
100~\mathrm{MeV},
$$

with particular emphasis on preserving the detailed electron trajectories
required by coherent radiation calculations such as parametric X-ray radiation
(PXR) and coherent bremsstrahlung (CBS).

The central design constraint is that PyRITE is not interested only in endpoint
transport observables such as range, dose, or exit angle. Its coherent-radiation
solver consumes the intermediate electron trajectory itself. The transport
engine must therefore preserve a physically meaningful sequence of
approximately straight-line trajectory segments between direction-changing
interactions.

```{important}
For PXR/CBS, the detailed trajectory is part of the radiation calculation.

A transport approximation that reproduces only the final angular distribution,
projected range, or mean energy loss is not automatically sufficient if it
alters the intermediate path and therefore the coherent radiation phase.
```

This document incorporates the newer EEDL-based bremsstrahlung and
characteristic-radiation work. Those additions materially change the previous
recommendations: the bremsstrahlung **energy-spectrum** model is no longer a
major gap, and EEDL shell-ionization data are now a defensible production
baseline for characteristic X-ray generation.

---

## Current transport and radiation architecture

The primary-electron transport can still be summarized approximately as

$$
\text{elastic collision}
\rightarrow
\text{straight segment}
\rightarrow
\text{elastic collision}
\rightarrow
\cdots,
$$

with collisional energy loss applied continuously or stochastically along the
segments rather than represented as explicit microscopic inelastic events.
Numerical energy-control substeps may subdivide a physical flight without
introducing a new physical deflection.

Radiation is then scored from those transported segments. The newer EEDL work
adds substantially more microscopic atomic data to the radiation estimators,
but it does **not** yet turn shell ionization or bremsstrahlung into discrete
transport events.

The implemented physics is broadly:

| Process | Current treatment |
| --- | --- |
| Elastic collision rate | Browning-type fit to Mott total cross sections |
| Elastic angular distribution | Screened-Rutherford form calibrated to a Mott transport moment where NIST data are available; fallback screening otherwise |
| Low-energy collisional stopping | Joy–Luo |
| Higher-energy collisional stopping | ICRU-37 / Berger–Seltzer-style relativistic Bethe stopping |
| Density effect | Quantified but omitted in production transport |
| Energy-loss straggling | Unrestricted Urban/Bichsel-inspired compound-Poisson treatment |
| Hard inelastic collisions | Not explicitly generated |
| Secondary electrons | Not transported |
| Inner-shell vacancy production for X-ray scoring | EEDL shell-resolved electron-impact ionization cross sections, `MF=23`, `MT=534`–`572` |
| Characteristic relaxation | Direct-vacancy fluorescence using xraydb yields, branching intensities, line energies, and level widths |
| Characteristic line shape | Analytically bin-integrated natural Lorentzian |
| Bremsstrahlung energy spectrum | EEDL `MF=23/MT=527` total cross section × `MF=26/MT=527` photon-energy distribution |
| Legacy bremsstrahlung fallback | Analytic Born/Bethe–Heitler-like model with Elwert correction |
| Bremsstrahlung angle | Isotropic EEDL/ENDF representation, $1/(4\pi)$ |
| Bremsstrahlung feedback on electron | None; radiation is scored from the pre-existing trajectory |
| Photon escape | Beer–Lambert attenuation through the implemented sample geometry |

For the current thin-target, tens-to-hundreds-of-keV regime, this is a
substantially stronger model than the previous baseline. In particular, the
EEDL bremsstrahlung spectrum is already descended from the Seltzer–Berger
bremsstrahlung calculations and should not be replaced merely to obtain
Seltzer–Berger physics under another interface.

The most important remaining gaps are now:

1. the approximate elastic differential-scattering model seen by the trajectory;
2. isotropic rather than physical bremsstrahlung angular distributions;
3. the lack of microscopic hard inelastic events and secondary-electron transport;
4. incomplete atomic-relaxation cascades after shell vacancies;
5. the Joy–Luo / ICRU stopping splice and omitted relativistic density effect;
6. lack of recoil and energy feedback from hard bremsstrahlung events at high energy.

---

## Recommended target architecture

The recommended long-term division of responsibilities is

$$
\boxed{
\begin{aligned}
\text{Elastic trajectory physics}
&\rightarrow
\text{full ELSEPA / NIST differential cross sections}
\\[1mm]
\text{Mean collisional stopping}
&\rightarrow
\text{SBETHE or equivalent corrected Bethe model}
\\[1mm]
\text{Soft microscopic inelastic response}
&\rightarrow
\text{restricted GOS or dielectric-response model}
\\[1mm]
\text{Hard electron--electron collisions}
&\rightarrow
\text{explicit Møller / GOS events}
\\[1mm]
\text{Inner-shell vacancy production}
&\rightarrow
\text{EEDL production baseline; Bote--Salvat option/reference}
\\[1mm]
\text{Atomic relaxation}
&\rightarrow
\text{full radiative + Auger/Coster--Kronig cascade}
\\[1mm]
\text{Bremsstrahlung energy spectrum}
&\rightarrow
\text{current EEDL / Seltzer--Berger-lineage data}
\\[1mm]
\text{Bremsstrahlung angle}
&\rightarrow
\text{partial-wave-derived or 2BN/2BS-type model}
\\[1mm]
\text{Stopping-power validation}
&\rightarrow
\text{SBETHE + ESTAR}
\end{aligned}
}
$$

The crucial architectural recommendation remains to **adopt improved interaction
physics without automatically adopting conventional condensed-history
trajectory generation**.

---

## 1. Preserve explicit trajectory segments as the canonical representation

### Coherent-radiation reason for explicit trajectories

A generic condensed-history transport scheme replaces many small physical
interactions with one statistically equivalent effective step. For ordinary
electron transport this can be an excellent approximation.

For PXR and CBS, however, the radiation amplitude depends on the intermediate
trajectory:

$$
\mathcal A(\omega,\mathbf n)
\sim
\int
F[\mathbf r(t),\mathbf v(t)]
e^{i\Phi(t)}
\,dt.
$$

After discretization, this is approximately

$$
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
$$

Two histories may have the same entry state, exit state, and total angular
variance while producing different coherent radiation amplitudes. For example,
the physical path

$$
\mathbf v_1
\rightarrow
\mathbf v_2
\rightarrow
\mathbf v_3
\rightarrow
\mathbf v_4
$$

is not generally equivalent, for coherent radiation, to a condensed-history
replacement

$$
\mathbf v_1
\rightarrow
\mathbf v_4
$$

at one effective hinge.

### Trajectory-preservation recommendation

Retain an explicit event-by-event trajectory mode in which every sufficiently
important direction-changing interaction creates a new physical segment.

The canonical physical segment boundaries should eventually include

$$
\boxed{
\text{elastic collisions}
+
\text{hard inelastic collisions}
+
\text{hard bremsstrahlung events}.
}
$$

Soft energy loss can remain condensed along those segments.

```{note}
This does not require reproducing the PENELOPE transport algorithm. PyRITE can
adopt PENELOPE-quality interaction models while retaining its own event-by-event
trajectory representation.
```

Numerical substeps used to integrate energy loss should continue to be treated
as quadrature nodes inside a physical flight, not as independent collisions or
sources of decoherence.

---

## 2. Upgrade elastic scattering to full ELSEPA differential cross sections

### Existing elastic approximation

The present angular treatment is effectively a screened-Rutherford distribution
of the form

$$
\frac{d\sigma}{d\Omega}
\propto
\frac{1}{
\left(
1-\cos\theta+2\alpha
\right)^2
},
$$

with the screening parameter chosen, where suitable tabulated data exist, so
that a low-order Mott transport moment is reproduced.

This approximately preserves the mean angular diffusion but does not preserve
the complete physical differential cross section

$$
\frac{d\sigma_{\mathrm{Mott}}}{d\Omega},
$$

including its large-angle tail, diffraction structure of the atomic potential,
and higher angular moments. Elements without the required transport tables fall
back to the approximate screening model entirely.

For a code that explicitly resolves individual elastic collisions, this is an
unnecessary loss of information.

### ELSEPA replacement strategy

For the current $1\text{–}300~\mathrm{keV}$ domain, use the full NIST/ELSEPA
elastic data:

$$
\boxed{
\sigma_{\mathrm{el}}(E,Z),
\qquad
\frac{d\sigma_{\mathrm{el}}}{d\Omega}(E,Z,\theta).
}
$$

The free-flight distance remains

$$
s
=
-\lambda_{\mathrm{el}}\ln\xi,
\qquad
\lambda_{\mathrm{el}}
=
\frac{1}{n\sigma_{\mathrm{el}}}.
$$

The polar scattering angle should be sampled directly from the cumulative
ELSEPA differential cross section, with a uniform azimuth for the present
orientation-averaged, non-channeling transport model.

For $E>300~\mathrm{keV}$, generate or tabulate the equivalent extended ELSEPA
data rather than reverting to a screened-Rutherford approximation.

### Elastic-scattering priority

**Highest priority for improving the present trajectory physics.**

This change directly improves the trajectory consumed by PXR and CBS and is
more important to the present coherent-radiation use case than adding a fully
microscopic treatment of every soft inelastic collision.

---

## 3. Replace the Joy–Luo / ICRU stopping splice with SBETHE

### Existing stopping model

The present collisional stopping treatment uses a material- or element-dependent
crossover:

$$
S_{\mathrm{col}}
=
\begin{cases}
S_{\mathrm{JL}}, & E<E_{\mathrm{cross}},
\\
S_{\mathrm{ICRU37}}, & E>E_{\mathrm{cross}}.
\end{cases}
$$

This is a reasonable practical solution in the current energy range. Joy–Luo
regularizes the low-energy failure of an uncorrected Bethe expression, while the
higher-energy branch uses a relativistic Berger–Seltzer/ICRU-style collision
stopping law.

The present documentation also explicitly quantifies the cost of omitting the
density-effect correction over the current production range, rather than
assuming it is negligible.

### SBETHE replacement strategy

Use SBETHE, or an equivalent modern corrected-Bethe implementation, as the
primary mean collisional stopping model:

$$
\boxed{
S_{\mathrm{col}}(E,\mathrm{material})
=
S_{\mathrm{SBETHE}}.
}
$$

This would remove the empirical low-energy splice while adding a systematic
shell correction and density-effect treatment.

Benefits include:

- smoother physics from the low-keV regime into relativistic energies;
- explicit shell corrections rather than relying on Joy–Luo as the low-energy
  surrogate;
- explicit density-effect corrections;
- a single stopping model from approximately $1~\mathrm{keV}$ upward;
- better suitability for eventual multi-MeV transport.

### ESTAR as an independent stopping reference

NIST ESTAR should remain an important independent validation reference,
especially above roughly tens of keV. It should not be treated as the low-keV
gold standard because its traditional collision-stopping formulation is less
complete there than a modern shell-corrected treatment.

### Stopping-power priority

**Medium priority in the present thin-target regime; high priority before a
multi-MeV validity claim.**

The existing stopping treatment is not currently the dominant physics weakness
for thin PXR/CBS targets.

---

## 4. Keep the EEDL bremsstrahlung energy spectrum

### What the new EEDL backend already accomplishes

The default bremsstrahlung spectrum is now constructed from

$$
\frac{d\sigma_Z}{dk}(T,k)
=
\sigma_Z^{23,527}(T)
P_Z^{26,527}(k\mid T),
$$

where `MF=23/MT=527` supplies the total electro-atomic bremsstrahlung cross
section and `MF=26/MT=527` supplies the normalized secondary-photon energy
distribution.

This is the correct decomposition for a track-length estimator: the total
cross section fixes the event probability per unit path, and the normalized
secondary distribution fixes the conditional photon-energy spectrum.

The EEDL bremsstrahlung data were derived from Seltzer–Berger photon-yield
spectra. Therefore the new production backend is already, in practical terms,
a Seltzer–Berger-lineage bremsstrahlung energy model.

```{important}
Do **not** replace the new EEDL bremsstrahlung spectrum merely to obtain
"Seltzer–Berger" physics. That would largely duplicate the physics already
present in EEDL.
```

The legacy analytic Born/Elwert backend remains useful for reproducibility,
fallback outside EEDL coverage, and regression testing, but it should not be
the preferred production spectrum when evaluated data are available.

### Bremsstrahlung spectrum validation

Validate the EEDL parser and interpolation independently by comparing

$$
\frac{d\sigma}{dk},
$$

the integrated total cross section, and the radiative stopping moment against a
direct Seltzer–Berger/PENELOPE implementation at representative values of
$Z$, $T$, and $k/T$.

This is primarily a **software/data-validation task**, not a recommendation to
change the production model.

### Bremsstrahlung-spectrum priority

**No major production-model replacement required.**

The new EEDL work closes most of the previously identified bremsstrahlung
energy-spectrum gap.

---

## 5. Replace isotropic bremsstrahlung emission with a physical angular model

### Limitation of the EEDL angular representation

The EEDL/ENDF photon subsection used by the current code supplies the photon
energy distribution without a detailed physical angular distribution. PyRITE
therefore retains an isotropic factor

$$
\frac{dP}{d\Omega}
=
\frac{1}{4\pi}.
$$

This is an ENDF representation choice, not a statement that physical
electron bremsstrahlung is isotropic.

That distinction matters because PyRITE predicts detector-direction-resolved
quantities such as

$$
\frac{d^2N}{dE\,d\Omega}.
$$

The energy spectrum can therefore be accurate while the observed directional
background remains biased.

### Recommended bremsstrahlung angular model

Retain the EEDL energy spectrum and replace only the angular factor with a
validated distribution depending on at least

$$
Z,
\qquad
E,
\qquad
k/E,
\qquad
\theta_\gamma.
$$

Strong candidates are:

- the modern PENELOPE analytical shape function fitted to partial-wave
  bremsstrahlung calculations;
- a Koch–Motz 2BN/2BS-type angular model where appropriate.

The spectrum estimator then becomes schematically

$$
\frac{d^2N}{dk\,d\Omega}
=
nL
\frac{d\sigma}{dk}
P_\Omega(\theta_\gamma;Z,E,k/E)
T_{\mathrm{abs}}.
$$

For the existing GPU architecture, this can remain a track-length estimator:
the main change is replacing the constant $1/(4\pi)$ by an angle-dependent
weight for each segment, detector direction, and photon energy.

### Bremsstrahlung-angle priority

**Very high priority for detector-resolved background accuracy.**

With the EEDL energy spectrum now in place, this is the clearest remaining
bremsstrahlung-model deficiency in the current energy range.

---

## 6. Retain EEDL shell ionization as a production baseline

### Current characteristic-radiation model

The new characteristic-radiation estimator uses EEDL shell-resolved
ionization cross sections for direct vacancy production and xraydb atomic data
for radiative relaxation.

For a segment $j$, element $a$, initially ionized subshell $i$, and line
$\ell$, the expected direct line yield is

$$
Y_{jai\ell}
=
\frac{
 n_a L_j
 \sigma_{ai}(T_j)
 \omega_{ai}
 I_{ai\ell}
}{4\pi N_e}
\exp[-\tau_j(E_{ai\ell})].
$$

This is a sound track-length estimator for **directly produced vacancies
followed by direct radiative relaxation**.

### EEDL versus Bote–Salvat for vacancy production

The earlier recommendation to replace EEDL shell-ionization cross sections
unconditionally with Bote–Salvat was too strong.

Bote–Salvat has a cleaner modern theoretical basis, using relativistic DWBA
near threshold and PWBA at larger overvoltage, and is an excellent reference or
optional production backend. However, EEDL shell-ionization cross sections are
still a defensible baseline, particularly for K-shell work, and existing
experimental comparisons do not establish a universal large accuracy advantage
for Bote–Salvat over EEDL in every shell and element.

The recommended architecture is therefore

$$
\boxed{
\text{EEDL shell ionization as a supported production model}
+
\text{Bote--Salvat as an optional/reference model}.
}
$$

Model disagreement can itself be reported as a useful systematic-uncertainty
indicator.

### Shell-ionization priority

**Do not block current characteristic-radiation work on replacing EEDL.**

Adding Bote–Salvat remains worthwhile, especially for extending the model to
higher energies and for independent validation, but it is no longer a
first-order correctness fix.

---

## 7. Upgrade direct-vacancy fluorescence to a full atomic-relaxation cascade

### Present relaxation scope

The current model applies xraydb fluorescence yields and conditional line
intensities to a directly created shell vacancy. It deliberately does not
propagate the subsequent vacancy cascade.

Missing processes include:

- Auger decay;
- Coster–Kronig redistribution among subshells;
- daughter vacancies created by nonradiative transitions;
- secondary fluorescence from those daughter vacancies;
- Auger-electron transport;
- multiple-vacancy shifts and broadening.

The current model is therefore best described as a **direct-vacancy
characteristic-X-ray estimator**, not a complete atomic-relaxation model.

### Recommended cascade model

After creating a vacancy in shell $i$, sample or expectation-propagate all
allowed radiative and nonradiative transitions until the vacancy population has
migrated to shells below the configured relaxation cutoff:

$$
\text{primary shell vacancy}
\rightarrow
\begin{cases}
\gamma_{\mathrm{X-ray}} + \text{daughter vacancy},\\
 e^-_{\mathrm{Auger/CK}} + \text{daughter vacancies}.
\end{cases}
$$

EADL-style transition probabilities are the natural reference source. xraydb
can remain useful for line energies, natural widths, and spectroscopy-facing
line metadata even if the cascade topology and probabilities come from another
source.

A deterministic expectation-value cascade may be preferable to stochastic
sampling when characteristic radiation is being scored as a low-noise spectrum
rather than coupled back into transport.

### Relaxation-cascade priority

**Medium-to-high priority if characteristic X-ray spectroscopy is a first-class
observable.**

The omission matters more for L/M-shell spectra and high-$Z$ materials than for
first-generation K-line yields.

---

## 8. Refine the characteristic line-shape and low-energy cutoff conventions

### Finite-grid Lorentzian normalization

The current implementation correctly integrates each natural Lorentzian across
energy-bin boundaries:

$$
q_{\ell b}
=
\frac{1}{\pi}
\left[
\tan^{-1}\!\left(
\frac{2(E_b^+-E_\ell)}{\Gamma_\ell}
\right)
-
\tan^{-1}\!\left(
\frac{2(E_b^--E_\ell)}{\Gamma_\ell}
\right)
\right].
$$

This is preferable to point-sampling narrow lines at bin centres.

The current code then renormalizes the bin weights over the requested grid when
the line centre lies inside that grid. That preserves the historical integrated
line yield, but it is not the most literal representation of a physical
$dN/dE$: Lorentzian probability lying outside the requested energy window is
artificially redistributed inside the window.

For a physically normalized spectrum, prefer

$$
\sum_{b\in\text{requested grid}}q_{\ell b}
\le 1
$$

when the spectral window truncates the line tails. A line centred outside the
grid can likewise contribute a small physical tail inside the grid.

If exact grid-independent line-yield conservation is required for a separate
diagnostic, store that integrated yield separately rather than enforcing it by
renormalizing the displayed spectral window.

### Characteristic transport cutoff

The current characteristic estimator uses the low-background electron
population with a nominal $\sim1~\mathrm{keV}$ transport cutoff. That is
reasonable for the present tens-to-hundreds-of-keV use case but becomes a
limitation if PyRITE claims accurate characteristic production down to the
bottom of the proposed $1\text{–}2~\mathrm{keV}$ incident-energy range.

For low-binding-energy shells, an electron can continue producing vacancies
below $1~\mathrm{keV}$. The eventual characteristic-radiation cutoff should
therefore be tied to the lowest ionization threshold being scored, plus the
validity limit of the transport model, rather than being treated as a universal
$1~\mathrm{keV}$ physics boundary.

### Characteristic-spectrum numerical priority

**Low-to-medium priority for present hard-X-ray applications; important before
claiming precision low-keV characteristic spectra.**

---

## 9. Separate mean stopping power from microscopic inelastic transport

A stopping-power model determines

$$
S(E)
=
-\left\langle
\frac{dE}{dx}
\right\rangle.
$$

It does not determine the microscopic transfer distribution

$$
\frac{d^2\sigma_{\mathrm{in}}}{dW\,dQ}.
$$

Those are distinct levels of modeling.

The new EEDL shell-ionization estimator does **not** change this fact. PyRITE can
now calculate the expected number of shell vacancies created along a segment,
but it does not yet sample those vacancies as transport events, transfer their
energy and momentum to the primary electron, or launch the ejected electron.

Likewise, the EEDL bremsstrahlung estimator scores photons from an existing
trajectory without changing that trajectory.

### Recommended soft/hard inelastic partition

Partition energy transfers at a configurable threshold $W_c$:

$$
W<W_c
\quad\Rightarrow\quad
\text{soft condensed loss},
$$

$$
W>W_c
\quad\Rightarrow\quad
\text{explicit hard inelastic event}.
$$

The corresponding stopping power is

$$
S(E)
=
n
\int
W
\frac{d\sigma(E)}{dW}
\,dW.
$$

### Soft inelastic response model

Use a generalized oscillator strength (GOS) or dielectric-response model for

- valence excitation;
- plasmon losses;
- soft ionization;
- material-dependent electronic response.

A PENELOPE-like GOS model is a strong general default. It describes the target
through an energy- and momentum-transfer response rather than through a single
mean loss rate.

For high-accuracy low-energy solid-state work, an optional material-specific
dielectric backend based on

$$
L(q,\omega)
=
\operatorname{Im}
\left[
-\frac{1}{\epsilon(q,\omega)}
\right]
$$

would be a meaningful additional model.

### Hard inelastic collision model

Generate explicit Møller-like electron–electron collisions above $W_c$,
producing

- a discrete primary-electron energy decrement;
- a physical recoil angle;
- a secondary electron when the transferred energy exceeds the tracking
  threshold.

These events should break the PXR/CBS trajectory into new physical segments.

### Microscopic-inelastic priority

**High priority once secondary electrons, ionization cascades, thick targets,
or multi-MeV transport become first-class goals.**

For present thin-film primary-electron PXR/CBS calculations, this remains less
urgent than full elastic differential scattering.

---

## 10. Add explicit secondary-electron transport when required

The current architecture does not generate explicit $\delta$ electrons.

A mixed inelastic model should eventually create secondary electrons above a
configurable production threshold. For a hard electron–electron collision,

$$
e^-(E)
\rightarrow
e^-(E-W)
+
e^-(W),
$$

both outgoing particles can be transported if their kinetic energy exceeds the
configured cutoff.

This becomes necessary for quantitatively reliable modeling of

- electron cascades;
- local energy deposition;
- secondary-electron spectra;
- thick-target response;
- secondary-induced characteristic radiation and bremsstrahlung.

It remains less important if PyRITE is intentionally restricted to primary
trajectories through sufficiently thin foils.

---

## 11. Couple hard bremsstrahlung events back into transport at high energy

The new EEDL spectrum substantially improves the **radiation yield model**, but
it does not change the electron state when a photon is scored.

At sufficiently high energy, that approximation fails. A hard emitted photon
must change the electron state:

$$
e^-(E,\mathbf p)
\rightarrow
e^-(E-k,\mathbf p')
+
\gamma(k,\mathbf k).
$$

This changes

- electron energy;
- electron momentum;
- subsequent stopping;
- subsequent elastic scattering;
- the trajectory seen by the PXR/CBS solver.

At tens of MeV in high-$Z$ materials, and certainly before a general
$100~\mathrm{MeV}$ validity claim, radiative losses can rival or dominate
collisional stopping.

```{warning}
The current "transport first, score bremsstrahlung afterward" architecture
should not be declared generally valid to $100~\mathrm{MeV}$ merely because the
EEDL photon spectrum itself extends to high energy.

Before entering the radiation-dominated regime, bremsstrahlung must become a
coupled transport process.
```

A practical implementation can divide bremsstrahlung into

$$
k<k_c
\quad\Rightarrow\quad
\text{continuous soft radiative loss},
$$

$$
k>k_c
\quad\Rightarrow\quad
\text{explicit hard photon event}.
$$

Hard events should create a photon and a new electron trajectory segment.

---

## 12. Make the density effect mandatory for relativistic extension

The present omission of the density-effect correction is explicitly quantified
and small enough to be acceptable over much of the current
$\lesssim300~\mathrm{keV}$ domain.

That result should not be extrapolated to much higher energy. At relativistic
energies, dielectric polarization suppresses long-range contributions to
collisional stopping, and the density-effect correction becomes an essential
part of the model.

Using SBETHE or an equivalent modern material-aware stopping treatment resolves
this systematically.

---

## 13. Retain event-by-event scattering as the high-accuracy reference mode

At high energies or in thick materials, explicitly simulating every tiny
elastic deflection may eventually become computationally expensive.

Do not solve this by replacing the canonical detailed model outright. Maintain
a detailed reference mode and, only when needed, introduce a radiation-aware
mixed mode.

### Fully detailed reference mode

Every physical elastic collision is sampled from the full differential cross
section. Explicit hard inelastic and hard bremsstrahlung events also create
segment boundaries.

The trajectory remains

$$
\text{physical event}
\rightarrow
\text{straight segment}
\rightarrow
\text{physical event}.
$$

This may be expensive, but it provides the reference against which faster
approximations can be judged.

### Radiation-aware mixed elastic mode

If performance eventually demands condensation, divide the elastic cross
section at an angular threshold $\theta_c$:

$$
\frac{d\sigma}{d\Omega}
=
\left(
\frac{d\sigma}{d\Omega}
\right)_{\theta<\theta_c}
+
\left(
\frac{d\sigma}{d\Omega}
\right)_{\theta>\theta_c}.
$$

Then

$$
\theta>\theta_c
\quad\Rightarrow\quad
\text{explicit physical collision},
$$

while

$$
\theta<\theta_c
\quad\Rightarrow\quad
\text{stochastic angular diffusion}.
$$

The soft-scattering trajectory must still be sampled finely enough for
coherent-radiation convergence.

### Radiation-aware convergence criterion

The appropriate step-size criterion is not merely

$$
\langle\theta^2\rangle_{\mathrm{step}}
\ll 1.
$$

The approximation must also preserve the radiation phase. Require approximately

$$
|\Delta\Phi_{\mathrm{step}}|
\ll 1,
$$

and keep the trajectory substep shorter than the relevant formation or
coherence scale,

$$
L_{\mathrm{step}}
\ll
L_f,
$$

where applicable.

The correct validation observable is therefore not only the electron exit
distribution but also the converged PXR/CBS spectrum.

---

## 14. Validation strategy

Each physics component should be validated against an independent reference
appropriate to that layer.

| Quantity | Primary validation reference |
| --- | --- |
| Elastic total cross section | ELSEPA / NIST SRD 64 |
| Elastic differential cross section | ELSEPA / NIST SRD 64 |
| Elastic angular moments | Derived independently from the full DCS |
| Mean collisional stopping | SBETHE |
| Conventional stopping reference | NIST ESTAR |
| Low-energy empirical stopping regression | Joy–Luo |
| Inelastic mean free path / loss spectrum | PENELOPE GOS and/or dielectric data |
| EEDL bremsstrahlung parser/interpolation | Direct Seltzer–Berger or PENELOPE tables |
| Bremsstrahlung angular distribution | PENELOPE partial-wave-fitted model and/or validated 2BN/2BS reference |
| Inner-shell ionization | EEDL versus Bote–Salvat / NIST SRD 164 and experimental production data |
| Characteristic X-ray production cross section | $\sigma_{\mathrm{ion}}\,\omega\,I$ compared with experimental/evaluated production data |
| Atomic relaxation cascade | EADL/PENELOPE-style transition accounting |
| Full-track transport | PENELOPE / Geant4 comparison under equivalent assumptions |
| Coherent radiation | Internal detailed-versus-approximate trajectory convergence |

### Process-specific validation requirement

Matching stopping power does not validate microscopic event distributions.

It is entirely possible for a model to satisfy

$$
S_{\mathrm{MC}}
\approx
S_{\mathrm{SBETHE}}
$$

while having incorrect

$$
\lambda_{\mathrm{in}},
\qquad
P(W),
\qquad
P(\theta),
\qquad
\sigma_K,
\qquad
\sigma_L.
$$

Likewise, matching the integrated bremsstrahlung yield does not validate the
photon angular distribution, and matching direct K-line production does not
validate an L/M relaxation cascade.

Validation should therefore remain process-specific.

---

## 15. Recommended implementation order

### Phase A — validate and finish the new EEDL radiation work

1. **Keep the EEDL `MF=23/26` bremsstrahlung energy spectrum as the production
   default.**
2. Compare its $d\sigma/dk$, integrated cross section, and radiative stopping
   moment directly against Seltzer–Berger/PENELOPE reference data.
3. **Replace the isotropic $1/(4\pi)$ bremsstrahlung factor with a validated
   angular model.**
4. Keep EEDL shell-ionization cross sections as the current characteristic
   vacancy-production baseline.
5. Compare characteristic X-ray production cross sections against independent
   experimental/evaluated references, not only parser and unit tests.
6. Decide whether finite-grid Lorentzian renormalization should remain a
   compatibility behavior or be replaced by physically truncated line tails.
7. Document the $\sim1~\mathrm{keV}$ characteristic-electron cutoff as a model
   limit for low-binding-energy shells.

### Phase B — improve the current trajectory physics

1. **Replace the screened-Rutherford/Mott-moment surrogate with full ELSEPA
   differential elastic scattering for every supported element.**
2. **Replace the Joy–Luo/ICRU stopping splice with SBETHE or an equivalent
   corrected-Bethe implementation.**
3. Revalidate PXR, CBS, continuum background, and characteristic spectra after
   the trajectory changes.

### Phase C — complete atomic relaxation and microscopic inelastic transport

1. Add a full radiative + Auger/Coster–Kronig atomic-relaxation cascade.
2. Add Bote–Salvat as an optional/reference inner-shell ionization backend.
3. Introduce a soft/hard energy-transfer cutoff $W_c$.
4. Use GOS/dielectric physics for soft losses.
5. Generate explicit Møller/GOS hard collisions.
6. Transport secondary electrons above a configurable threshold.

At this point, physical trajectory segments should be bounded by elastic,
hard-inelastic, and hard-radiative direction-changing events.

### Phase D — extend safely into the multi-MeV regime

Before declaring general multi-MeV or $100~\mathrm{MeV}$ validity:

1. include the material density-effect correction in production stopping;
2. extend elastic scattering beyond the NIST $300~\mathrm{keV}$ tables using
   ELSEPA or another vetted source;
3. make hard bremsstrahlung an explicit transport event;
4. include radiative energy loss and recoil;
5. transport emitted photons when downstream photon interactions matter;
6. consider a radiation-aware mixed elastic mode for performance;
7. validate radiation spectra against the fully detailed transport mode.

---

## 16. Recommended production and reference stack

The resulting long-term stack should separate production physics from
independent validation data:

| Physics role | Recommended production model | Important independent reference |
| --- | --- | --- |
| Elastic trajectory physics | ELSEPA full DCS | NIST SRD 64 / alternate ELSEPA generation |
| Mean collisional stopping | SBETHE or equivalent | ESTAR; existing JL/ICRU regression |
| Soft inelastic response | PENELOPE-like GOS | dielectric-loss/IMFP data |
| Optional low-energy solid-state response | $\epsilon(q,\omega)$ / Penn-like dielectric model | experimental optical/EELS data |
| Hard electron–electron collisions | Møller / GOS sampling | PENELOPE / Geant4 |
| Inner-shell vacancy production | EEDL baseline, Bote–Salvat option | NIST SRD 164; experiment |
| Atomic relaxation | full EADL-like cascade | xraydb line energies/widths; PENELOPE |
| Bremsstrahlung energy spectrum | current EEDL `MF=23/26` backend | direct Seltzer–Berger / PENELOPE |
| Bremsstrahlung angle | partial-wave-fitted / 2BN/2BS-type model | independent angular benchmarks |
| High-energy radiative transport | explicit hard-photon events + soft radiative loss | PENELOPE / Geant4 |

EEDL should therefore **not** be demoted to validation-only status. It now has a
legitimate production role in PyRITE for bremsstrahlung energy spectra and
shell-resolved vacancy production. The parts of the transport stack where EEDL
is not sufficient should be identified process by process rather than by a
general rule that evaluated atomic data are inferior to PENELOPE-style models.

---

## Final recommendation after the EEDL upgrade

PyRITE's present event-by-event trajectory architecture should be preserved.
The design is unusually well suited to PXR/CBS because the radiation solver
depends on the intermediate trajectory rather than only on macroscopic
transport observables.

The new EEDL work should also be preserved. It has already solved most of the
previously identified **bremsstrahlung energy-spectrum** problem and introduced
a credible direct-vacancy characteristic-X-ray estimator.

The near-term strategy is therefore

$$
\boxed{
\text{preserve the trajectory representation}
+
\text{preserve the new EEDL spectral physics}
+
\text{upgrade the remaining interaction physics}.
}
$$

The most important immediate improvements are now:

1. full ELSEPA differential elastic scattering;
2. a physical bremsstrahlung angular distribution while retaining the EEDL
   energy spectrum;
3. SBETHE-quality stopping;
4. a full atomic-relaxation cascade if characteristic spectroscopy is a primary
   goal.

Microscopic GOS/Møller inelastic transport, explicit secondaries, and coupled
hard-bremsstrahlung recoil become the next structural layer when PyRITE expands
from accurate thin-target primary trajectories and radiation scoring toward a
general-purpose electron–photon transport code.

For any later mixed/condensed implementation, the acceptance criterion should
remain **convergence of the calculated PXR/CBS radiation**, not merely agreement
in electron range, exit angle, or stopping power.
