# Atomic-electron angular deflection

The elastic tables of [Elastic scattering](elastic-scattering.md) describe scattering by the screened nucleus, whose rate scales as $Z^2$. The $Z$ atomic electrons also deflect a passing electron, in the same inelastic collisions that remove its energy. Continuous stopping ([Stopping power](stopping-power.md)) removes that energy but samples no collision, so before issue #317 it omitted their deflection entirely. The deficit grows like $1/Z$. At Si 300 keV PyRITE backscattered 9 % fewer primaries than Geant4's event-by-event DPWA reference, whose `isscpcor` correction includes this deflection; at W the gap is within statistics.

`atomic_electron_deflection="kawrakow"` restores it, following Bethe's $Z^2 \to Z(Z+1)$ prescription{cite:p}`bethe1953` with Kawrakow's correction for collisions the transport samples explicitly{cite:p}`kawrakow1997,kawrakow2021egsnrc`. It is the default of every run built from `pyrite.Numerics`; `"none"` reproduces elastic-only transport bit for bit. Low-level `simulate_trajectories` defaults to `None`.

## Model

Each element's elastic cross section is multiplied by

```{math}
:label: eq-aed-rate-scale

\sigma_i(E) \;\to\; \sigma_i(E)\left(1 + \frac{\xi(E, T_c)}{Z_i}\right),
```

with the angular distribution of every elastic collision unchanged. The parameter

```{math}
:label: eq-aed-xi

\xi(T, T_c) = \xi_0 - \min\!\left(\frac{g_M(\tau, \tau_c)}{g_R(\eta)},\; \xi_0\right),
\qquad \xi_0 = 1,
```

removes the share of atomic-electron deflection that the transport already samples in hard collisions above $T_c$. Here $\tau = T/mc^2$ and $\tau_c = T_c/mc^2$. $g_M$ is the $\sin^2\theta$ moment of Møller collisions with energy transfer in $[T_c, T/2]$, per atomic electron (PIRS-701 Eq. 4.7.21):

```{math}
:label: eq-aed-gm

\begin{aligned}
g_M(\tau,\tau_c) ={}& \ln\frac{\tau}{2\tau_c}
 + \left[1 + \frac{(\tau+2)^2}{(\tau+1)^2}\right]\ln\frac{2(\tau-\tau_c+2)}{\tau+4} \\
&- \left[\frac{(\tau+2)^2}{4} + \frac{(\tau+2)(\tau+1/2)}{(\tau+1)^2}\right]
   \ln\frac{(\tau+4)(\tau-\tau_c)}{\tau(\tau-\tau_c+2)} \\
&+ \frac{(\tau-2\tau_c)(\tau+2)}{2}\left[\frac{1}{\tau-\tau_c} - \frac{1}{(\tau+1)^2}\right],
\end{aligned}
```

and $g_M = 0$ for $\tau \le 2\tau_c$, where no Møller collision is hard. $g_R$ is the same moment of the screened-Rutherford nucleus per $Z^2$,

```{math}
:label: eq-aed-gr

g_R(\eta) = (1+2\eta)\ln\!\left(1+\frac{1}{\eta}\right) - 2,
```

with the Molière screening parameter of the layer (PIRS-701 Eqs. 4.7.6–4.7.8)

```{math}
:label: eq-aed-eta

\eta = \frac{\chi_{cc}^2}{4\, b_c\, (mc^2)^2\, \tau(\tau+2)},
\qquad
\frac{\chi_{cc}^2}{b_c} = \frac{0.1569~\mathrm{MeV^2}}{7821.6}\,
\exp\!\left(\frac{Z_X - Z_E}{Z_S}\right),
```

$Z_S = \sum_i p_i Z_i(Z_i+\xi_0)$, $Z_E = \sum_i p_i Z_i(Z_i+\xi_0)\ln Z_i^{-2/3}$, $Z_X = \sum_i p_i Z_i(Z_i+\xi_0)\ln(1 + 3.34\,\alpha^2 Z_i^2)$, and $p_i$ the atom fractions. The density cancels from $\eta$.

### Derivation of the normalization

Write both moments in units $mc^2 = 2\pi r_0^2 = 1$. The screened-Rutherford nucleus, $d\sigma/d\mu = Z^2/[\beta^2\tau(\tau+2)(1-\mu+2\eta)^2]$, has $\int(1-\mu^2)\,d\sigma = 2 Z^2 g_R/[\beta^2\tau(\tau+2)]$. The Møller cross section per electron, with the primary's deflection $\sin^2\theta = 2w/[\tau(\tau-w+2)]$ after losing $w$, gives the same prefactor times $2 g_M$; a direct quadrature of the Møller DCS reproduces {eq}`eq-aed-gm` to $10^{-9}$. If all atomic-electron deflection is $Z\xi_0$ in units of the nuclear scattering power, the explicit hard part is $Z g_M/g_R$, which leaves $Z(Z + \xi_0 - g_M/g_R)$, i.e. {eq}`eq-aed-xi`.

PIRS-701 Eq. 4.7.20 prints $\xi = \xi_0[1 - g_M/((\bar Z+\xi_0)g_R)]$, which removes only a $1/(\bar Z+1)$ fraction of the hard share. Geant4's `G4eDPWAElasticDCS::InitSCPCorrection`{cite:p}`geant4dpwa` instead multiplies $Z(Z+1)$ by $1 - g_M/[(Z+1)g_R]$, which for one element is {eq}`eq-aed-xi`. PyRITE follows the derivation above; the printed Eq. 4.7.20 bracket is the multiplier of $Z(Z+\xi_0)$, not $\xi$. The printed Eq. 4.7.22 also misplaces its bracket, $(1+2\eta)[\ln(1+1/\eta)-2]$, which turns negative above $\eta \approx 0.16$; {eq}`eq-aed-gr` is the $\sin^2$ moment derived directly. Geant4 also omits $(mc^2)^2$ from $\eta$, making it about 3.8 times smaller; that matters only above twice its production cut, so not in the #183 references.

## Mode by mode

- **Continuous stopping** samples no atomic-electron collision, so $T_c = \infty$ and $\xi = 1$: every element's rate is scaled by $(Z_i+1)/Z_i$. The factor is constant, so it multiplies the screened-Rutherford, Mott and ELSEPA rate coefficients exactly, on every core, the energy LUT and the element choice.
- **Shell soft/hard** deflects explicitly in hard collisions above $W_c$, so $T_c = W_c$. The energy-dependent $\xi$ is evaluated on each ELSEPA energy node and joins the log-log rate interpolation. The analytic `"sr"` and `"mott"` rates have no energy grid to carry it, so `simulate_trajectories` rejects the combination, and a run case on deprecated Mott that resolves to shell transport keeps elastic-only deflection with a warning.

At the production cutoff $W_c = 50$ eV in silicon, $g_M > g_R$ from 315 eV upward, so $\xi$ clips to 0 and shell transport is unchanged there ($\xi = 0.16$ at 200 eV, 0.012 at 300 eV). The free-electron Møller law overstates distant soft collisions near binding energies. The shell GOS model gives a soft share of 0.3–1.6 % of the angular diffusion rate (`checks/soft_inelastic_deflection.py`), which this correction does not restore. For $W_c \ge 1$ keV, $\xi$ is between 0.29 and 1 for Si from 2.5 keV to 1 MeV.

## Measured effect

Against Geant4 11.4.2 TestEm5 with event-by-event DPWA elastic scattering (`isscpcor` on), 100,000 primaries per case, ELSEPA + SBETHE + coupled BremsLib, 10 keV cutoff, continuous stopping, no straggling (`checks/full_track_bremslib/`, 2026-10-04):

| Case | Geant4 DPWA T / R | PyRITE `"none"` T / R (z) | PyRITE `"kawrakow"` T / R (z) |
|---|---|---|---|
| Si 300 keV, 100 µm | 0.8428 / 0.1109 | 0.8613 / 0.0987 (+11.7 / −8.9) | 0.8469 / 0.1084 (+2.6 / −1.8) |
| Si 800 keV, 100 µm | 0.9936 / 0.0063 | 0.9943 / 0.0056 (+2.1 / −2.0) | 0.9936 / 0.0063 (0.0 / 0.0) |
| W 300 keV, 10 µm | 0.5448 / 0.4248 | 0.5486 / 0.4207 (+1.7 / −1.9) | 0.5443 / 0.4247 (−0.2 / −0.1) |
| W 800 keV, 10 µm | 0.9191 / 0.0806 | 0.9221 / 0.0775 (+2.5 / −2.6) | 0.9202 / 0.0794 (+1.0 / −1.0) |

T and R are the transmitted and backscattered primary fractions; z is the difference in combined binomial standard errors. Shell soft/hard Si runs at $W_c = 50$ eV are unchanged by the correction, count for count, because $\xi = 0$ above the 10 keV cutoff.

## Assumptions

- Atomic-electron deflection follows the nuclear angular law of the active elastic model; only its rate is added. This is the Bethe/EGSnrc single-parameter treatment. Fano's separate inelastic screening logarithm{cite:p}`fano1954` is not used.
- The hard share uses free-electron Møller kinematics for every atomic electron, ignoring binding; the transported particles are electrons, so no Bhabha form is needed.
- $\xi$ is a property of the layer, through $\eta$; each element's factor $1 + \xi/Z_i$ then follows from its own electron count.

## Limiting cases

- $T_c \to \infty$ (continuous mode, or $T \le 2T_c$): $\xi = 1$, rate $\times (Z+1)/Z$.
- $Z \to \infty$: the factor tends to 1, so high-$Z$ transport is barely changed (W: $\times 1.0135$).
- $g_M \ge g_R$: $\xi = 0$, which is elastic-only transport with explicit hard collisions.
- `atomic_electron_deflection=None` or `"none"`: no coefficient is touched, bit for bit.

## Validation

`Validation: inelastic-angular-deflection` — see the row in the [transport and background ledger](../../validation/ledger-transport-background.md#inelastic-angular-deflection).
