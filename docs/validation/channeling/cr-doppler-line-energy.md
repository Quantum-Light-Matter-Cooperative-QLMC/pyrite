# `cr-doppler-line-energy` re-derivation

Claim: `materials/channeling/transverse.py::{doppler_denominator,line_energy_eV}`. Source: relativistic kinematics; Andersen et al. (1983).

## Independent derivation (before reading the body)

An electron of velocity $\mathbf v$, $\beta=v/c$, loses transverse energy $\Delta E_\perp$ and emits a photon with $\hbar\omega$ along $\hat{\mathbf n}$. Emission of a photon with wave vector $\mathbf k$ from a charge moving at constant $\mathbf v$ (energy $\hbar\omega$ against momentum $\hbar\mathbf k$, with $d E=\mathbf v\cdot d\mathbf p$ for the electron) gives $\Delta E_\perp=\hbar\omega-\mathbf v\cdot\hbar\mathbf k=\hbar\omega(1-\beta\cos\theta)$, with $\theta$ the angle between $\hat{\mathbf n}$ and $\mathbf v$. Neglecting recoil, $O(\hbar\omega/E)$,

$$
\hbar\omega=\frac{\Delta E_\perp}{1-\beta\cos\theta}.
$$

Cancellation-free form. Using $\cos\theta=1-2\sin^2(\theta/2)$,

$$
1-\beta\cos\theta=(1-\beta)+2\beta\sin^2\frac\theta2,\qquad 1-\beta=\frac{1-\beta^2}{1+\beta}=\frac{1}{\gamma^2(1+\beta)} .
$$

Therefore

$$
\boxed{1-\beta\cos\theta=\frac{1}{\gamma^2(1+\beta)}+2\beta\sin^2\frac\theta2 .}
$$

Forward ($\theta=0$): $\hbar\omega=\gamma^2(1+\beta)\Delta E_\perp\to2\gamma^2\Delta E_\perp$. With $\gamma=1+T/m_ec^2$, $\gamma^{-1}=m_ec^2/(m_ec^2+T)$ and $\beta=\sqrt{1-\gamma^{-2}}$ no subtraction of nearly equal numbers remains.

PXR cross-check: the vacuum resonance has Doppler denominator $1-\hat{\mathbf n}\cdot\boldsymbol\beta$, the same quantity with $\theta$ measured from $\mathbf v$.

## Independent numeric check (mpmath, 50 digits, and float)

| Case | Cancellation-free form | $1-\beta\cos\theta$ | Relative difference |
| --- | --- | --- | --- |
| 5 MeV, 3° | 0.005672662024408 | 0.005672662024409 | $-1.5\times10^{-14}$ |
| 5 MeV, 20° | 0.0643556698220521 | 0.0643556698220521 | 0 |
| 30 keV, 120° | 1.164188088180141 | 1.164188088180141 | 0 |
| 1 GeV, $10^{-4}$ rad | $1.35426641497\times10^{-7}$ | $1.35426641497\times10^{-7}$ | agree to about $10^{-32}$ in 50-digit arithmetic |
| 40 MeV, $\theta=0$ | $7.9557543\times10^{-5}$ | $1-\beta\cos0$ in floats | float form $2.5\times10^{-13}$ off, due to cancellation |

The last line shows the naive form loses digits and the rewritten form does not. For $\theta=0$, `line_energy_eV(1,40e6,0)/(gamma**2*(1+beta))` is $1.0000000000000002$, so the forward factor is $\gamma^2(1+\beta)$.

## Diff against the implementation

`doppler_denominator`: `inverse_gamma = m/(m+T)`, `beta = sqrt(1-inverse_gamma**2)`, return `inverse_gamma**2/(1+beta) + 2*beta*sin(theta/2)**2`. This is identical to the derived form. `line_energy_eV` divides $\Delta E_\perp$ by it, which is $\hbar\omega=\Delta E_\perp/(1-\beta\cos\theta)$. Photon recoil is neglected, as stated.

Cross-check against `materials/emission.py::photon_energies`: I did not re-derive the PXR resonance denominator there, and I confirm only that the docstring convention ($\theta$ from $\mathbf v$, $\beta$ total speed) matches the derived $1-\mathbf n\cdot\boldsymbol\beta$. The ledger anchor `test_doppler_denominator_matches_pxr_resonance` pins the numeric equality. The docstring phrase "$2\gamma^2\Delta E_\perp(1+\beta)/2$" is correct but awkward; it equals $\gamma^2(1+\beta)\Delta E_\perp$.

Si(110) line: $\hbar\omega(1\to0)=76.69\ \mathrm{keV}$ from $2\gamma^2\Delta\varepsilon$, against Azadegan's 76.2071 keV (see `cr-transverse-eigenproblem`).

## Verdict

`rederived`.
