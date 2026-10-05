# Positron transport

With `pair_production_model` the secondary cascade converts coupled hard photons into electron–positron pairs ([pair conversion](../radiation-physics/hard-bremsstrahlung-events.md#pair-conversion-of-hard-photons)). `positron_transport=True` (issue #276; `pyrite profile numerics set --positron-transport`) transports those positrons instead of only recording them. The default `False` leaves every run bit for bit unchanged, and the key joins run identity only when it is on.

A positron interacts like an electron except for its charge and its distinguishability from target electrons. PyRITE therefore keeps the electron algorithm and replaces only the tables and the close-collision law:

| Process | Electron | Positron |
| --- | --- | --- |
| Elastic | ELSEPA, exchange + correlation–polarization | ELSEPA positron mode: no exchange, LDA correlation–polarization (separate `elsepa-positron-tables` release) |
| Soft collision stopping | SBETHE `stp.dat` | SBETHE positron mode (`sbethe-positron-tables`) |
| Hard close collisions | Møller, $W_{\max}=(E+U_k)/2$ | Bhabha, $W_{\max}=E$ |
| Hard distant collisions | shell GOS | shell GOS, bounded by $W_{\max}=E$ |
| Bremsstrahlung | BremsLib | BremsLib $\times F_p(Z,E)$ |
| Annihilation | — | not simulated (#295) |

## Close collisions: Bhabha

The close energy-loss DCS of oscillator $k$ is PENELOPE-2024 Eq. 3.92 {cite:p}`salvat2024penelope,bhabha1936`,

```{math}
:label: eq-positron-bhabha

\frac{d\sigma^{(+)}_{\rm clo}}{dW} = \frac{2\pi e^4}{m_ev^2}\,f_k\,\frac{F^{(+)}(E,W)}{W^2},
\qquad
F^{(+)} = 1 - b_1x + b_2x^2 - b_3x^3 + b_4x^4,\quad x = W/E,
```

on $Q_k\le W\le E$, with $Q_k=U_k$ for bound shells and $W_{cb}$ for the conduction band. With $g=((\gamma-1)/\gamma)^2$ (Eq. 3.90):

```{math}
b_1 = g\,\frac{2(\gamma+1)^2-1}{\gamma^2-1},\quad
b_2 = g\,\frac{3(\gamma+1)^2+1}{(\gamma+1)^2},\quad
b_3 = g\,\frac{2\gamma(\gamma-1)}{(\gamma+1)^2},\quad
b_4 = g\,\frac{(\gamma-1)^2}{(\gamma+1)^2}.
```

The moments $\sigma^{(n)}=\int W^{n}\,d\sigma$ use the closed forms $J^{(+)}_n$ of Eqs. 3.112–3.114. Every $b_k\to0$ as $E\to0$, recovering Rutherford's $1/W^2$; as $\gamma\to\infty$, $(b_1,b_2,b_3,b_4)\to(2,3,2,1)$. The distant longitudinal and transverse terms are charge-independent at first Born order, so only their upper bound changes from $(E+U_k)/2$ to $E$. At high energy the summed stopping reaches the Bethe formula with Eq. 3.122,

```{math}
f^{(+)}(\gamma) = 2\ln2 - \frac{\beta^2}{12}\left[23 + \frac{14}{\gamma+1} + \frac{10}{(\gamma+1)^2} + \frac{4}{(\gamma+1)^3}\right],
```

which the tests check to $2\times10^{-4}$.

**Closure and inner shells.** As for electrons ([shell soft/hard transport](shell-soft-hard-transport.md)), the outer shells scale by one $\mathcal N(E)$ so that soft plus mean hard loss equals the adopted stopping; for positrons that is the SBETHE positron table. EEDL ionization cross sections are electron-impact only, and the PENELOPE positron tables (`pdpsi`) are not vendored, so positron inner shells keep their own Bhabha GOS cross sections (scale 1). Their binding energy is still reserved at a hard inner-shell collision, as for electrons.

**Sampling.** The CPU and CUDA kernels invert the same restricted CDF as for electrons, with $J^{(+)}_0$ in place of $J^{(-)}_0$. Positron tables tag each channel's branch code with $+3$, so the kernel signatures and the electron path are unchanged; `hard_channel` keeps the species-independent $3\times$oscillator$+$branch. Binary-collision kinematics do not depend on the charge, so the primary and secondary polar cosines are the electron ones. The struck electron takes $W$ (or $W-U_k$ for an inner shell) and becomes an ordinary cascade secondary; the positron continues with $E-W$.

## Bremsstrahlung: $F_p$ scaling

BremsLib tabulates electrons only. Following PENELOPE (Eq. 3.153), the positron DCS is the electron DCS times a factor independent of photon energy and angle,

```{math}
:label: eq-positron-fp

F_p(Z,E) = 1 - \exp\!\left(\sum_{j=1}^{7} a_j t^j\right),\qquad t = \ln\!\left(1 + \frac{10^6\,E}{Z^2 m_ec^2}\right),
```

with the coefficients of Eq. 3.154, a fit to Kim et al.'s positron/electron radiative stopping ratio {cite:p}`kim1986positron` to about 0.5 %. SBETHE's positron mode applies the same factor. `positron_bremslib_tables` multiplies each element's scaled SDCS and DDCS at every incident node by $F_p$; the radiative soft/hard partition, photon sampling and coupling are then unchanged. $F_p\to1$ at high energy and decreases with $Z$, since nuclear repulsion suppresses positron bremsstrahlung.

## Cascade and energy accounting

A pair positron with $E_+>T_s$ is launched in the next generation with `launch_kind = 3`, from the interaction point and with its sampled direction; its stream key hashes (seed, parent track, photon ordinal) under a salt of its own. Each generation runs its electrons and its positrons as two transport calls and joins them, electrons first, into one generation. Hard collisions of a positron launch electron secondaries; its hard photons may convert again. Positron rows carry the usual fields; `secondary_tracks["launch_kind"]` identifies them.

Annihilation is not simulated (#295). A positron below $T_s$ deposits its kinetic energy locally (`positron_subthreshold`, part of `deposited`), and a transported one ends by escaping or at the tracking cutoff like an electron. `secondary_energy_balance` books each pair's $2m_ec^2$ as `positron_rest_pending` in place of `pair_rest_mass`, whatever the positron's fate, so

```{math}
k = \text{deposited} + \text{escaped}(e^-,e^+,\gamma) + \text{positron\_rest\_pending}
```

closes for each pair, and the balance closes per history. The run warns that no annihilation photons are emitted.

## Requirements and limits

Positron transport needs `pair_production_model` (hence the secondary cascade, shell-soft-hard transport with catalog keys, and coupled BremsLib), the installed `sbethe-positron-tables` and, with `elastic_model="elsepa"`, `elsepa-positron-tables` (`pyrite tables fetch sbethe-tables --projectile positron` and `pyrite tables fetch elsepa --projectile positron`). The deprecated Mott model has no positron counterpart and is refused. `atomic_electron_deflection="kawrakow"` applies its electron $\xi$ unchanged to positrons: the subtracted hard share is the Møller rather than the Bhabha $\sin^2\theta$ moment. Positronium, channeling and annihilation in flight are out of scope.

Validation: `bhabha-close`, `sbethe-positron-stopping`, `elsepa-positron-elastic-sampling`, `positron-brems-scaling`.
