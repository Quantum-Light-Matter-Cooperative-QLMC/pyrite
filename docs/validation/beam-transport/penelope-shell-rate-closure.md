# PENELOPE inner-shell rate substitution and stopping closure

Validation: `penelope-shell-rate-closure`. Status: unverified. The rescaled
moments are host-side only. Nothing samples them, and no transport mode
uses them.

## Source and intended quantity

PENELOPE-2024 §3.2.6.1 (NEA/MBDAV/R(2024)1, pp. 139–140) replaces the GOS
total cross section of each inner shell with a separate ionization cross
section. It also rescales the outer shells so that the adopted stopping power
is unchanged. The oscillator total cross section is split into inner shells
$i$ and outer shells $j$:

$$
\sigma_{\rm in}(E)=\sum_i\sigma_{{\rm in},i}(E)+\sum_j\sigma_{{\rm in},j}(E)
\qquad(\text{Eq. 3.141}),
$$

and then set to

$$
\sigma_{\rm in}(E)=\sum_i\sigma_{{\rm si},i}(E)+\mathcal N(E)\sum_j\sigma_{{\rm in},j}(E)
\qquad(\text{Eq. 3.142}).
$$

Three rules from the manual text accompany Eq. 3.142:

1. $\sigma_{{\rm si},i}$ replaces the total cross section of inner shell $i$
   "without altering details of the PDFs of the energy loss and scattering
   angle defined by the GOS model".
2. To approximate the density effect, the ionization cross section is
   "reduced by a factor equal to the ratio of the cross sections obtained from
   the GOS model with and without the density effect correction, $\delta_F$".
3. $\mathcal N(E)$ is common to all outer shells and chosen so that "the
   adopted stopping power is reproduced exactly". The manual notes that this
   is equivalent to $f_j\to f_j\mathcal N(E)$.

Compounds use additivity (Eq. 3.140).

## Implementation

`montecarlo/transport/shell_rates.py` starts from the raw per-oscillator
moments $\sigma^{(n)}_k$, $n=0,1,2$ of
[`penelope-shell-gos-moments`](penelope-shell-gos-moments.md). It evaluates,
per formula unit,

$$
\rho_i=\frac{\sigma^{(0)}_i(\delta_F)}{\sigma^{(0)}_i(\delta_F=0)},\qquad
\tilde\sigma_i=\rho_i\,\sigma_{{\rm si},i},\qquad
s_i=\frac{\tilde\sigma_i}{\sigma^{(0)}_i(\delta_F)},
$$

$$
\mathcal N(E)=\frac{S_{\rm adopted}(E)-\sum_is_i\sigma^{(1)}_i}{\sum_j\sigma^{(1)}_j},
$$

and returns $s_i\sigma^{(n)}_i$ for inner and $\mathcal N\sigma^{(n)}_j$ for
outer oscillators. Every channel is scaled: distant longitudinal, distant
transverse, and close. One factor per oscillator leaves
$\sigma^{(1)}_i/\sigma^{(0)}_i$ and $\sigma^{(2)}_i/\sigma^{(0)}_i$
unchanged. It therefore keeps the GOS energy-loss PDF, as rule 1 requires.
By construction, $\sum_k\sigma^{(1)}_k=S_{\rm adopted}$.

For the $\delta_F=0$ evaluation, the code recomputes `shell_gos_moments` on a
copy of the oscillators with $\Omega_p=0$. In the shell GOS, $\Omega_p$ enters
the moments only through $\delta_F$ (Eqs. 3.70–3.72), and $\Omega_p=0$ gives
$F(0)=0<1-\beta^2$, hence $\delta_F=0$. The code checks this assumption on
every call. The two runs must have bit-identical distant-longitudinal and
close channels and a zero $\delta_F$; otherwise a `RuntimeError` is raised.
Only the transverse bracket $\ln\gamma^2-\beta^2-\delta_F$ differs.

### Inner-shell set

The manual defines inner shells as K–N7 subshells with $U_i>E_c$ (§2.6;
footnote in §7.1), where

$$
E_c=\max\{50\ {\rm eV},\ U_{\max,\rm out}(Z_m)\}\qquad(\text{Eq. 2.112}),
$$

and $U_{\max,\rm out}(Z_m)$ is the largest O/P/Q ionization energy of the
heaviest element. §3.2.6.1 prints "binding energies less than the cut-off
energy $E_c$". That contradicts both other definitions and the physical
intent, so it is treated as a misprint. §3.2.6 also restricts simulated
ionizations to $U_i>{\rm EABS}$. That is a transport-cutoff choice with no
host analogue yet, so it is not applied here.

`inner_shell_cutoff_eV` implements Eq. 2.112 with the 50 eV floor passed as
`threshold_eV`. The default, `DEFAULT_INNER_SHELL_THRESHOLD_EV = 50 eV`,
equals `spectrum.characteristic._MIN_RELAXATION_CUTOFF_EV`, the existing
characteristic-relaxation cutoff; a test keeps the two equal. The
characteristic scorer therefore relaxes only vacancies that this closure
treats as inner-shell ionizations. `is_inner_shell` requires a bound
oscillator ($Z>0$) in a K, L, M or N series with the SBETHE (`pdatconf.p14`)
$U_k>E_c$. The conduction band is always outer. For Au and U, $E_c$ from
`pdatconf.p14` is 114 eV and 329 eV, which matches the manual's quoted
values. The manual's 92 eV for Ba equals the Ba N5 energy, not the largest
Ba O-shell energy (31 eV), so the stated definition gives 50 eV. The code
follows the stated definition. For Si, SiO₂ and MoS₂, $E_c=50$ eV. The inner
shells are Si K, L1–L3; O K; S K, L1–L3; and Mo K, L1–L3, M1–M5, N1. Mo N2/N3
(45/42 eV) are outer.

### Ionization cross sections and adopted stopping

$\sigma_{{\rm si},i}$ comes from packaged EEDL MF=23, not PENELOPE's
DWBA/Bote–Salvat database. This is an owner-approved baseline; the
Bote–Salvat backend is #92. It is interpolated linearly by
`material_shell_ionization_rates` ([`eedl-material-shell-rates`](eedl-material-shell-rates.md)).
The SBETHE shell and EEDL subshells are joined by x-ray label
(`match_eedl_shells`). EEDL spin-orbit partners that SBETHE leaves empty are
summed into the filled $n,l$ shell. Multiplying by the formula count $n_Z$
gives per-formula-unit values. An inner oscillator without an EEDL channel
raises. An accessible shell outside its EEDL grid also raises, with no
extrapolation.

**Deviation from PENELOPE.** $S_{\rm adopted}$ is the corrected SBETHE
`stp.dat` stopping cross section (owner decision for #93). It is
interpolated log-log (`adopted_stopping_cs`), and energies outside the table
(1 keV–1 GeV) raise. PENELOPE keeps its own GOS stopping. The raw GOS
stopping here exceeds `stp.dat` by up to 26% at 1 keV
([`penelope-shell-gos-moments`](penelope-shell-gos-moments.md)). Closing to
`stp.dat` therefore also removes that excess from the outer shells.

### Fail-closed rules

- $\mathcal N$ must be finite and positive. If inner-shell stopping reaches
  $S_{\rm adopted}$, a `ValueError` is raised.
- A positive $\sigma_{{\rm si},i}$ on a shell with zero GOS cross section has
  no loss PDF to carry it and raises. This can happen only when EEDL and
  SBETHE binding energies straddle $E$; they are equal for Si, O, S and Mo.
- Non-finite or negative $\sigma_{{\rm si},i}$, keys that are not bound
  oscillators, and non-positive $S_{\rm adopted}$ raise.

## Assumptions

- Free-atom EEDL subshell cross sections, additive in compounds, with no
  solid-state or density effect beyond the GOS $\rho_i$ factor.
- The GOS $W$ PDF of an inner shell is kept although its normalization comes
  from EEDL. The binding energy defining the PDF is SBETHE $U_k$.
- One $\mathcal N(E)$ for all outer shells, applied to all three moments.
  This is equivalent to $f_j\to\mathcal N f_j$ with $\delta_F$ held fixed.
- Electrons only; positron moments are not constructed.

## Limiting cases

- No inner shells (empty mapping): $\mathcal N=S_{\rm adopted}/\sigma^{(1)}_{\rm GOS}$.
- $\delta_F=0$ (MoS₂ at $\le20$ keV): every $\rho_i=1$ exactly.
- $\sigma_{{\rm si},i}=\sigma^{(0)}_i(\delta_F=0)$ for every inner shell and
  $S_{\rm adopted}=\sigma^{(1)}_{\rm GOS}$ give $s_i=1$ and $\mathcal N=1$:
  the raw GOS moments are returned unchanged.
- $E\le U_i$: the shell is closed in both GOS and EEDL and contributes nothing.

## Diagnostics (recorded, not tuned)

The following values are for the measured-conduction-band oscillators of
[`penelope-shell-oscillators`](penelope-shell-oscillators.md), with
$E_c=50$ eV.

$\mathcal N(E)$:

| material | 1 keV | 2 keV | 5 keV | 10 keV | 20 keV | 50 keV | 100 keV | 1 MeV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silicon | 0.965 | 1.061 | 1.125 | 1.141 | 1.112 | 1.108 | 1.095 | 1.056 |
| sio2 | 0.770 | 0.803 | 0.836 | 0.860 | 0.844 | 0.832 | 0.819 | 0.791 |
| mos2 | 1.038 | 1.032 | 1.110 | 1.187 | 1.227 | 1.270 | 1.281 | 1.269 |

Total $\sigma^{(0)}$ (inverse IMFP), closed/raw:

| material | 1 keV | 2 keV | 5 keV | 10 keV | 20 keV | 50 keV | 100 keV | 1 MeV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silicon | 0.952 | 1.038 | 1.099 | 1.115 | 1.092 | 1.090 | 1.080 | 1.050 |
| sio2 | 0.775 | 0.810 | 0.846 | 0.871 | 0.857 | 0.847 | 0.836 | 0.810 |
| mos2 | 1.007 | 1.001 | 1.073 | 1.144 | 1.183 | 1.223 | 1.234 | 1.225 |

IMFP at 1/10/100 keV, raw → closed (Å): Si 18.6→19.6, 128.7→115.5,
810.7→750.4; SiO₂ 15.4→19.8, 107.0→122.9, 674.8→807.4; MoS₂ 15.0→14.9,
101.1→88.4, 629.8→510.5.

Total $\sigma^{(2)}$ (straggling), closed/raw:

| material | 1 keV | 2 keV | 5 keV | 10 keV | 20 keV | 50 keV | 100 keV | 1 MeV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silicon | 0.806 | 0.733 | 0.844 | 0.943 | 0.954 | 0.971 | 0.983 | 1.002 |
| sio2 | 0.815 | 0.836 | 1.061 | 1.080 | 1.131 | 1.110 | 1.100 | 1.088 |
| mos2 | 0.647 | 0.685 | 0.690 | 0.784 | 0.826 | 0.870 | 0.904 | 0.956 |

EEDL / GOS inner-shell $\sigma^{(0)}$ (with $\delta_F$; — means the shell is
closed):

| shell | 1 keV | 2 keV | 5 keV | 10 keV | 20 keV | 50 keV | 100 keV | 1 MeV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Si L2 (Si) | 0.808 | 0.836 | 0.887 | 0.912 | 0.967 | 0.993 | 1.010 | 1.103 |
| Si L3 (Si) | 0.828 | 0.854 | 0.906 | 0.931 | 0.987 | 1.013 | 1.031 | 1.126 |
| Si L1 (Si) | 0.558 | 0.586 | 0.612 | 0.617 | 0.641 | 0.642 | 0.643 | 0.672 |
| Si K (Si) | — | 0.014 | 0.502 | 0.993 | 0.935 | 0.953 | 0.996 | 1.072 |
| Si L2 (SiO₂) | 1.353 | 1.263 | 1.290 | 1.308 | 1.376 | 1.406 | 1.428 | 1.597 |
| Si L3 (SiO₂) | 1.385 | 1.290 | 1.315 | 1.334 | 1.404 | 1.434 | 1.457 | 1.631 |
| Si L1 (SiO₂) | 0.681 | 0.922 | 0.900 | 0.891 | 0.917 | 0.912 | 0.911 | 0.976 |
| O K (SiO₂) | 0.115 | 0.532 | 1.765 | 1.483 | 1.494 | 1.494 | 1.505 | 1.668 |
| Si K (SiO₂) | — | 0.006 | 0.287 | 0.910 | 1.523 | 1.416 | 1.450 | 1.587 |
| Mo N1 (MoS₂) | 0.315 | 0.335 | 0.338 | 0.330 | 0.333 | 0.317 | 0.306 | 0.294 |
| S L3 (MoS₂) | 0.651 | 0.739 | 0.813 | 0.851 | 0.911 | 0.947 | 0.970 | 1.078 |
| S L1 (MoS₂) | 0.404 | 0.465 | 0.509 | 0.525 | 0.552 | 0.560 | 0.563 | 0.596 |
| Mo M5 (MoS₂) | 0.460 | 0.556 | 0.683 | 0.734 | 0.804 | 0.855 | 0.887 | 1.013 |
| Mo M1 (MoS₂) | 0.114 | 0.320 | 0.395 | 0.435 | 0.466 | 0.477 | 0.480 | 0.500 |
| S K (MoS₂) | — | — | 0.363 | 0.713 | 0.722 | 0.773 | 0.800 | 0.873 |
| Mo L3 (MoS₂) | — | — | 0.232 | 0.548 | 0.601 | 0.671 | 0.727 | 0.851 |
| Mo K (MoS₂) | — | — | — | — | — | 0.460 | 0.675 | 0.807 |

The density factor $\rho_i$ is at least 0.994 at 100 keV. At 1 MeV its
minima are 0.951 (Si), 0.909 (SiO₂) and 0.936 (MoS₂).

Reading. $\mathcal N$ stays within 0.77–1.28 and is positive at every energy.
The 1 keV Si and SiO₂ values below 1 come mainly from removing the raw GOS
stopping excess over `stp.dat`. At higher energies $\mathcal N$ compensates
the difference between EEDL and GOS inner-shell rates. Si L shells in Si and
SiO₂ have different EEDL/GOS ratios because Eq. 3.63 gives different $W_k$
($a=2.20$ versus 3.47). Near threshold, EEDL K-shell rates fall far below the
GOS values; for Si K at 2 keV ($E/U=1.08$) the ratio is 0.014. The closure
changes the IMFP by −23% to +23% and the straggling by −35% to +13%. These
changes come from the redistribution between inner-shell (large $W$) and
outer-shell (small $W$) losses. The IMFP moves more for MoS₂, where many
inner shells have EEDL/GOS $<1$ and $\mathcal N$ compensates in the
high-$\sigma^{(0)}$ outer oscillators. None of these values has been compared
with IMFP or straggling measurements.

## Checks

`tests/montecarlo/test_shell_rates.py` covers the following:

- exact closure $\sum\sigma^{(1)}=S_{\rm stp}$ ($10^{-12}$, `abs=0`), with
  finite positive $\mathcal N$ for three materials at eight energies;
- inner $\sigma^{(1)}/\sigma^{(0)}$ and $\sigma^{(2)}/\sigma^{(0)}$ unchanged
  ($10^{-12}$), and outer shells scaled by one $\mathcal N$;
- the adopted inner $\sigma^{(0)}$ equals an independent EEDL lin-lin
  interpolation times $n_Z\rho_i$;
- $\rho_i$ versus an independent transverse-bracket recomputation at 1 MeV;
- $\rho_i=1$ where $\delta_F=0$;
- the no-inner-shell limit and the GOS-consistent identity limit;
- spin-orbit merging (L2+L3 into a fixture L2) and the formula count;
- $E_c$ selection, including a heavy-element O-shell and Mo N1 versus N2/N3;
- failures for $\mathcal N\le0$, invalid stopping or keys, EEDL above a
  closed GOS shell, and energies outside the `stp.dat` or EEDL grid;
- the default threshold equals the characteristic relaxation cutoff.

## Validation still needed

Fresh-context source-to-code verification of Eqs. 3.141–3.142 and of the
inner-shell rule, including the §3.2.6.1 "less than $E_c$" misprint and the
Ba example. Comparison of the closed IMFP with measured or
optical-data IMFPs, and of inner-shell rates with Bote–Salvat (#92). The
effect on the $W_c$ soft/hard partition is still open, and
[`penelope-shell-gos-moments`](penelope-shell-gos-moments.md) must be
validated before this closure is trusted.
