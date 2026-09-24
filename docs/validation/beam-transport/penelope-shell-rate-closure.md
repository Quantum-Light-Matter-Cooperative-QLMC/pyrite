# PENELOPE inner-shell rate substitution and stopping closure

Validation: `penelope-shell-rate-closure`. Status: rederived. The rescaled
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

### Independent SiO₂ total-rate gate

[Shinotsuka et al., *Surface and Interface Analysis* 51 (2019),
Table 5](https://pmc.ncbi.nlm.nih.gov/articles/PMC7047655/) calculate SiO₂
IMFPs from an optical loss function with the relativistic full Penn algorithm.
Their Table 1 uses 2.19 g/cm³, near this catalog's 2.19993 g/cm³. Table 5
reports $E=T-E_g-E_v$, while their Eq. 7 uses $T'=T-E_g$; for SiO₂,
$E_v=10$ eV. The benchmark evaluates this model at $E+10$ eV to match
$T'$. The tabulated model values below use that offset; it changes the model
rate by 0.04–0.40% at these energies. The published
values are calculated reference data, not direct measurements.

| Energy (eV) | Full Penn (nm) | Raw GOS (nm) | Closed GOS (nm) | Closed/reference |
| ---: | ---: | ---: | ---: | ---: |
| 1998.2 | 5.09 | 2.741 | 3.430 | 0.674 |
| 9897.1 | 19.0 | 10.615 | 12.257 | 0.645 |
| 19930.4 | 33.9 | 19.191 | 22.389 | 0.660 |

The model IMFP is $10^7/[N\sigma^{(0)}]$ nm for molecular number density
$N=\rho N_A/M$ in cm⁻³ and cross section in cm² per formula unit. The
33–35% IMFP deficit means the closed model predicts 48–55% more
inelastic collisions per path than this independent calculation. The
recorded ratios are pinned by a non-failing characterization test; see
[IMFP note](#total-imfp-note-not-a-transport-gate). This is a total-rate
discrepancy, separate from the
exact corrected-stopping closure. The raw GOS already has a shorter IMFP;
the correction moves it toward the reference. The conduction-band oscillator
accounts for 95.7–96.8% of the closed total rate at these energies. Its
distant-longitudinal channel alone contributes 83.5–85.3% of the total;
its close channel contributes 10.4–13.3%, and its transverse channel is
negligible. The 22 eV point-loss conduction response therefore dominates
the model's total rate, though the aggregate IMFP cannot identify whether the
oscillator response, finite-momentum construction, or reference method causes
it. A differential loss-spectrum comparison is needed before changing the
rate.

### Independent Si total-rate gate

[Shinotsuka et al., *Surface and Interface Analysis* 47 (2015),
Table 2](https://mdr.nims.go.jp/datasets/faa3fcd0-cc22-4955-bbb3-0b686504eea4?locale=en)
calculate elemental Si IMFPs from measured optical energy-loss functions
with the relativistic full Penn algorithm. The table's kinetic energy is
measured above the Fermi level; no SiO₂-style valence-band offset applies.
The same catalog-density conversion gives:

| Energy (eV) | Full Penn (nm) | Raw GOS (nm) | Closed GOS (nm) | Closed/reference |
| ---: | ---: | ---: | ---: | ---: |
| 1998.2 | 4.25 | 3.288 | 3.166 | 0.745 |
| 9897.1 | 16.04 | 12.760 | 11.459 | 0.714 |
| 19930.4 | 28.77 | 23.076 | 21.124 | 0.734 |

The closed Si IMFP is 26–29% shorter, corresponding to a 34–40% collision-rate
excess. Its conduction-band oscillator supplies 91–94% of the closed total
rate. Both Si and SiO₂ therefore fail this independent total-rate check;
the sign is shared, while the material-specific oscillator and reference
responses differ. These are calculated reference IMFPs, not direct
measurements, and neither table resolves the differential loss spectrum.

### Conduction-band rate diagnosis

The closed Si conduction-band *distant-longitudinal* channel supplies
81.1–81.9% of the total zeroth moment at the three benchmark energies;
SiO₂'s supplies 83.5–85.3%. This is the PENELOPE point-loss oscillator at
the sourced $W_{cb}$, not a measured differential rate. To test whether its
resonance alone could explain the deficit, rebuild the oscillator set and
stopping closure with $W_{cb}$ changed, leaving $f_{cb}$, catalog $I$,
$\Omega_p$, EEDL inner rates, and corrected stopping fixed. This also
re-solves the Sternheimer factor, so it is a self-consistent sensitivity
calculation, **not** an endorsed material-parameter change:

| Material | $W_{cb}$ (eV) | Closed IMFP / full Penn at 2, 10, 20 keV |
| --- | ---: | --- |
| Si | 16.70 (sourced) | 0.745, 0.714, 0.734 |
| Si | 20.04 (1.2×) | 0.854, 0.811, 0.827 |
| SiO₂ | 22.00 (sourced) | 0.674, 0.645, 0.660 |
| SiO₂ | 26.40 (1.2×) | 0.780, 0.731, 0.740 |
| SiO₂ | 30.80 (1.4×) | 0.880, 0.812, 0.814 |

All three Si comparisons enter the 20% band only after a shift of roughly
20% from its sourced 16.7 eV optical peak. SiO₂ needs roughly a 40% shift
from the 22 eV amorphous plasmon peak; even the cited 23.6 eV fit from a
denser sample is insufficient. Since these shifts conflict with the measured
peak positions, retuning $W_{cb}$ to the IMFP would damage the loss spectrum.
The alternative bound-shell $p_{\rm dis}(W)/W_k$ convention cannot change the
conduction-band delta contribution: $W=W_k$ there. The rate discrepancy is
therefore insensitive to that source conflict in its dominant channel.

The existing Si finite-momentum valence fit in
[`dielectric-bulk-loss.md`](dielectric-bulk-loss.md) gives another diagnostic.
Integrating its 1.12–100 eV partial spectrum on a 0.25 eV grid gives the
following path moments; GOS non-band is the closed total minus the
conduction-band oscillator. Units are Å$^{-1}$ and eV Å$^{-1}$:

| Si energy (eV) | Full Penn total rate | Dielectric valence rate | Closed GOS band rate | Dielectric valence + GOS non-band rate | Dielectric valence + GOS non-band stopping / corrected stopping |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1998.2 | 0.023529 | 0.022936 | 0.029568 | 0.024958 | 0.820 |
| 9897.1 | 0.006234 | 0.005732 | 0.008060 | 0.006399 | 0.781 |
| 19930.4 | 0.003476 | 0.003069 | 0.004331 | 0.003472 | 0.784 |

The partial finite-momentum valence rate is below the point-oscillator rate,
and the naive hybrid total falls inside the IMFP band. But its first moment
misses corrected stopping by about 18–22%; the valence fit is incomplete at
high losses, and the GOS non-band component has no validated non-overlap rule
with it. This hybrid is **not** a replacement model or a transport mode.
A justified material-general change still needs a finite-momentum response
or equivalent sourced differential construction for both materials, positive
core composition, corrected first-moment closure, and independent rate and
loss-spectrum checks.

### Total IMFP note (not a transport gate)

Owner decision, 2026-09-24: the 26–35% total-IMFP deficit above is a
documented note, not a blocker for the soft/hard transport mode. The
conduction-band distant channel carries 76–85% of the closed total rate
(Si 81–82%, SiO₂ 84–85%, MoS₂ 76–77% at the benchmark energies), and all of
it is a point loss at exactly $W_{cb}$ (16.7, 22 and 23 eV). For any cutoff
$W_c>W_{cb}$ that loss is soft: it enters transport only through the soft
first and second moments, whose sum with the hard first moment is closed to
corrected `stp.dat` exactly. PENELOPE's mixed scheme targets stopping,
straggling and hard events, not the total IMFP; the hard rate at
$W_c=50$ eV is only 9–16% of the total model rate. The note is therefore
benign **only** when $W_c>W_{cb}$; a transport mode built on this model must
reject a cutoff at or below any layer's $W_{cb}$.

`test_closed_imfp_to_full_penn_ratio_matches_recorded_note` pins the six
recorded closed/full-Penn ratios (Si 0.745, 0.714, 0.734; SiO₂ 0.674,
0.645, 0.660) to ±5×10⁻⁴, so a model change cannot move them silently.
`test_imfp_excess_channel_is_condensed_above_conduction_resonance` checks
that the conduction-band distant hard rate is zero at $W_c$ just above
$W_{cb}$, 50 eV and 100 eV. A full-Penn dielectric valence alternative was
prototyped and is not implemented. Hard-event spectra and any claim about
total inelastic mean free paths remain unvalidated by this model.

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
