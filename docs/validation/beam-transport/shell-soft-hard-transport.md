# Shell soft/hard inelastic transport

`Validation: shell-soft-hard-transport` — status `unverified`. Physics:
[shell soft/hard transport](../../physics/beam-transport/shell-soft-hard-transport.md).
Ledger row: [`shell-soft-hard-transport`](../ledger-transport-background.md#shell-soft-hard-transport).

This record covers the opt-in `inelastic_model="shell-soft-hard"` CPU transport:
the tables built from the stopping-closed shell partition, the Numba hard-event
and soft-loss kernels, their integration into the flight scheduler and event
contract, and the macroscopic transport observables. The partition, sampler,
recoil and closure rows it builds on keep their own records.

## Unit anchors

`tests/montecarlo/test_shell_soft_hard_transport.py`:

- **Kernel parity.** For every active Si and MoS₂ hard channel at 6 and 30 keV
  and three uniform pairs, the Numba transfer inverse and primary cosine equal
  the host sampler (`shell_sampling.sample_shell_hard_collision`) to
  $10^{-10}$ and $10^{-9}$ relative.
- **Soft sampler.** 40,000 seeded draws in each PENELOPE case (truncated
  Gaussian, uniform, delta plus uniform) reproduce the mean within
  $5\sigma/\sqrt n$ and the variance within 3%. None is negative; zero
  variance returns the mean.
- **Table closure.** At every Si node from 5 to 30 keV,
  $S_s+S\sigma_h^{(1)}/\sigma^{(1)}=S$, the tabulated hard rate equals
  $S\sigma_h^{(0)}/\sigma^{(1)}$, and the channel rates sum to it, all to
  $10^{-12}$.
- **Continuous limit.** With $W_c=10^9$ eV, output rows are bitwise the
  continuous midpoint transport on the lockstep LUT, lockstep exact and
  per-electron LUT cores.
- **Event contract.** Four core/LUT/straggling combinations pass
  `check_segment_event_contract` with $W_c=50$ eV. Every hard row satisfies
  $E_{\rm end}-E_{\rm start,next}=W$ to $10^{-12}$, and
  deposit + secondary + binding $=W$.
- **Energy conservation.** Soft plus hard loss plus the final residual equals
  $E_0$ for every electron in a thick slab.
- **Reproducibility.** A fixed seed reproduces every row, and the hard keys
  are disjoint from the transport keys.
- **Rejection.** $W_c\le W_{cb}$ (Si, SiO₂), `energy_model="frozen"`, the
  CUDA LUT core, missing stopping tables, stray cutoff/material arguments and
  unknown models all raise. On CUDA the runner disables the LUT so shell
  cases reach the exact kernel.
- **CUDA (hardware-gated).** `tests/montecarlo/test_shell_soft_hard_cuda.py`
  checks replay determinism, first-row agreement with the per-electron CPU
  core (`rtol=1e-12`, with and without soft straggling, Si and MoS₂), the event
  contract and energy bookkeeping, and ensemble agreement of backscatter,
  cutoff, hard-event count and mean transfer at five sigma. Offline, both
  CUDA kernels transpile and NVRTC-compile.

The unchanged default path was also checked outside the test suite. Ninety
configurations were run before and after the change: lockstep and
per-electron cores, LUT on and off, frozen/midpoint/substepped rows, straggling
on and off, with and without SBETHE tables, single- and two-layer. All returned
arrays were bitwise identical. Six further per-electron-LUT configurations
without straggling but with SBETHE tables raised `ZeroDivisionError` before the
change. The Urban scale was computed from zero dummy element tables. They now
run, because the scale is evaluated only when Urban is sampled.

## Macroscopic observables

`checks/shell_soft_hard_transport_observables.py` (full run, 2026-09-24, local
CPU, about 4 minutes). It compares the continuous transport, continuous plus
Urban straggling, and the shell mode at $W_c=30,50,100,200$ eV (with soft
straggling) and at 50 eV without it. All runs use the same SBETHE tables,
`energy_model="midpoint"`, `max_dE_frac=0.02`, the per-electron core, 600
electrons, and seeds 11, 23, 37, 41, 53. Uncertainties are standard errors
across seeds. $R$ is the CSDA range of the table between $E_0$ and the cutoff
(1 keV at 5 and 20 keV, 5 keV at 100 keV). Geometries are a 0.05 $R$ film for
straggling, a 0.3 $R$ film for transmission and backscatter, and a 3 $R$ slab
for bulk backscatter and path length.

All 945 runs passed `check_segment_event_contract`. The largest per-electron
energy-conservation residual was $6.1\times10^{-13}$ keV.

### Stopping closure along trajectories

Realized loss $\sum(E_{\rm start}-E_{\rm end}+W)$ over every row, divided by
$\sum S(E_{\rm repr})L$ over the same rows, where $S$ is the full corrected
stopping. Terminal rows must be included. Dropping them selects against large
losses and biased an earlier version of this estimator low by up to 3% at
5 keV. Urban's cutoff truncation discards the overshoot of its last sampled
loss, which explains its slightly low ratios at 5 keV.

| material, E₀ | shell closure, W_c = 30/50/100/200 eV | no soft straggling (50 eV) | Urban |
| --- | --- | ---: | ---: |
| silicon 5 keV | 1.0011, 1.0009, 0.9962, 0.9960 (σ 0.0071) | 0.9998 ± 0.0053 | 0.9902 ± 0.0049 |
| silicon 20 keV | 0.9998, 1.0048, 1.0033, 0.9979 (σ 0.0053) | 1.0058 ± 0.0035 | 0.9935 ± 0.0028 |
| silicon 100 keV | 0.9969, 1.0011, 1.0031, 0.9979 (σ 0.0050) | 1.0012 ± 0.0050 | 1.0028 ± 0.0024 |
| sio2 5 keV | 1.0041, 1.0044, 1.0005, 0.9989 (σ 0.0075) | 1.0055 ± 0.0063 | 0.9865 ± 0.0041 |
| sio2 20 keV | 1.0027, 1.0051, 1.0002, 0.9983 (σ 0.0038) | 1.0063 ± 0.0034 | 0.9984 ± 0.0028 |
| sio2 100 keV | 0.9977, 0.9949, 0.9999, 1.0015 (σ 0.0048) | 0.9952 ± 0.0043 | 1.0019 ± 0.0042 |
| mos2 5 keV | 1.0053, 1.0027, 0.9998, 0.9947 (σ 0.0074) | 1.0030 ± 0.0065 | 0.9884 ± 0.0088 |
| mos2 20 keV | 1.0019, 1.0045, 0.9987, 1.0011 (σ 0.0067) | 1.0039 ± 0.0063 | 1.0002 ± 0.0037 |
| mos2 100 keV | 0.9984, 0.9993, 1.0011, 0.9983 (σ 0.0053) | 0.9981 ± 0.0022 | 0.9969 ± 0.0049 |

Every shell ratio is within 0.63% of unity and within 2σ. The mean hard plus
soft loss reproduces corrected `stp.dat` along real trajectories, as the table
construction requires.

### Energy-loss straggling (0.05 R film)

$\Omega^2_{\rm eff}={\rm Var}(\Delta E-\int S\,ds)/\langle{\rm path}\rangle$ over
transmitted electrons, in $10^{-4}$ keV² Å$^{-1}$, against the models at $E_0$:
the closed shell $\sigma^{(2)}$ (soft plus hard), Urban's analytic variance
(scaled to SBETHE stopping), and SBETHE's unrestricted `stp.dat` column.

| material, E₀ | closed σ⁽²⁾ | Urban | SBETHE `stp.dat` | Ω²_eff shell 50 eV | Ω²_eff Urban | shell 50 eV / closed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silicon 5 keV | 2.21 | 2.26 | 5.01 | 2.34 ± 0.53 | 2.38 ± 0.15 | 1.06 ± 0.24 |
| silicon 20 keV | 2.34 | 2.15 | 3.04 | 2.42 ± 0.21 | 1.97 ± 0.16 | 1.03 ± 0.09 |
| silicon 100 keV | 2.66 | 2.56 | 2.83 | 2.28 ± 0.22 | 2.09 ± 0.24 | 0.86 ± 0.08 |
| sio2 5 keV | 2.45 | 2.10 | 4.22 | 2.56 ± 0.46 | 1.90 ± 0.30 | 1.04 ± 0.19 |
| sio2 20 keV | 2.52 | 2.08 | 2.69 | 2.45 ± 0.41 | 1.84 ± 0.45 | 0.97 ± 0.16 |
| sio2 100 keV | 2.80 | 2.51 | 2.62 | 2.19 ± 0.36 | 2.61 ± 0.33 | 0.78 ± 0.13 |
| mos2 5 keV | 3.68 | 3.88 | 10.97 | 3.57 ± 0.45 | 4.54 ± 0.45 | 0.97 ± 0.12 |
| mos2 20 keV | 4.16 | 3.88 | 6.94 | 3.89 ± 0.46 | 3.92 ± 0.36 | 0.93 ± 0.11 |
| mos2 100 keV | 5.07 | 4.80 | 6.03 | 3.53 ± 0.51 | 4.89 ± 0.66 | 0.70 ± 0.10 |

Without Urban, the shell mode's loss fluctuation is set by its own soft and
hard second moments. The measured width matches the closed $\sigma^{(2)}$ at
5 and 20 keV. At 100 keV it is 14–30% low. The hard loss is heavy-tailed up
to $E/2$; a 600-electron sample under-represents its variance, and the rare
largest losses deflect electrons out of the transmitted set. The soft sampler
and table second moments are tested exactly in the unit anchors, so this is
a sampling and selection limit of the observable, not a demonstrated model
error. SBETHE's unrestricted column exceeds both models at 5 keV by 1.7–3×;
that is a known difference between loss models, not a transport check.

### Transmission, backscatter, transmitted spectrum and path length

$T$ and $B$ are for the 0.3 $R$ film, $\eta$ for the 3 $R$ slab, and
⟨path⟩/R is the mean path length of cutoff-stopped electrons in the 3 $R$
slab.

| material, E₀ | mode | T(0.3R) | B(0.3R) | ⟨E_T⟩ (keV) | σ(E_T) (keV) | η(3R) | ⟨path⟩/R | hard events/e |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silicon 5 keV | continuous | 0.705 ± 0.011 | 0.158 ± 0.007 | 3.63 ± 0.01 | 0.56 ± 0.01 | 0.163 ± 0.006 | 1.000 ± 0.000 | 0.0 |
| silicon 5 keV | continuous+urban | 0.685 ± 0.010 | 0.153 ± 0.005 | 3.72 ± 0.01 | 0.79 ± 0.01 | 0.165 ± 0.006 | 1.024 ± 0.005 | 0.0 |
| silicon 5 keV | shell W_c=50 | 0.676 ± 0.006 | 0.164 ± 0.005 | 3.75 ± 0.02 | 0.76 ± 0.02 | 0.179 ± 0.006 | 1.018 ± 0.006 | 4.9 |
| silicon 20 keV | continuous | 0.702 ± 0.005 | 0.151 ± 0.004 | 14.25 ± 0.04 | 2.52 ± 0.07 | 0.154 ± 0.005 | 1.000 ± 0.000 | 0.0 |
| silicon 20 keV | continuous+urban | 0.689 ± 0.006 | 0.148 ± 0.005 | 14.47 ± 0.10 | 3.11 ± 0.10 | 0.153 ± 0.004 | 1.017 ± 0.003 | 0.0 |
| silicon 20 keV | shell W_c=50 | 0.666 ± 0.007 | 0.152 ± 0.006 | 14.64 ± 0.02 | 3.05 ± 0.03 | 0.162 ± 0.008 | 1.013 ± 0.003 | 21.5 |
| silicon 100 keV | continuous | 0.729 ± 0.006 | 0.137 ± 0.005 | 71.34 ± 0.21 | 11.79 ± 0.54 | 0.140 ± 0.005 | 1.000 ± 0.000 | 0.0 |
| silicon 100 keV | continuous+urban | 0.718 ± 0.008 | 0.132 ± 0.007 | 72.34 ± 0.21 | 14.14 ± 0.16 | 0.138 ± 0.006 | 1.003 ± 0.002 | 0.0 |
| silicon 100 keV | shell W_c=50 | 0.699 ± 0.010 | 0.143 ± 0.007 | 72.03 ± 0.30 | 14.25 ± 0.35 | 0.150 ± 0.008 | 1.007 ± 0.005 | 96.3 |
| sio2 5 keV | continuous | 0.735 ± 0.007 | 0.148 ± 0.004 | 3.67 ± 0.01 | 0.55 ± 0.01 | 0.153 ± 0.003 | 1.000 ± 0.000 | 0.0 |
| sio2 5 keV | continuous+urban | 0.722 ± 0.006 | 0.142 ± 0.003 | 3.77 ± 0.01 | 0.74 ± 0.01 | 0.155 ± 0.003 | 1.020 ± 0.003 | 0.0 |
| sio2 5 keV | shell W_c=50 | 0.699 ± 0.011 | 0.143 ± 0.002 | 3.79 ± 0.02 | 0.77 ± 0.02 | 0.155 ± 0.003 | 1.020 ± 0.006 | 3.8 |
| sio2 20 keV | continuous | 0.756 ± 0.011 | 0.116 ± 0.005 | 14.59 ± 0.08 | 2.09 ± 0.14 | 0.118 ± 0.005 | 1.000 ± 0.000 | 0.0 |
| sio2 20 keV | continuous+urban | 0.748 ± 0.009 | 0.113 ± 0.004 | 14.76 ± 0.06 | 2.82 ± 0.03 | 0.120 ± 0.005 | 1.015 ± 0.005 | 0.0 |
| sio2 20 keV | shell W_c=50 | 0.713 ± 0.003 | 0.131 ± 0.007 | 14.75 ± 0.05 | 3.05 ± 0.06 | 0.141 ± 0.006 | 1.012 ± 0.005 | 15.5 |
| sio2 100 keV | continuous | 0.790 ± 0.008 | 0.103 ± 0.005 | 73.24 ± 0.28 | 10.29 ± 0.16 | 0.106 ± 0.006 | 1.000 ± 0.000 | 0.0 |
| sio2 100 keV | continuous+urban | 0.782 ± 0.011 | 0.101 ± 0.007 | 74.00 ± 0.62 | 13.01 ± 0.53 | 0.106 ± 0.008 | 1.007 ± 0.004 | 0.0 |
| sio2 100 keV | shell W_c=50 | 0.745 ± 0.004 | 0.112 ± 0.004 | 73.79 ± 0.30 | 12.94 ± 0.35 | 0.119 ± 0.004 | 1.017 ± 0.004 | 67.0 |
| mos2 5 keV | continuous | 0.417 ± 0.013 | 0.328 ± 0.013 | 3.32 ± 0.02 | 0.67 ± 0.02 | 0.329 ± 0.014 | 1.000 ± 0.000 | 0.0 |
| mos2 5 keV | continuous+urban | 0.397 ± 0.011 | 0.330 ± 0.011 | 3.53 ± 0.02 | 0.85 ± 0.02 | 0.336 ± 0.011 | 1.016 ± 0.009 | 0.0 |
| mos2 5 keV | shell W_c=50 | 0.395 ± 0.009 | 0.333 ± 0.011 | 3.47 ± 0.01 | 0.84 ± 0.02 | 0.337 ± 0.010 | 1.011 ± 0.009 | 5.6 |
| mos2 20 keV | continuous | 0.360 ± 0.007 | 0.312 ± 0.007 | 12.60 ± 0.04 | 2.91 ± 0.05 | 0.312 ± 0.006 | 1.000 ± 0.000 | 0.0 |
| mos2 20 keV | continuous+urban | 0.348 ± 0.007 | 0.326 ± 0.006 | 13.24 ± 0.13 | 3.37 ± 0.07 | 0.328 ± 0.006 | 1.003 ± 0.005 | 0.0 |
| mos2 20 keV | shell W_c=50 | 0.341 ± 0.011 | 0.320 ± 0.007 | 13.32 ± 0.10 | 3.23 ± 0.09 | 0.324 ± 0.007 | 1.002 ± 0.007 | 23.9 |
| mos2 100 keV | continuous | 0.371 ± 0.008 | 0.316 ± 0.009 | 64.13 ± 0.37 | 13.68 ± 0.10 | 0.318 ± 0.008 | 1.000 ± 0.000 | 0.0 |
| mos2 100 keV | continuous+urban | 0.373 ± 0.007 | 0.302 ± 0.009 | 66.48 ± 0.51 | 15.49 ± 0.63 | 0.304 ± 0.009 | 1.007 ± 0.005 | 0.0 |
| mos2 100 keV | shell W_c=50 | 0.349 ± 0.009 | 0.322 ± 0.012 | 65.47 ± 0.37 | 16.41 ± 0.19 | 0.325 ± 0.012 | 1.004 ± 0.002 | 104.2 |

The shell mode reproduces the straggling-driven changes that Urban makes to
the continuous transport: the transmitted spectrum is wider, its mean higher,
and the mean path 0.2–3% longer. It also adds explicit inelastic
deflection, which lowers transmission by 0.2–3.7 percentage points
relative to Urban. Bulk backscatter is equal within errors for MoS₂ and for SiO₂ at
5 keV. It is 0.008–0.023 higher than continuous transport for Si at all
three energies and for SiO₂ at 20/100 keV.
Continuous and Urban transport have no inelastic deflection at all, so this
difference is expected physics; independent backscatter or angular data are
needed before calling it an improvement.

### $W_c$ convergence

χ² of each observable across $W_c=30,50,100,200$ eV about its weighted mean
(3 degrees of freedom per entry):

| material, E₀ | T | B | ⟨E_T⟩ | σ(E_T) | η | path/R | Ω²_eff | closure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silicon 5 keV | 2.9 | 4.2 | 2.5 | 2.6 | 2.3 | 2.6 | 0.4 | 0.9 |
| silicon 20 keV | 1.0 | 1.4 | 10.5 | 1.8 | 2.6 | 0.2 | 1.7 | 2.8 |
| silicon 100 keV | 0.9 | 1.1 | 2.8 | 4.0 | 1.6 | 1.9 | 3.6 | 3.3 |
| sio2 5 keV | 0.2 | 2.5 | 1.5 | 3.4 | 4.3 | 4.3 | 0.5 | 0.9 |
| sio2 20 keV | 0.5 | 3.2 | 0.8 | 0.5 | 2.9 | 1.4 | 0.2 | 3.6 |
| sio2 100 keV | 3.6 | 2.3 | 2.2 | 3.9 | 1.6 | 1.7 | 1.0 | 2.3 |
| mos2 5 keV | 1.1 | 5.5 | 4.0 | 3.0 | 7.5 | 4.4 | 0.7 | 3.6 |
| mos2 20 keV | 0.6 | 11.7 | 1.5 | 6.6 | 9.6 | 0.9 | 1.4 | 0.5 |
| mos2 100 keV | 1.7 | 1.2 | 2.5 | 11.1 | 1.4 | 3.0 | 2.6 | 0.3 |

Total χ² = 195 for 216 degrees of freedom. The largest entries (MoS₂ 20 keV backscatter 11.7 and 9.6; MoS₂ 100 keV
transmitted width 11.1; Si 20 keV transmitted mean 10.5) are about what 72
tests with five-seed error estimates produce by chance. No observable shows a
significant monotonic trend. Soft inelastic angular deflection is omitted: a
soft collision changes energy but not direction. Its effect on backscatter
should fall as $W_c$ decreases, and it is below this study's resolution for
30–200 eV.

## Findings and limits

- Stopping closes to corrected `stp.dat` along trajectories within 0.6% at
  every material, energy and cutoff.
- The soft fraction $S_s/S$ has a 1–3% sawtooth between SBETHE nodes. It is
  inherited from linear EEDL-node interpolation in the inner-shell
  substitution, which kinks $\mathcal N(E)$. The transport LUT then reports a
  stopping interpolation error of about $2\times10^{-4}$
  (`TransportLUTToleranceWarning`).
- Omitted: soft inelastic angular deflection, PENELOPE's Eq. 4.65
  energy-dependence correction of the soft DCS, and secondary and vacancy
  transport (#94). The primary's balance closes exactly; secondary energy is
  recorded per hard row, not deposited.
- The total-IMFP note of `penelope-shell-rate-closure` carries over. Bound
  distant losses follow the adjudicated Eq. 3.94 law of
  `penelope-shell-hard-loss-sampling`, not Eq. 3.125. Every
  admissible cutoff is above $W_{cb}$.
- The exact CUDA kernel transcribes the per-electron CPU core. Its hardware
  tests pass 7/7 on an NVIDIA GeForce RTX 5080, CuPy 14.2.0; parity is
  per first row and in aggregate, not bit-for-bit along whole histories. The
  CUDA LUT core rejects the mode.
- Needs fresh-context verification. Only a human may mark the row
  `signed-off`.
