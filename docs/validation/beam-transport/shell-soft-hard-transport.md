# Shell soft/hard inelastic transport

## Independent verification (2026-09-27; derived before implementation inspection)

Source: [PENELOPE-2024, NEA/MBDAV/R(2024)1](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf), §§3.2 and 4.2, especially Eqs. 3.124–3.134, 4.44–4.47, and 4.50–4.63. This claim concerns the inelastic collision partition only; the report's Eqs. 4.46–4.47 also include a separate radiative contribution, which is absent here.

Let $d\sigma_k/dW$ be the shell-model energy-loss cross section for channel $k$, and define restricted moments $\sigma_{s,k}^{(n)}=\int_{W\leq W_c}W^n(d\sigma_k/dW)\,dW$ and $\sigma_{h,k}^{(n)}=\int_{W>W_c}W^n(d\sigma_k/dW)\,dW$. Sum over channels to obtain $M_1=\sigma_s^{(1)}+\sigma_h^{(1)}$. At one stopping-table node, the model's raw stopping is $N M_1$ in eV per length, whereas the corrected SBETHE table prescribes $S$ in the same units. Multiplying every restricted rate and moment by the single positive factor $S/(N M_1)$ gives

$$
S_s=S\frac{\sigma_s^{(1)}}{M_1},\qquad
\mu_k=S\frac{\sigma_{h,k}^{(0)}}{M_1},\qquad
\Omega_s^2=S\frac{\sigma_s^{(2)}}{M_1}.
$$

Consequently $S_s+\sum_k\mu_k\langle W\rangle_{h,k}=S$, because $\langle W\rangle_{h,k}=\sigma_{h,k}^{(1)}/\sigma_{h,k}^{(0)}$. If $S$ uses eV/Å, the dimensions are $[S_s]=\mathrm{eV/\mathring A}$, $[\mu_k]=\mathring A^{-1}$, and $[\Omega_s^2]=\mathrm{eV^2/\mathring A}$. The limiting all-soft cutoff gives $\sigma_h^{(n)}=0$, hence $S_s=S$, $\mu_h=0$, and no hard-event draw can affect the continuous trajectory. The strict $W_c>W_{cb}$ domain keeps the conduction-band delta loss below the hard threshold.

With the hazards frozen over a row, independent unit-rate exponential optical depths $\tau_{\rm el}$ and $\tau_h$ imply candidate distances $s_{\rm el}=\tau_{\rm el}/\mu_{\rm el}$ and $s_h=\tau_h/\mu_h$. The next row ends at the shortest physical candidate or geometry, energy-cutoff, or numerical-cap distance; a truncated flight consumes the appropriate optical depth. A hard event selects channel $k$ with probability $\mu_k/\mu_h$, samples $W>W_c$ conditional on that channel, and changes the primary energy by $E_{\rm next}=E_{\rm end}-W$. Both $S_s$ and $W$ are nonnegative energy losses. This is a left-endpoint hazard approximation: varying the rates along the step would require the integrated optical-depth treatment discussed around Eq. 4.65 of the source.

For a row of length $s$, Eqs. 4.51–4.56 give $m=\langle w\rangle=S_s s$ and $v=\operatorname{var}(w)=\Omega_s^2s$. These have units of energy and energy squared. The source's nonnegative artificial loss law has three branches. For $m^2>9v$, Eq. 4.59 uses a normal draw centred on $m$, with width $1.015387\sqrt v$, truncated symmetrically at $3\sqrt v$; the factor restores the variance lost to truncation. For $3v<m^2<9v$, Eqs. 4.60–4.61 make $w$ uniform on $[m-\sqrt{3v},m+\sqrt{3v}]$, with mean $m$ and variance $v$. For $m^2<3v$, Eqs. 4.62–4.63 put a delta at zero with probability $a$ and otherwise draw uniformly on $[0,w_0]$. Solving $m=(1-a)w_0/2$ and $m^2+v=(1-a)w_0^2/3$ independently gives

$$
w_0=\frac{3(m^2+v)}{2m},\qquad
a=1-\frac{4m^2}{3(m^2+v)}.
$$

The report specifies strict inequalities, leaving exact equality at $m^2=3v$ and $m^2=9v$ unstated. At $m^2=3v$, $a=0$ and $w_0=2m$, matching the adjacent uniform branch. At $v=0$, the deterministic value is $w=m$. For $m=0$, positive $v$ would be inconsistent with a nonnegative shell-loss distribution; zero mean and variance give zero loss. These filters establish dimensions, limiting behavior, and loss sign before comparison with the implementation.

### Source-to-code comparison and verdict

After the derivation above was recorded, I inspected the implementation. In `shell_transport.py`, `build_shell_inelastic_tables` computes $M_1$ as `soft1 + hard1`, adds $\ln(\sigma_s^{(1)}/M_1)$ to the logarithmic stopping table, and uses `stopping_eV_per_ang / total1` for each channel's zeroth moment and the soft second moment. Its factor `1e-6` converts the latter from eV²/Å to keV²/Å. Thus the three node formulas and their units match. `validate_shell_cutoff` enforces the strict $W_c>W_{cb}$ inequality.

In `cores.py`, independent negative logarithms initiate elastic and hard optical depths. The hard candidate is $\tau_h/\mu_h$; after a truncated row the code subtracts $s\mu_h$ from the surviving depth. It interpolates channel rates at row-start energy, draws a channel against their cumulative sum, samples $W$ and recoil at that same energy, and marks an ordinary hard row `EVENT_HARD_INELASTIC`. It applies $E_{\rm next}=E_{\rm end}-W$ after recording that row. A hard collision that leaves the primary at or below $E_{\rm cut}$ instead marks the terminal row `EVENT_CUTOFF`; its nonzero hard transfer remains recorded. The exact CUDA kernel uses the same hazard, channel, and energy update conventions. `hard_event_energy_accounting` assigns the inner-shell binding share and splits the remaining $W$ into either emitted secondary energy or local deposit, so those three shares sum to $W$.

In `hard_inelastic.py` and `_jit_shell_device.py`, the soft sampler's case boundaries are $m^2>9v$ and $m^2>3v$. Its uniform half-width is $\sqrt{3v}$; its final branch has `weight` $=(3v-m^2)/(3v+3m^2)=a$ and `top` $=3(v+m^2)/(2m)=w_0$. Both implementations return $m$ when $v=0$ and zero when $m\leq0$. The symmetric Gaussian branch uses the documented width correction and rejects draws beyond three target standard deviations. There is no divergent factor, sign, or exponent in these inspected terms.

The focused CPU anchor `tests/montecarlo/test_shell_soft_hard_transport.py` passed 19/19 tests on 2026-09-27. It includes node closure, all three sampler regimes, the bitwise all-soft limit, hard event bookkeeping, and per-electron energy conservation. The pre-existing macroscopic anchor on this page supplies additional numerical evidence across 5, 20, and 100 keV. I did not rerun its heavy sweep or the hardware-gated CUDA suite. The local official report was inspected on printed pp. 131–134 (PDF pp. 151–154) and 169–171 (PDF pp. 189–191). Its Eq. 3.124 selects a hard channel by restricted zeroth-moment weight, Eqs. 4.44–4.47 define the restricted hazard and soft moments, and Eqs. 4.59–4.63 confirm the precise $1.015387$ Gaussian width, truncation at $3\sqrt v$, and the two non-Gaussian branches. PENELOPE samples its truncated Gaussian by RITA with aliasing; PyRITE uses Box–Muller rejection for the same density. The code assigns equality at $m^2=9v$ to the uniform branch and at $m^2=3v$ to the delta-plus-uniform branch, conventions the report leaves open; all assigned branches retain the target mean and variance.

**Verdict:** `rederived` for the stopping closure, hard hazard, event energy update, and soft-loss law; no source-to-code discrepancy identified. This does not independently certify hard-transfer and recoil distributions, which have separate ledger claims, or imply human sign-off. The full docs build succeeded. The generated HTML for this page renders its math spans and display blocks, and contains no literal dollar delimiter.

`Validation: shell-soft-hard-transport` — status `rederived`. Physics:
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

## Soft inelastic angular deflection (sizing)

PENELOPE folds soft inelastic deflection into its random hinge through the
transport mean free paths of the soft angular DCS (Eqs. 4.101–4.118). PyRITE
has no hinge: it simulates every elastic collision, and PXR/CBS read each
segment's direction. `checks/soft_inelastic_deflection.py` sizes the
omission from the same closed shell model and stopping normalization the
transport uses. It evaluates $1/\lambda_{\rm in,1}^{(s)}$ from the soft
distant longitudinal DCS (recoil density $1/[Q(Q+2m_ec^2)]$ on
$[Q_-,Q'_k]$, $\mu(Q)$ from Eq. 4.101) and the soft close DCS ($Q=W$,
Eq. 3.134); distant transverse losses do not deflect. It compares that rate
with the elastic $1/\lambda_{\rm el,1}$ of the default `mott` model (NIST
SRD 64 transport cross sections; screened Rutherford with the Joy $\alpha$
for O, as transport does). The close-collision quadrature reproduces
`shell_gos`'s $\sigma^{(0)}$ to $6\times10^{-15}$.

Soft/el and hard/el are the soft and hard inelastic shares of the angular
diffusion rate ($\langle\theta^2\rangle$ per path), relative to elastic.
$\theta_{\rm row}=\sqrt{2\lambda_{\rm el}/\lambda^{(s)}_{\rm in,1}}$ is the
rms soft deflection over one elastic mean free path.

| material, E₀ | λ_el (Å) | λ_el,1 (Å) | soft/el, W_c=50 | hard/el, W_c=50 | soft/el, W_c=200 | band share of soft, 50 eV | θ_row rms, 50 eV (mrad) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silicon 5 keV | 46 | 1.96e+03 | 1.20% | 5.12% | 2.69% | 100% | 23.6 |
| silicon 20 keV | 165 | 2.21e+04 | 0.87% | 5.62% | 2.02% | 100% | 11.4 |
| silicon 100 keV | 785 | 3.78e+05 | 0.67% | 6.13% | 1.58% | 100% | 5.3 |
| sio2 5 keV | 55 | 2.21e+03 | 1.55% | 5.93% | 3.13% | 100% | 27.8 |
| sio2 20 keV | 206 | 2.58e+04 | 1.18% | 6.64% | 2.43% | 100% | 13.7 |
| sio2 100 keV | 992 | 4.69e+05 | 0.95% | 7.64% | 1.98% | 100% | 6.3 |
| mos2 5 keV | 21 | 558 | 0.51% | 2.31% | 1.11% | 95% | 19.7 |
| mos2 20 keV | 65 | 5.28e+03 | 0.35% | 2.30% | 0.76% | 95% | 9.2 |
| mos2 100 keV | 270 | 8.22e+04 | 0.26% | 2.40% | 0.56% | 95% | 4.1 |

Soft plus hard is independent of $W_c$ (Si 6.3–6.9%, SiO₂ 7.5–8.6%,
MoS₂ 2.6–2.8% of the elastic rate), as the partition requires, and near
Fano's $\sim1/Z$ estimate of atomic-electron scattering.

- Continuous transport has no inelastic deflection. It therefore
  under-diffuses directions by the whole 2.6–8.6%. The shell mode at
  $W_c=50$ eV restores 81–90% of that through explicit hard recoil.
- The remaining soft omission is 0.3–1.6% of the angular diffusion rate at
  $W_c=50$ eV, and 0.6–3.1% at 200 eV. Its effect on the depth-accumulated
  angular spread is half that, 0.1–0.8% in rms angle.
- The conduction-band loss at $W_{cb}$ carries 95–100% of the soft angular
  rate at $W_c\le50$ eV. The same channel carries the total-IMFP deficit of
  `penelope-shell-rate-closure`, so the soft share is uncertain by up to
  that deficit's 34–40% rate excess. That shifts the table's soft column,
  not its conclusion.
- Per segment the picture differs. $\theta_{\rm row}$ is 4–28 mrad, about
  10% of a typical elastic kink. Coherent resonances move by
  $\delta\omega/\omega\sim\delta\theta$, and a segment's own sinc width
  $\sim c/(\omega L)$ narrows as rows lengthen at high energy. At 100 keV a
  row's soft wander can therefore approach or exceed the per-segment line
  width, even though the ensemble spread barely moves. Whether that matters
  for PXR/CBS line shapes needs an emission-level comparison. It can't be
  settled from transport moments.

### Line-spectrum response (emulated)

`checks/soft_deflection_line_sensitivity.py` ran on an RTX 5080 (job 19,
2026-09-25). It used Si catalog cases at 30 and 100 keV, 1000 Å and 1 µm,
tilt 30°, with 20000 electrons and six seeds per case, at $W_c=50$ eV. Each
seed was transported once. The incoherent line spectrum was then evaluated on
the same segments with the transported directions and with two emulations of
the omitted soft deflection:

- **vertex:** a cumulative per-electron random walk at the soft rate, i.e.
  the extra angular diffusion;
- **row:** each segment tilted by the mean deviation that a within-row soft
  random walk produces. Folding the deflection into existing vertices would
  miss this.

Paired differences are compared with the seed-to-seed SD at 20000 electrons.

- **vertex, $k=1$:** every line's yield, centroid and rms width moves by at
  most 0.35%. The largest shifts are about 3 SD (100 keV, 1 µm width,
  +0.22%). The extra diffusion is negligible.
- **row, $k=1$:** the lowest-energy feature (~42 eV) at 30 keV widens 2.5–3.4%
  and loses 1.4% of its yield, 6–8 SD. PXR lines move by 0.1–1.3% in width
  and up to 0.9 eV in centroid, up to about 7 SD. The largest is a 2.7% width
  change at 100 keV, 1 µm (2.4 SD).
- **$k=4$** (a bound on the conduction-band rate uncertainty): row effects
  grow to about 10% on the ~42 eV feature and 1–2% on PXR lines. Vertex
  effects stay below 1%.

The row emulation is crude. It tilts the whole segment rather than bending
it, and substep rows of one flight are tilted independently. It still shows
that the within-segment direction wander, not the added diffusion, is the
part that reaches line shapes at the percent level. Folding the soft
deflection into vertices would therefore not capture it. Settling it needs
discrete soft angular events that split rows, whose rate is the
conduction-band rate carrying the total-IMFP deficit.

## Findings and limits

- Stopping closes to corrected `stp.dat` along trajectories within 0.6% at
  every material, energy and cutoff.
- The soft fraction $S_s/S$ has a 1–3% sawtooth between SBETHE nodes. It is
  inherited from linear EEDL-node interpolation in the inner-shell
  substitution, which kinks $\mathcal N(E)$. The transport LUT then reports a
  stopping interpolation error of about $2\times10^{-4}$
  (`TransportLUTToleranceWarning`).
- Omitted: soft inelastic angular deflection (sized above: 0.3–1.6% of the
  angular diffusion rate at $W_c=50$ eV), PENELOPE's Eq. 4.65
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
