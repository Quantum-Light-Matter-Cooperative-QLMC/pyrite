# `temporal-intensity-profile`

## Claim and source equation

The opt-in line temporal profile is the squared modulus of the unitary inverse transform of the coherent-emission spectral field, summed incoherently over polarization, reflection, and mosaic orientation rows:

$$
E(t)=\frac{1}{\sqrt{2\pi\hbar c}}\int E(E)\,e^{-iEt/\hbar c}\,dE,
\qquad
I(t)=\sum_{\rm rows}w_{\rm row}\,|E_{\rm row}(t)|^2,
\qquad
\int I\,dt=\int\frac{d^2N}{dE\,d\Omega}\,dE .
$$

The field $E_j(E)=c_j F_j(E)\exp\{i[\omega d_j-\mathbf g\cdot\mathbf r_j-\delta\omega L_{\rm esc,j}]\}$ is the one derived and validated under `coherent-emission`, `coherent-segment-midpoint-time`, and `coherent-formation-absorption`; this record covers only its transform to arrival time.

- **Incoherent policy.** With amplitude frozen at resonance, the inverse transform of $Q(\omega)=\int_{-t_L/2}^{t_L/2}e^{iD(\omega-\omega_r)\tau}d\tau$ is $D^{-1}e^{-i\omega_r(t-d)}$ on $|t-d|\le Dt_L/2$ and zero elsewhere. Each line is therefore a box of duration $Dt_L=2\hbar c\,a$ carrying its sinc² mass $w\pi/a$. For substepped flights, mass per piece is $\sum_{\rm pol}|c|^2\,\pi\langle e^{-\tau}\rangle/a_{\rm vac}$, using $\int|F|^2dv=\pi\langle e^{-\tau}\rangle$ from `coherent-formation-absorption`.
- **Coherent policy.** Each row's field is rebuilt on a uniform grid $E_m=E_0+m\,\delta E$ and transformed by FFT. With $t_k=t_0+k\Delta t$ and $n\,\delta E\,\Delta t=2\pi\hbar c$, $e^{-iE_mt_k/\hbar c}=e^{-iE_m t_0/\hbar c}\,e^{-iE_0k\Delta t/\hbar c}\,e^{-2\pi imk/n}$. The middle factor is one global phase per sample and drops out of $|E|^2$; the first is folded into the field as $\omega(d_j-t_0)$. Discrete Parseval is exact: $\sum_k|E(t_k)|^2\Delta t=\sum_m|E_m|^2\delta E$.
- **Bunch offsets.** For independent offsets with characteristic function $\chi$, $\langle S_eS^*_{e'}\rangle$ for $e\ne e'$ is $\chi(\omega)S^{\rm geo}_e(\omega)\,[\chi(\omega')S^{\rm geo}_{e'}(\omega')]^*$. The time-domain ensemble average is therefore $\sum_e|S_e(t)|^2+|\mathcal F^{-1}[\chi\sum_eS^{\rm geo}_e]|^2-\sum_e|\mathcal F^{-1}[\chi S^{\rm geo}_e]|^2$, whose Parseval image is the spectrum's $(1-F)G+F\,C$ blend with $F=|\chi|^2$ (`coherent-inter-electron-decoherence`).

## Assumptions

- Narrow-band: $c_j$ (amplitude, $\sqrt\omega$ prefactor, couplings) is frozen at the resonance, as in the spectrum.
- Incoherent boxes use the row-mean transmission (flat) and drop the dispersive group delay $\partial_\omega(-\delta\omega L)=\delta L$.
- The coherent profile is band-limited to the line axis; the incoherent profile carries each line's full mass.
- The FFT grid is periodic, with period twice the arrival-time span, so ringing does not wrap onto the signal.

## Limiting cases and checks

| Check | Expectation | Evidence |
|---|---|---|
| Units | $c_j^2$ in spectrum units × eV → photons/sr/e⁻; ÷ Å → × $c$ per fs | `TemporalProfile.result` |
| Parseval, incoherent | $\int I\,dt=\int{\rm spec}\,dE$ up to out-of-axis mass (2%) | `test_parseval_matches_spectrum[False]` |
| Parseval, coherent | $\int I\,dt=\int{\rm spec}\,dE$ to quadrature (2e-3) | `test_parseval_matches_spectrum[True]` |
| Single flight | box centred at $d$, rms $Dt_L/\sqrt{12}$; both routes agree to <10% L1 | `test_single_flight_is_doppler_compressed_box_at_arrival_time` |
| Two flights | two separated pulses of equal mass, empty gap | `test_two_flights_give_two_separated_pulses` |
| Long bunch | rms² adds the offset variance; coherent and incoherent yields agree | `test_bunch_offsets_spread_the_envelope` |
| Box quadrature | exact mass split across bins, zero-width pulses kept | `test_add_boxes_conserves_mass_and_support` |
| Option off | spectra, case payload, and identity unchanged | `test_temporal_does_not_change_the_spectrum`, `test_temporal_profile_opt_in_is_divergence_only` |

## Status

`anchored`: the fresh-context derivation below matches with no factor, sign, unit, or convention divergence, and the regression tests above are green. Human sign-off is tracked in #277.

## Fresh-context adjudication

A separate context that did not write the implementation verified the claim on 2026-10-01. It read the ledger row, this record's claim, the physics page, and the `coherent-formation-absorption` record for the field. It derived the results below before reading `lines/_temporal.py`, the `_per_hkl.py` hooks, `_setup.py` (`temporal_tau`, `temporal_tau_geo`, `d_all`, `d_all_geom`), and `_spectrum.py::_finalize_spectrum`.

### Independent derivation

**Transform and sign.** With $\omega=E/\hbar c$ and $dE=\hbar c\,d\omega$,

$$
E(t)=\frac{\hbar c}{\sqrt{2\pi\hbar c}}\int E(\omega)\,e^{-i\omega t}\,d\omega,
\qquad
\int|E(t)|^2dt=\frac{(\hbar c)^2}{2\pi\hbar c}\,2\pi\int|E(\omega)|^2d\omega=\int|E(E)|^2dE .
$$

The field is the Fourier image $\int E(t')e^{+i\omega t'}dt'$ of an emitter whose retarded emission time is $t'-\hat{\mathbf n}\cdot\mathbf r$, so it carries $e^{+i\omega d_j}$. The kernel $e^{-i\omega t}$ is the matching inverse, and the stationary phase of $e^{i\omega(d_j-t)}$ puts the pulse at $t=d_j$. The time ordering is therefore causal, with later segments arriving later.

**Box (no absorption).** Take $F(\omega)=\int_{-t_L/2}^{t_L/2}e^{iD(\omega-\omega_r)\tau}d\tau$. Using $\int e^{i\omega(D\tau+d-t)}d\omega=2\pi\delta(D\tau+d-t)$,

$$
\int F(\omega)e^{i\omega(d-t)}d\omega=\frac{2\pi}{D}\,e^{-i\omega_r(t-d)}\quad(|t-d|\le Dt_L/2),
\qquad
|E(t)|^2=\frac{2\pi\hbar c\,|c|^2}{D^2}.
$$

The duration is $Dt_L$. With the sinc argument $a(E-E_r)=D(\omega-\omega_r)t_L/2$, this is $Dt_L=2\hbar c\,a$. The mass is $2\pi\hbar c|c|^2t_L/D$. In energy, $\int|c|^2t_L^2\operatorname{sinc}^2[a(E-E_r)]\,dE=|c|^2t_L^2\pi/a=2\pi\hbar c|c|^2t_L/D$, which is the same. With the spectrum peak height $w=|c|^2t_L^2$ (transmission and couplings included), the mass is $w\pi/a$ and the half-width is $\hbar c\,a$.

**Absorbed piece.** The field is $c\,T\,e^{i\Phi_c}F(v)$ with $v=a_{\rm vac}(E-E_{\rm vac})+\dots$. By the record's Parseval, $\int|cTF|^2dE=|c|^2T^2\pi\langle e^{-\tau}\rangle/a_{\rm vac}$. With $\tau(s)=\tau_c+2qs$ on $s\in[-1,1]$,

$$
\langle e^{-\tau}\rangle=e^{-\tau_c}\frac{\sinh 2q}{2q}=\frac{a^2-b^2}{4q}=-\frac{(a+b)(b-a)}{4q},
$$

and the limit is $\langle e^{-\tau}\rangle\to e^{-\tau_c}=\bigl(\tfrac{a+b}{2}\bigr)^2$ as $q\to0$. The time-domain intensity inside the piece is really $\propto e^{-\tau(s)}$, so a flat box keeps the mass but not that slope; this is the documented assumption. For a straight flight, consecutive piece midpoints are separated by $\Delta d=(1-\mathbf v\cdot\hat{\mathbf n})(T_k+T_{k+1})/2$, so the boxes tile.

**FFT bookkeeping.** Put $E_m=E_0+m\,\delta E$, $t_k=t_s+k\Delta t$, and $n\,\delta E\,\Delta t=2\pi\hbar c$. Then

$$
E(t_k)=\frac{\delta E}{\sqrt{2\pi\hbar c}}\,e^{-iE_0k\Delta t/\hbar c}\sum_m\bigl[E_m\,e^{-i\omega_mt_s}\bigr]e^{-2\pi imk/n},
\qquad
I_k=\frac{\delta E^2}{2\pi\hbar c}\Bigl|\mathrm{FFT}_k\bigl[E_me^{-i\omega_mt_s}\bigr]\Bigr|^2 .
$$

The global phase drops out, $t_s$ folds into $\omega_m(d_j-t_s)$, the forward `numpy.fft.fft` kernel is the right one, and the period is $n\Delta t=2\pi\hbar c/\delta E$. Discrete Parseval gives $\sum_kI_k\Delta t=\delta E\sum_m|E_m|^2$.

**Units.** Lengths and times are in Å with $c=1$, so $I$ is photons sr⁻¹ e⁻¹ Å⁻¹. Per fs, $I_{\rm fs}=I_{\rm Å}\,c[\text{Å/fs}]$ and $t_{\rm fs}=t_{\rm Å}/c[\text{Å/fs}]$.

**Offsets.** Write $S_e=S_e^{\rm geo}e^{i(\omega A_e-B_e)}$ with iid offsets, $\chi=\langle e^{i(\omega A-B)}\rangle$, and $\langle S_e(t)\rangle=\mathcal F^{-1}[\chi S_e^{\rm geo}]$. Independence gives $\langle S_e(t)S_{e'}^*(t)\rangle=\langle S_e(t)\rangle\langle S_{e'}(t)\rangle^*$ for $e\ne e'$. Then

$$
\langle I(t)\rangle=\sum_e\langle|S_e(t)|^2\rangle+\Bigl|\mathcal F^{-1}\bigl[\chi\textstyle\sum_eS_e^{\rm geo}\bigr]\Bigr|^2-\sum_e\bigl|\mathcal F^{-1}[\chi S_e^{\rm geo}]\bigr|^2 .
$$

The realized $|S_e(t)|^2$ is a one-sample unbiased estimate of the self term. Parseval and $|S_e|=|S_e^{\rm geo}|$ give $\int I\,dt=\int[(1-|\chi|^2)\sum_e|S_e|^2+|\chi|^2|\sum_eS_e^{\rm geo}|^2]\,dE$, which is the spectrum's $(1-F)\,\text{Grouped}+F\,\text{Flat}$. For an infinite slab, $A_e=t_{0,e}-\hat{\mathbf n}_\perp\cdot\Delta\mathbf r_{\perp,e}$ and $B_e=\mathbf g_\perp\cdot\Delta\mathbf r_{\perp,e}$; a transverse shift leaves $L_{\rm esc}$ and the formation coefficients unchanged. For a Gaussian $t_0$ only, $\chi=e^{-(\omega\sigma_z)^2/2}$ and $|\chi|^2=e^{-(\omega\sigma_z)^2}$.

### Implementation diff

There is no factor, sign, unit, or convention divergence.

- `_temporal.py::TemporalProfile.result`: $t/c$ and $I\cdot c$ with `C_ANG_PER_FS` $=2997.92458$. This is correct.
- `segment_arrival_times`: $d=t_{\rm ang}+\tfrac12L/\beta(E_{\rm repr})+t_0-\hat{\mathbf n}\cdot\mathbf r_{\rm mid}$. It matches `_setup.py` `d_all` term for term (same `E_repr_keV` choice, same expanded piece rows) in float64.
- `temporal_tau_geo` $=\tau-t_0+\mathbf{xy}_0\cdot\hat{\mathbf n}_{xy}$ for an infinite slab and $\tau-t_0$ for a finite footprint. This equals `d_all_geom` $=t_{\rm mid}-\hat{\mathbf n}\cdot\mathbf r_{\rm geom}$ with $\mathbf r_{\rm geom}=\mathbf r-(\mathbf{xy}_0,0)$, respectively $\mathbf r_{\rm geom}=\mathbf r$. So $d-d_{\rm geo}=$ `decoherence_A_pop` and $g_{\rm phase}-g_{\rm phase,geo}=$ `xy0_pop @ g[:2]`, which matches `coherent_offset_chi` and `_row_decoherence_factor` ($|\chi|^2$).
- `temporal_profile_for`: $\delta E=2\pi\hbar c/(2\,\text{span}_t)$ and $\Delta t=2\pi\hbar c/(n\,\delta E)$, so the period is exactly $2\,\text{span}_t$. The window $d\pm t_L$ bounds $D t_L/2$ for $D\le2$.
- `_field_ifft_power`: forward FFT with scale $\delta E^2/(2\pi\hbar c)$. `_row_fields` uses the phase $\omega(d-t_s)-g_{\rm phase}-L\,\delta\omega$ and the full formation profile. Both match the derivation.
- `add_coherent_row` with $\chi$: per-electron fields come from `reduceat`, and the realized shift is read from the electron's first row. That is valid because the shift is constant per electron and blocks hold whole electrons. The routine forms $\text{self}+\text{cross}-\text{cross\_self}$ per polarization. This matches.
- Incoherent hook (`_per_hkl.py`, step 7): half-width `HBARC_EV_ANG * a_width` and mass `weight * pi / a_width`. Grouped hook (7b): half-width $\hbar c\,a_{\rm vac}$ and mass $\sum_{\rm pol}|c|^2\pi\langle e^{-\tau}\rangle/a_{\rm vac}$ with $c$ = `amp * t_L * A`. Both match.
- `formation_mean_transmission`: $-\text{apb}\cdot\text{bma}/(4q)$, with $(\text{apb}/2)^2$ at $q=0$. For $|q|<$ `FORMATION_SMALL_Q`, `bma` $=-2e^{-\tau_c/2}\sinh q$, so the ratio has no cancellation.
- `_finalize_spectrum`: commits at $1/N_e$, like `spec`.

### Numerical checks (independent quadrature, not the FFT path)

- $\langle e^{-\tau}\rangle$ against direct quadrature for $(\tau_s,\tau_e)=(0.3,1.7)$, $(1,1)$, $(0.2,0.2+10^{-7})$, $(2,0.1)$, $(0,30)$ agrees to $\le2\times10^{-9}$ relative, which is the reference quadrature's own error.
- A two-segment coherent row (one absorbed with $q\ne0$, complex $c$, distinct $d$ and $\mathbf g\cdot\mathbf r$) was checked through `add_coherent_row` against the directly integrated unitary transform on a 40k-point energy mesh. The pulses land at $t=d$, with intensities agreeing to $2\times10^{-4}$ (1.0329 vs 1.0327, 0.12267 vs 0.12261). Discrete Parseval agrees to $6\times10^{-5}$ (quadrature). The `add_boxes` height of the undamped segment equals $2\pi\hbar c|c_{\rm amp}|^2/D^2$ exactly (written in terms of the code's $c$ = `amp * t_L * A`, it is $\pi|c|^2/(2\hbar c\,a^2)$).
- `add_boxes` conserves mass to $10^{-14}$.
- The offset blend from `add_coherent_row` with an empirical $\chi$ integrates to the spectrum-side $(1-|\chi|^2)G+|\chi|^2\,\text{Flat}$ to $5\times10^{-14}$.
- With Gaussian $t_0$ ($\sigma=400$ Å, 3000 realizations), the kernel's mean $I(t)$ matches the brute-force ensemble mean of $|\sum_eS_e(t)|^2$ to 2.4% of peak (Monte Carlo noise of the brute force), and the integrals agree to $10^{-3}$.
- `tests/montecarlo/test_temporal_profile.py`: 13 passed.

### Minor notes (no change required)

- The incoherent box uses the bulk-root $a$ (with $\operatorname{Re}n$), so its duration is $D_{\rm bulk}t_L$ rather than the field's vacuum $D_{\rm vac}t_L$. The difference is $O(\delta)\sim10^{-6}$ relative. This choice is what makes the box Parseval-consistent with the stored incoherent spectrum.
- Box tiling of substeps is exact only for a common velocity. Energy-loss substeps (`max_dE_frac`) differ in $\beta$ by $O(\Delta E/E)$, which leaves $O(\Delta\beta\,t_L)$ gaps or overlaps between boxes. The mass is unaffected.
- With a finite footprint, the self term uses the realized $t_0$ while the cross term uses the analytic Gaussian $\sigma_z$ from `longitudinal_rms_fs`. The time-domain shape is consistent only if transport drew $t_0$ with that $\sigma_z$. The Parseval image is unaffected, as in the spectrum.
- Beyond the line axis, `delta_omega_on_profile` holds $\delta\omega$ constant (`interp` clamps) over the at most one $\delta E$ of overhang. This is negligible.

**Verdict:** `rederived`. Suggested ledger edit: Status `rederived`; replace "Fresh-context derivation pending." with "2026-10-01 fresh-context verification: matches term for term (sign/causality, box duration and mass, FFT phase bookkeeping, units, $\langle e^{-\tau}\rangle$ and its $q\to0$ limit, offset blend and its Parseval reduction, `temporal_tau_geo` ≡ `d_all_geom`)." The task owner applies the edit; human sign-off stays with #277.
