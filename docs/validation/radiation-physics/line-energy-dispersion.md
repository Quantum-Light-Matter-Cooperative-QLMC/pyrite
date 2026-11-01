# line-energy-dispersion

Fresh-context re-derivation (2026-10-05, #338). This replaces the earlier write-up, which derived the resonance with an $\exp[i(\mathbf k\cdot\mathbf r-\omega t)]$ outgoing wave. The module's coefficients are defined in the opposite, $\exp(+i\omega t)$ convention. That mix left the coupling pairing unresolved.

The claim has two parts:

1. **Line energy.** $\omega_{\rm res}=\mathbf v\cdot\mathbf g/(1-\hat{\mathbf n}\cdot\mathbf v)$ and $E=\hbar c\,\omega$, with $\omega$ and $\mathbf g$ in $\mathring{\mathrm A}^{-1}$ and $\mathbf v$ in units of $c$.
2. **Coupling pairing.** The line resonant on $\mathbf g=\mathbf g(hkl)$ couples through $\chi_{\rm line}=\overline{\chi(-hkl)}$ and $U_{\rm line}=\overline{U(-hkl)}=U(hkl)$ in kernels that sum the conjugate field $\exp\{i[\omega t-(\mathbf k+\mathbf g)\cdot\mathbf r]\}$.

The derivation below came first. It used only the ledger row, the claim statement, and the docstrings of `refractive_index`, `optical_constants`, `chi_g` and `reciprocal_g_vector`. The implementation bodies were read afterwards.

## Conventions taken as given

- **Time factor.** Fields are $\propto\exp(+i\omega t)$, so a plane wave is $\exp\{i[\omega t-\mathbf k\cdot\mathbf r]\}$. The index $n=1-\delta-i\beta$ with $\beta>0$ then gives $\exp(-\beta\omega z)$ decay along $+z$. Since $\chi_0=n^2-1\approx-2\delta-2i\beta$, an absorbing medium has $\operatorname{Im}\chi<0$ in this convention.
- **Atomic factor.** $f=f_0+f'+if''$ with $f''>0$ (Henke/Chantler), and $\chi\propto-(f_0+f'+if'')$. Hence $\operatorname{Im}\chi\propto-f''<0$, consistent with the time factor above.
- **Reciprocal vector.** `reciprocal_g_vector` returns $\mathbf g(hkl)=hkl\cdot B$, whose rows carry the $2\pi$. So $\mathbf g\cdot\mathbf R=2\pi\,hkl\cdot\mathbf R_{\rm frac}$ and $|\mathbf g|=2\pi/d_{hkl}$.
- **Crystallographic coefficient.** The ledger and the `chi_g` docstring give

$$
\chi(hkl)=-\frac{r_e\lambda^2}{\pi V}\sum_j\left(f_0+f'+if''\right)_j
\exp\!\left(+2\pi i\,hkl\cdot\mathbf R_j\right)e^{-W_j}.
$$

With $\chi(hkl)=V^{-1}\int_V\chi(\mathbf r)\exp(+i\mathbf g_{hkl}\cdot\mathbf r)\,d^3r$, the coefficient's spatial reconstruction is

$$
\boxed{\chi(\mathbf r,\omega)=\sum_{hkl}\chi(hkl)\exp(-i\mathbf g_{hkl}\cdot\mathbf r)}.
$$

Because $\chi(\mathbf r)$ is complex when $f''\neq0$, $\chi(-hkl)\neq\overline{\chi(hkl)}$ in general (Bijvoet pairs). The potential is real, so $U(-hkl)=\overline{U(hkl)}$ exactly.

## Independent derivation

### Source spectrum of a uniformly moving charge

Take $\mathbf r_e(t)=\mathbf r_0+\mathbf v t$ and $c=1$. In the $\exp(+i\omega t)$ convention, $J(\mathbf r,\omega)=\int dt\,J(\mathbf r,t)e^{-i\omega t}$ and $J(\mathbf r,\omega)=\int\frac{d^3q}{(2\pi)^3}J_{\mathbf q}e^{-i\mathbf q\cdot\mathbf r}$. Then

$$
J_{\mathbf q}(\omega)=e\mathbf v\int dt\,e^{i\mathbf q\cdot\mathbf r_e(t)-i\omega t}
=2\pi e\mathbf v\,e^{i\mathbf q\cdot\mathbf r_0}\,\delta(\omega-\mathbf q\cdot\mathbf v).
$$

The co-moving Coulomb field is therefore a superposition of modes $\exp\{i[\omega t-\mathbf q\cdot\mathbf r]\}$ with $\omega=\mathbf q\cdot\mathbf v$. As a check, $f(\mathbf r-\mathbf v t)=\sum_{\mathbf q}f_{\mathbf q}\exp\{i[(\mathbf q\cdot\mathbf v)t-\mathbf q\cdot\mathbf r]\}$. Each mode amplitude $\mathbf E_{\mathbf q}$ is a convention-fixed factor ($\pm i$) times real kinematics. That factor is common to every term below.

### PXR: which harmonic radiates at the line

The induced polarization is $\mathbf P(\mathbf r,\omega)=\chi(\mathbf r,\omega)\mathbf E(\mathbf r,\omega)/4\pi$. A Coulomb mode times one harmonic gives

$$
\chi(h)\,e^{-i\mathbf g_h\cdot\mathbf r}\;\mathbf E_{\mathbf q}e^{-i\mathbf q\cdot\mathbf r}
=\chi(h)\,\mathbf E_{\mathbf q}\exp[-i(\mathbf q+\mathbf g_h)\cdot\mathbf r].
$$

This is a source with wavevector $\mathbf k=\mathbf q+\mathbf g_h$ at frequency $\omega=\mathbf q\cdot\mathbf v$. The outgoing $\exp(+i\omega t)$ Green function in the far field is $\propto e^{-i\omega|\mathbf x|}/|\mathbf x|$ times $\int d^3r\,e^{+i\mathbf k\cdot\mathbf r}J(\mathbf r,\omega)$, with $\mathbf k=\omega\hat{\mathbf n}$. It selects this source when $\mathbf q=\mathbf k-\mathbf g_h$, so

$$
\omega=(\mathbf k-\mathbf g_h)\cdot\mathbf v
\quad\Longrightarrow\quad
\omega=\frac{-\mathbf v\cdot\mathbf g_h}{1-\hat{\mathbf n}\cdot\mathbf v}.
$$

The line at $\omega_{\rm res}=+\mathbf v\cdot\mathbf g(hkl)/(1-\hat{\mathbf n}\cdot\mathbf v)>0$ is therefore produced by $\mathbf g_h=-\mathbf g(hkl)$, that is $h=-hkl$. The virtual photon is $\mathbf q=\mathbf k+\mathbf g$ and the photon receives momentum transfer $-\mathbf g$. The coefficient is the $e^{+i\mathbf g\cdot\mathbf r}$ harmonic of $\chi(\mathbf r)$, which is $\chi(-hkl)$.

For a finite straight flight, the same projection reads

$$
A_{\rm PXR}(\omega,\mathbf k)\propto\chi(-hkl)\,\mathcal K_{\rm PXR}
\int dt\,\exp\{-i[\omega t-(\mathbf k+\mathbf g)\cdot\mathbf r_e(t)]\},
$$

where $\mathcal K_{\rm PXR}$ is the polarization-projected kinematic factor. Here it is the Feranchuk Eq. (13) bracket over the real detuning $|\mathbf k+\mathbf g|^2-\omega^2$, times the common $\pm i$. Stationary phase in $t$ reproduces $\omega=(\mathbf k+\mathbf g)\cdot\mathbf v$, the same resonance.

### CBS: same selection

The electron is accelerated by $-\nabla\varphi$ with $\varphi(\mathbf r)=\sum_hU(h)e^{-i\mathbf g_h\cdot\mathbf r}$. Along the path, the $h$ term oscillates as $\exp[-i\mathbf g_h\cdot\mathbf r_e(t)]$. The radiation integral $\int dt\,e^{-i\omega t+i\mathbf k\cdot\mathbf r_e}\,\dot{\mathbf v}(t)$ again selects $\omega=(\mathbf k-\mathbf g_h)\cdot\mathbf v$. So the same line takes $U(-hkl)$:

$$
A_{\rm CBS}\propto U(-hkl)\,\mathcal K_{\rm CBS}
\int dt\,\exp\{-i[\omega t-(\mathbf k+\mathbf g)\cdot\mathbf r_e(t)]\}.
$$

PXR and CBS of one line share the transfer $-\mathbf g$, the same $h=-hkl$ and the same exponential. Their relative real sign is the content of `cbs-amplitude` and is not adjudicated here.

### The conjugate-field kernel

The kernels sum $\exp\{+i[\omega t-(\mathbf k+\mathbf g)\cdot\mathbf r]\}$, the complex conjugate of the exponential above. Conjugating the whole amplitude, with real $\mathcal K$ up to the common factor, gives

$$
\overline{A}\propto\Bigl[\overline{\chi(-hkl)}\,\mathcal K_{\rm PXR}
+\overline{U(-hkl)}\,\mathcal K_{\rm CBS}\Bigr]
\int dt\,\exp\{+i[\omega t-(\mathbf k+\mathbf g)\cdot\mathbf r_e(t)]\},
$$

and $|\overline{A}|^2=|A|^2$. Hence

$$
\boxed{\chi_{\rm line}=\overline{\chi(-hkl)},\qquad
U_{\rm line}=\overline{U(-hkl)}=U(hkl)}
$$

This matches the claim exactly.

**Only the relative conjugation matters.** Because $\mathcal K$ is real, the pair $\{\chi(-hkl),U(-hkl)\}$ (unconjugated, in the native $\exp\{-i[\ldots]\}$ form) gives the identical $|A|^2$. The physical content is therefore that PXR and CBS are both taken at $-hkl$ with the same conjugation status. The *mixed* pairs $\{\chi(-hkl),U(hkl)\}$ and $\{\overline{\chi(-hkl)},U(-hkl)\}$ are wrong, as is the historical $\{\chi(hkl),U(hkl)\}$.

### Cross-check in the opposite time convention

With $\exp(-i\omega t)$ the physical response is $\chi_{\rm phys}(\mathbf r)=\overline{\chi(\mathbf r)}$, and the atomic factor reads $f_0+f'-if''$ in that convention. Coulomb modes are $e^{+i\mathbf q\cdot\mathbf r}$, and the $e^{+i\mathbf G\cdot\mathbf r}$ harmonic radiates at $\omega=-\mathbf v\cdot\mathbf G/(1-\hat{\mathbf n}\cdot\mathbf v)$, so the line takes $\mathbf G=-\mathbf g$. The $e^{-i\mathbf g\cdot\mathbf r}$ harmonic of $\overline{\chi(\mathbf r)}$ is $\overline{\chi(-hkl)}$, and the natural phase is $\exp\{i[(\mathbf k+\mathbf g)\cdot\mathbf r-\omega t]\}$. The modulus is the same, so the result does not depend on the time convention. The earlier write-up instead paired an $\exp(-i\omega t)$ phase with a coefficient defined with $+if''$, which is the convention mix.

### Origin translation

Under $\mathbf R_j\to\mathbf R_j+\mathbf a$, $\overline{\chi(-hkl)}\to\overline{\chi(-hkl)}\,e^{+i\mathbf g\cdot\mathbf a}$ and $U(hkl)\to U(hkl)\,e^{+i\mathbf g\cdot\mathbf a}$. The kernel phase picks up $e^{-i\mathbf g\cdot\mathbf a}$ (plus a global $e^{-i\mathbf k\cdot\mathbf a}$), so $|A|^2$ is origin-independent and the PXR–CBS relative phase is preserved. The old pairing is also translation-covariant, so this check does not discriminate between pairings. It only confirms that the coefficient and the $\mathbf g\cdot\mathbf r$ phase share one origin.

## Cheap filters

| check | result |
| --- | --- |
| units | $\mathbf v\cdot\mathbf g$ in $\mathring{\mathrm A}^{-1}$, denominator dimensionless, $E=\hbar c\,\omega$ with `HBARC_EV_ANG` $=1973.269804$ eV Å: pass |
| denominator | $1-\hat{\mathbf n}\cdot\mathbf v\geq1-\lvert\mathbf v\rvert>0$ for $\lvert\mathbf v\rvert<1$: pass |
| $v\to0$, $\mathbf v\perp\mathbf g$ | $\omega\to0$: no line. Pass |
| $\mathbf g\to-\mathbf g$ | numerator flips. The $-hkl$ line radiates only when $\mathbf v\cdot\mathbf g(hkl)<0$, and then couples to $\overline{\chi(hkl)}$. The two members of a Friedel pair swap roles consistently. Pass |
| $f''=0$ | $\chi(-hkl)=-C\sum(f_0+f')e^{-i\mathbf g\cdot\mathbf R}e^{-W}$, so $\overline{\chi(-hkl)}=\chi(hkl)$ exactly. Reduces to the crystallographic coefficient: pass |
| centrosymmetric, $f''\neq0$ | About the inversion centre $\chi_c(-hkl)=\chi_c(hkl)$ and $U_c$ is real. Then $\lvert\overline{\chi_c}\mathcal K_P+U_c\mathcal K_C\rvert=\lvert\chi_c\mathcal K_P+U_c\mathcal K_C\rvert$. A shifted origin adds one common phase (above). $\lvert A\rvert^2$ is unchanged from the old pairing: pass |
| noncentrosymmetric, $f''\neq0$ | $\lvert\chi(hkl)\rvert\neq\lvert\chi(-hkl)\rvert$, so the pairing is observable in PXR. Bijvoet sensitivity: expected |
| opposite tilts | Only $\mathbf v\cdot\mathbf g$ and $\hat{\mathbf n}\cdot\mathbf v$ enter $\omega$. For the default beam-aligned $\mathbf g\parallel$ normal geometry, $\mathbf v\cdot\mathbf g=\beta\lvert\mathbf g\rvert\cos\alpha$ is even in $\alpha$, so the line energy is even. Pairing is tilt-independent. Pass |

## Numeric toy check (implementation-free)

A 1-D noncentrosymmetric chain makes the selection rule concrete. It has period $a=3$ Å and Gaussian atoms ($w=0.15$ Å) at $x=0$ and $x=0.27a$ with $f=10+2i$ and $6+0.5i$, so $\chi(x)\propto-\sum f_j\,G(x-x_j)$. The test uses $v=0.6$, forward $\hat{\mathbf n}=+\hat x$, $g=2\pi/a$ and $\omega=vg/(1-v)$. Over 400 periods, the integral $A/T=T^{-1}\int dt\,e^{-i\omega t+ikx(t)}\chi(x(t))$ was compared with the crystallographic coefficients $\chi(h)=a^{-1}\int\chi(x)e^{+2\pi ihx/a}dx$:

```text
A/T           (-1.16243378 + 0.47901851j)
chi(+1)       (-1.04407779 - 0.94125340j)
chi(-1)       (-1.16243378 + 0.47901851j)
conj(A)       (-1.16243378 - 0.47901851j)  = conj(chi(-1))
<exp{i[wt-(k+g)x]}>  1 - 4e-13j        (kernel phase is stationary)
|chi(+1)|, |chi(-1)|   1.4057, 1.2573
```

The line at $+vg/(1-v)$ is driven by $\chi(-1)$ to $10^{-12}$, and the conjugate-field coefficient is $\overline{\chi(-1)}$. In this toy, the Bijvoet moduli differ by 12%.

## Implementation comparison

- **Resonance.** `montecarlo/spectrum/cxr/_kernels.py::_cxr_kin_core` computes `omega_res = v_dot_g / denom` and the detuning `g2 + 2*k_dot_g`, which equals $|\mathbf k+\mathbf g|^2-\omega^2$. It also uses `v_dot_kg = v_dot_g + k_dot_v`, which equals $(\mathbf k+\mathbf g)\cdot\mathbf v=\omega$ at resonance. So the virtual photon is $\mathbf q=\mathbf k+\mathbf g$ and the photon's transfer is $-\mathbf g$, matching the derivation. `_per_hkl._accumulate_reflection` step 1 uses the same $+\mathbf v\cdot\mathbf g$ numerator, with the in-medium denominator of `xray-in-medium-resonance` (not part of this claim). Its step 2 discards nonpositive roots. `_anchor_conditions.py::line_energy_eV` evaluates $\hbar c\,\beta|\mathbf g|/(1-\beta\cos\theta_{\rm obs})$, the same formula for $\mathbf v\parallel\mathbf g$.
- **Kernel phase.** `_accumulate_reflection_coherent` builds `arg = d * omega_grid - g_phase` with `d` $=t_{\rm abs}-\hat{\mathbf n}\cdot\mathbf r$ and `g_phase` $=\mathbf g\cdot\mathbf r$, then multiplies by `exp(1j * arg)`. That is $\exp\{i[\omega t-(\mathbf k+\mathbf g)\cdot\mathbf r]\}$, the conjugate-field form assumed above.
- **Coupling tables.** `materials/crystal.py::emission_coupling_tables` calls `reflection_coupling_tables` at `-hkl` and returns `(chi_re, -chi_im, u_re, -u_im)`, i.e. $\overline{\chi(-hkl)}$ and $\overline{U(-hkl)}$. `reflection_coupling_tables` accumulates $P_{\rm el}=\sum e^{+2\pi i\,hkl\cdot\mathbf R}$ with $f'+if''$ for $\chi$ and $Z-f_0-f'$ (no $f''$) for $U$, matching the coefficient definitions above. $U(-hkl)$ is therefore exactly $\overline{U(hkl)}$, so $\overline{U(-hkl)}=U(hkl)$.
- **Kernels.** `_kernels.py::_reflection_tables` now delegates to `emission_coupling_tables`. It is the only coupling source for both routes (`_per_hkl.py` line 838, `_batched.py` line 256). Step 3 interpolates the tables at the segment's $E_r$. Step 5 forms `A_PXR = chi/detuning*(real)` and `A_CBS = -eUg_over_m/(gamma*vdg)*(real)`, and the fused incoherent `_cxr_amp_sq_core` uses the same real $f_{\rm pxr}$ and $f_{\rm cbs}$. This matches $\overline{\chi(-hkl)}\mathcal K_{\rm PXR}+U(hkl)\mathcal K_{\rm CBS}$.
- **Oracle.** `validation/feranchuk_spence.py` (`amplitudes_PXR_CBS_both`, `amplitudes_PXR_CBS_sweep`) uses `chi = conj(chi_g(mate))` and `eUg = conj(U_g(mate))` with `mate = -hkl`, with $\mathbf k_g=\mathbf k+\mathbf g$ and $\mathbf g\cdot\mathbf v_0>0$. This is the same pairing, so the oracle is consistent but shares the convention rather than testing it independently.

Symbolically, every factor, sign and conjugation agrees with the independent result. No diverging term was found.

## Anchors

Run with `PYRITE_MC_BACKEND=cpu pyrite-dev test tests/montecarlo/spectrum/test_line_friedel_pairing.py tests/materials/test_crystallography.py -k "friedel or emission_coupling"`: **7 passed**.

- `tests/montecarlo/spectrum/test_line_friedel_pairing.py` point-reflects one 4H-SiC $(102)$ segment ($\mathbf v,\hat{\mathbf n},hkl\to-\mathbf v,-\hat{\mathbf n},-hkl$) at tilts $-12^\circ,0^\circ,+12^\circ$, with a vacuum index and line energies 2531, 2883 and 3182 eV near the Si K edge. It checks the closed-form peak, CBS invariance, and $\sum{\rm PXR}_-/\sum{\rm PXR}_+=|\chi(hkl)|^2/|\chi(-hkl)|^2$.
  - **Discriminating.** The new code gives ratios 0.66986, 0.73434 and 0.77384, matching the expected values to $10^{-7}$. With `emission_coupling_tables` swapped back to `reflection_coupling_tables` (the old $\chi(+hkl)$ pairing), the ratios are 1.4928, 1.3618 and 1.2923, the inverses. The test fails by 50–120%, well outside its `rtol=2e-3`.
  - **Independent?** Partly. The expected ratio uses the module's `chi_g`. That is legitimate here because the claim is about which index is paired, not the value of $\chi$, and the expectation is taken from the derivation rather than the kernel. The line-energy check is an independent closed form.
  - **Blind spot.** The test compares PXR and CBS components only, not the PXR–CBS interference. It cannot detect a *mixed* conjugation, for example $\{\chi(-hkl),U(hkl)\}$. A scratch run with that mixed table left PXR and CBS bit-identical but cut the total line yield by a factor of 3.4 (e.g. $3.35\times10^{-9}\to9.96\times10^{-10}$ at $-12^\circ$). The jointly unconjugated $\{\chi(-hkl),U(-hkl)\}$ reproduced the total bit-for-bit, as derived.
- `tests/materials/test_crystallography.py::test_emission_coupling_tables_are_the_conjugate_friedel_mate` pins the tables to $\overline{\chi(-hkl)}$ and $U(hkl)/m_ec^2$ at $10^{-12}$ for three 4H-SiC reflections over 2–30 keV, and asserts a >10% modulus difference from $\chi(+hkl)$. This is the only anchor that catches the mixed-conjugation error above. It restates the formula through the per-atom `chi_g` and `U_g` rather than through physics.
- `test_emission_coupling_tables_reduce_to_crystallographic_without_fpp` checks the $f''=0$ limit bit-for-bit on HOPG, hBN and LiF.

## Secondary findings (not blocking)

1. **Interference blind spot in the regression.** As above, the integration anchor does not constrain the relative PXR/CBS conjugation. A total-yield assertion against an independently assembled $|\overline{\chi(-hkl)}\mathcal K_P+U(hkl)\mathcal K_C|^2$, or comparing `spec` with the oracle, would close it.
2. **Convention-dependent wording in `_anchor_conditions.py::line_energy_eV`.** The docstring says the repository $\mathbf g$ "labels the line by minus its `exp(+i g.r)` harmonic". That is true in an $\exp(-i\omega t)$ field convention. In the module's documented $\exp(+i\omega t)$ convention, the line is the $e^{+i\mathbf g\cdot\mathbf r}$ harmonic of $\chi(\mathbf r)$. The appended "momentum transfer $-\mathbf g$" is unambiguous and correct, so stating only that, or naming the convention, avoids repeating the original mix. The `tilted_geometry` docstring says only "momentum transfer `-g`", which is fine.
3. **`coherent-emission` write-up.** Its phase result $\exp\{i[\omega(t-\hat{\mathbf n}\cdot\mathbf r)-\mathbf g\cdot\mathbf r]\}$ is convention-free and agrees with this derivation. Its coefficient statement ("$S(+\mathbf g)$ reconstructs as $\chi_{\mathbf g}e^{-i\mathbf g\cdot\mathbf r}$") is correct as a reconstruction, but it should not be read as the line's coupling when $f''\neq0$.
4. **Oracle independence.** The Feranchuk–Spence oracle now applies the same `mate`/`conj` pairing internally. It confirms implementation consistency but is not an independent check of the pairing.

## Adjudication

**rederived.** The independent derivation, written in the module's own $\exp(+i\omega t)$ / $f'+if''$ convention, gives $\omega_{\rm res}=+\mathbf v\cdot\mathbf g/(1-\hat{\mathbf n}\cdot\mathbf v)$ for transfer $-\mathbf g$. For the conjugate-field kernel it gives $\chi_{\rm line}=\overline{\chi(-hkl)}$ and $U_{\rm line}=U(hkl)$, the same as `emission_coupling_tables`, `_reflection_tables`, the step-5 amplitudes and the oracle. The historical $\chi(+hkl)$ pairing was wrong in modulus for noncentrosymmetric crystals with $f''\neq0$, and the fix in #338 is correct. The earlier numerator-sign "discrepancy" is resolved as a pure relabelling: in the $\exp(+i\omega t)$ convention, the line on $+\mathbf g$ is driven by the $e^{+i\mathbf g\cdot\mathbf r}$ harmonic of $\chi(\mathbf r)$.

## Owner remediation after the verdict

Added by the task owner, not the verifier.

- Finding 1 is closed by `tests/montecarlo/spectrum/test_line_friedel_pairing.py::test_total_yield_matches_the_unconjugated_native_pairing`. It feeds the kernel the jointly unconjugated native pair $\{\chi(-hkl),U(-hkl)\}$ and requires the production total at `rtol=1e-9`. A mutant returning the mixed $\{\overline{\chi(-hkl)},U(-hkl)\}$ fails it.
- Finding 2 is fixed: the `line_energy_eV` docstring and the dependent notes now state only the momentum transfer $-\mathbf g$.

## Deterministic photon-energy API: independent derivation (#469)

Fresh-context verifier, 2026-10-12. This section was written from the ledger,
public signature and derivation docstrings before reading implementation bodies.
The scope is the vacuum, straight-beam API `materials.emission.photon_energies`,
not an intensity prediction or an in-medium resonance.

For electron kinetic energy $T$, let $\gamma=1+T/(m_ec^2)$ and
$\beta=\sqrt{1-\gamma^{-2}}$. Phase matching of a straight electron with the
outgoing wave and reciprocal transfer $-\mathbf g$ gives
$\omega=(\omega\hat{\mathbf n}+\mathbf g)\cdot\boldsymbol\beta$.
Because $1-\beta\cos\theta_{\rm obs}>0$, choosing the positive-projection
Friedel mate yields

$$
E_\gamma=\hbar c\,\frac{\beta\lvert\hat{\mathbf z}\cdot\mathbf g_{\rm lab}\rvert}
{1-\beta\cos\theta_{\rm obs}}.
$$

Here $\mathbf g$ includes $2\pi$ and carries inverse angstroms; $\hbar c$
carries eV angstroms. Detector azimuth cancels because the beam is lab $+z$.
As $T\to0$, $E_\gamma\to0$; zero projection gives no positive-frequency line.
The API's NaN at zero projection and at $E_\gamma>T$ is a selection policy:
the classical resonance alone does not enforce recoil or photon energy conservation.
The absolute value denotes mate selection for energy only; it must not be used
to infer equal intensities of Bijvoet pairs with anomalous scattering.

Let $O$ carry the catalogue direct cut axis or reciprocal surface normal to
sample $+z$. For reciprocal cuts, $\mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij}$
proves that $\sum h_i\mathbf b_i$ is normal to the Miller plane, including
nonorthogonal cells. For unit cut normal $\mathbf u$, the minimal proper
rotation has axis $\mathbf u\times\hat{\mathbf z}$ and angle
$\arccos(\mathbf u\cdot\hat{\mathbf z})$; the antiparallel case requires a
proper half-turn, not an inversion. Define the tilted sample rotation
$S=R_z(\varphi)R_y(\alpha)R_z(-\varphi)$. Then $S\hat{\mathbf z}$ points toward
positive lab $x$ at positive $\alpha$ and zero $\varphi$, and

$$
\hat{\mathbf z}\cdot\mathbf g_{\rm lab}
=-\sin\alpha\cos\varphi\,(O\mathbf g)_x
-\sin\alpha\sin\varphi\,(O\mathbf g)_y
+\cos\alpha\,(O\mathbf g)_z.
$$

A normal-aligned reflection therefore gives $\lvert\mathbf g\rvert
\lvert\cos\alpha\rvert$, independent of tilt azimuth; an off-normal reflection
retains its azimuth dependence and can distinguish the tilt sign.

For default-plane selection, group fractional basis sites by element:

$$
P_s(hkl)=\sum_{j\in s}\exp(2\pi i\,hkl\cdot\mathbf R_j),\qquad
F(hkl,E)=\sum_s f_s(hkl,E)P_s(hkl).
$$

If every $P_s$ vanishes, extinction is independent of atomic factors and photon
energy. Cancellation only between different species is not a systematic
extinction. A per-element relative amplitude tolerance of $10^{-5}$ is a
numerical policy, not a universal physical threshold. Searching the first 32
positive harmonics of the primitive cut is finite default-selection policy;
explicit Miller triples may still give kinematic energies for extinct planes.

### API comparison and evidence

After the independent expression above was recorded, implementation inspection
found no divergent sign, factor, unit or composition order. `photon_energies`
uses the derived sample projection and a stable identity for the denominator:

$$
1-\beta\cos\theta
=\frac{\gamma^{-2}}{1+\beta}+2\beta\sin^2(\theta/2).
$$

Its mask implements the stated $10^{-12}$ relative grazing tolerance and
$0<E_\gamma\leq T$. `_primary_hkl` reduces the cut by its integer gcd, takes
the lowest squared-index-norm pinned parallel family, or visits harmonics
1--32 and retains the first with any element-group amplitude greater than
$10^{-5}$ times that element's site count. This matches the derived
energy-independent extinction criterion under the stated full-occupancy
assumption. Pinned families are authoritative and deliberately bypass the
extinction estimate. Explicit triples bypass default selection.
`materials.crystal._orientation_R` composes the derived proper minimal rotation
and right-handed roll; `montecarlo.geometry._orientation_R` forwards every
argument to it. The half-turn branch of `_rotation_between` is proper.

Independent numeric construction used the catalogue's six cell parameters,
a Cartesian direct basis, $2\pi A^{-\mathsf T}$, and an independently assembled
Rodrigues matrix; it used no PyRITE geometry or reciprocal helpers. At 80 keV,
polar tilt $23^\circ$, tilt azimuth $37^\circ$, and lab observation polar angle
$119^\circ$, the following results were obtained with `uv run python` through
the project environment:

| material | reflection | independent energy (eV) | API energy (eV) |
| --- | --- | --- | --- |
| HOPG | $(101)$ | 224.404512716 | 224.404512747 |
| HOPG | $(002)$ | 1374.087449302 | 1374.087449487 |
| monoclinic GeP | $(101)$ | 111.333920974 | 111.333920989 |
| monoclinic GeP | $(002)$ | 654.911097203 | 654.911097292 |
| silicon | $(101)$ | 921.723869507 | 921.723869631 |
| silicon | $(002)$ | 713.730917372 | 713.730917469 |

All six agree within $1.35\times10^{-10}$ relative; independently supplied
physical constants account for the common residual. GeP tests the nonorthogonal
cut rather than a direct-axis approximation. HOPG $(002)$ at grazing tilt gives
NaN; its 1 eV incident-energy resonance also gives NaN. GeP opposite $(101)$
Friedel mates return exactly equal energies. The silicon extinct $(002)$
example deliberately confirms that explicit overrides compute kinematics.

Source access: the [published Zhai article](https://www.nature.com/articles/s41467-025-66063-6)
directly displays the same vacuum resonance in its main-text Eq. (1).
The requested Eq. (10) numbering was not independently retrieved in this pass;
it appears to refer to a supplement or earlier manuscript. The full
Feranchuk--Spence 2000 paper was not retrieved; its attribution remains inherited
from the ledger. The independent phase-matching derivation and the accessible
Zhai expression supply the physics evidence for this scoped API verification.

**Verdict: rederived** for the added API applications of
`line-energy-dispersion`, `surface-hkl-orientation` and `structure-factor`.
No physics discrepancy found. Suggested owner-applied ledger edits: add
`src/pyrite/materials/emission.py::photon_energies` to the line-energy and
orientation owners; make `src/pyrite/materials/crystal.py::_orientation_R` the
orientation definition owner, retaining the Monte Carlo wrapper as plumbing;
add `src/pyrite/materials/emission.py::_primary_hkl` to the structure-factor
owners with the full-occupancy and tolerance scope. Existing claim statuses
need not be downgraded; new API anchors belong to the task owner. Human
sign-off is not conferred by this pass.

Relocation check: AST comparison against the worktree's pre-change `HEAD`
confirmed the executable body of `_orientation_R` is identical after moving
from `montecarlo.geometry` to `materials.crystal` (docstrings excluded).
`pyrite-dev test tests/dev/test_docs.py`: **6 passed**.
`pyrite-dev test tests/montecarlo/transport/test_surface_orientation.py
 tests/materials/test_emission.py`: **53 passed** (three existing detector
argument deprecation warnings). Direct basis-sum probes confirm selected
HOPG/hBN $(002)$ and silicon $(220)$ have normalized per-element amplitude 1;
GeP $(20\bar2)$ has Ge/P amplitudes 0.34598/0.08468, both allowed. These
catalogue probes supplement the independent derivation; they are not independent
measurements of the atomic positions.
