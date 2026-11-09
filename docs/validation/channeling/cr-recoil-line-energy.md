# `cr-recoil-line-energy` re-derivation

Claim: `quantum/_longitudinal.py::channeling_recoil_line_energy_eV`. Source: relativistic energy–momentum conservation in the transverse-mass continuum model of `cr-transverse-eigenproblem`, the Hellmann–Feynman theorem, and the [research derivation](../../research/physics/longitudinal-coherence-recoil.md) §3.3. The recoil substitution of Baier & Katkov, Sov. Phys. JETP 26, 854 (1968), was not available. Every statement below that relies on it is marked.

Fresh context, 2026-11-08, commit `50107731a`. I derived (a)–(e) from the ledger row and the function signature and docstring before reading the function body.

## Model

Axes: plane normal $x$, free in-plane direction $y$, channel axis $z$. $E=T+mc^2$ is the total energy. Each electron state satisfies

$$
(p_zc)^2+(p_yc)^2=E^2-m^2c^4-2E\,\varepsilon(E),\qquad
H_\perp(E)=\frac{p_x^2c^2}{2E}+U(x),\qquad H_\perp\lvert n\rangle=\varepsilon_n(E)\lvert n\rangle .
$$

$U$ is static and depends only on $x$, so $E$, $p_y$ and $p_z$ are good quantum numbers and $p_x$ is not. The photon has energy $\hbar\omega$ and direction $\hat{\mathbf n}=(\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta)$, with $\theta$ measured from $z$ and $\phi$ from $x$. The initial state has $p_y=0$.

## (a) Exact conservation equation

After emission: $E'=E-\hbar\omega$, $p_z'c=p_zc-\hbar\omega\cos\theta$, $p_y'c=-\hbar\omega\sin\theta\sin\phi$, and the final level is $\varepsilon_f(E')$, an eigenvalue of $H_\perp(E')$. The final mass shell is

$$
(p_zc-\hbar\omega\cos\theta)^2+(\hbar\omega)^2\sin^2\theta\sin^2\phi
=E'^2-m^2c^4-2E'\varepsilon_f(E').
$$

Write $s\equiv\sin^2\theta\cos^2\phi$. Then $\cos^2\theta+\sin^2\theta\sin^2\phi=1-s$. Subtract the initial shell $(p_zc)^2=E^2-m^2c^4-2E\varepsilon_i$:

$$
-2p_zc\,\hbar\omega\cos\theta+(1-s)(\hbar\omega)^2
=-2E\hbar\omega+(\hbar\omega)^2+2E\varepsilon_i-2(E-\hbar\omega)\varepsilon_f(E').
$$

The $(\hbar\omega)^2$ terms cancel except for $-s(\hbar\omega)^2$, which gives

$$
\boxed{\hbar\omega\bigl[2(E-p_zc\cos\theta)-2\varepsilon_f(E')\bigr]-s\,(\hbar\omega)^2=2E\bigl[\varepsilon_i-\varepsilon_f(E')\bigr]}
$$

This matches the ledger row term for term. The $-s(\hbar\omega)^2$ term exists only because the photon's $x$ momentum is not charged to the electron. If the electron also absorbed $\hbar k_x$ (free-like kinematics, with all three momenta conserved), $s$ would be replaced by 0.

**Is lattice absorption of $\hbar k_x$ consistent?** Yes, within the stated model. A static $U(x)$ breaks $x$-translation symmetry. Transverse eigenstates carry no $p_x$ quantum number, so no $x$-momentum constraint exists. The kinematics depend on $k_x$ only through the photon's energy–momentum relation, and the line position for a given $(i,f)$ is exact. Lattice recoil energy $(\hbar k_x)^2/2M$ is zero for the static planes. The photon's $e^{-ik_xx}$ enters only the transition amplitude $\langle f\rvert e^{-ik_xx}\hat p_x\lvert i\rangle$. Dropping it there is the dipole approximation, with relative corrections $O(k_xa)$. Its largest value is $k_x\le\gamma\Delta\varepsilon/\hbar c$, reached at $\gamma\theta=1$, so $k_xa\approx0.05$ for $\gamma=100$, $\Delta\varepsilon=2$ eV and $a=0.5$ Å. The approximation changes which lines are bright and how bright they are. It does not move any line, so keeping the $s$ term exactly while truncating the matrix element is consistent.

A subtlety, not an error: $e^{-ik_xx}$ raises the mean transverse energy of the kicked state by exactly $(\hbar k_xc)^2/2E=s(\hbar\omega)^2/2E$. This is the kinetic-energy sum rule for $H_\perp(E)$, since the term linear in $k_x$ vanishes for real bound states. Weighted over all final levels, that shift cancels the $-s(\hbar\omega)^2$ term. A per-level line therefore carries the anisotropy, but an intensity centroid over levels (the quasi-classical picture) need not. See (d).

## (b) Hellmann–Feynman level dependence

$\partial H_\perp/\partial E=-p_x^2c^2/2E^2=-T_\perp/E$, and $U$ does not depend on $E$. Hence

$$
\frac{d\varepsilon_n}{d\ln E}=E\,\langle n\rvert\partial_EH_\perp\lvert n\rangle=-\langle T_\perp\rangle_n,
\qquad
\varepsilon_f(E')\simeq\varepsilon_f(E)-\langle T_\perp\rangle_f\ln\frac{E'}{E}.
$$

This is a first-order (linearized) relation, not an identity, because $\langle T_\perp\rangle_f$ itself varies with $E$. Independent check on a finite-difference $-20\,{\rm sech}^2(x/0.5\,\text{Å})$ eV well at 17 MeV: the central-difference $d\varepsilon/d\ln E$ equals $-\langle T_\perp\rangle$ to $3\times10^{-7}$ eV for the three lowest levels. At $\hbar\omega=50$ keV the linearization error is $2$–$5\times10^{-6}$ eV, compared with the term itself, $\langle T_\perp\rangle\hbar\omega/E=4$–$13\times10^{-3}$ eV. Both the docstring and the code use the linearized form. The ledger writes "=", which should read "≃" (see the wording fixes).

## (c) First-order expansion

Use $\ln(E'/E)\simeq-\hbar\omega/E$, $\Delta\varepsilon=\varepsilon_i-\varepsilon_f$ and $x=\hbar\omega$, and divide (a) by $2E$:

$$
x\Bigl[1-\tfrac{p_zc}{E}\cos\theta-\tfrac{\varepsilon_f}{E}\Bigr]
-\frac{\langle T_\perp\rangle_fx^2}{E^2}-\frac{s\,x^2}{2E}
=\Delta\varepsilon-\frac{\langle T_\perp\rangle_fx}{E}.
$$

Write $\beta=\sqrt{1-\gamma^{-2}}$ with $\gamma=E/mc^2$ (the `cr-doppler-line-energy` $\beta$) and $D=1-\beta\cos\theta$. Then $p_zc/E=\sqrt{\beta^2-2\varepsilon_i/E}\simeq\beta-\varepsilon_i/(\beta E)$, so

$$
1-\frac{p_zc}{E}\cos\theta-\frac{\varepsilon_f}{E}
\simeq D+\frac{\varepsilon_i-\varepsilon_f}{E}+\frac{\varepsilon_i}{E}\Bigl(\frac{\cos\theta}{\beta}-1\Bigr).
$$

The last term is the absolute-level remainder. Because $\lvert\cos\theta/\beta-1\rvert\simeq\lvert\theta^2/2-1/2\gamma^2\rvert\le D$, it changes $\omega$ by a relative $O(\varepsilon_i/E)$ at most. Set $x=\omega_D(1+\delta)$ with $\hbar\omega_D=\Delta\varepsilon/D$ and keep first order in $\hbar\omega_D/E$:

$$
\boxed{\frac{\omega}{\omega_D}-1\simeq-\frac{\hbar\omega_D}{E}\Bigl[1+\frac{\langle T_\perp\rangle_f}{\Delta\varepsilon}-\frac{\sin^2\theta\cos^2\phi}{2(1-\beta\cos\theta)}\Bigr]+O\!\Bigl(\frac{\varepsilon}{E}\Bigr)+O\!\Bigl(\frac{\hbar\omega_D}{E}\Bigr)^2}
$$

Term by term: the "1" comes from $(\varepsilon_i-\varepsilon_f)/E$ in the effective Doppler denominator. The $\langle T_\perp\rangle_f/\Delta\varepsilon$ term comes from $-\langle T_\perp\rangle_f x/E$ on the right side; the $x^2/E^2$ term is second order. The anisotropy term comes from $-sx^2/2E$. This matches the ledger row, the research note first-order equation (§3.3), and the test's first-order formula, including signs. In 50-digit arithmetic over six cases (5 MeV–1 GeV, $\theta\le1/\gamma$, several $\phi$), the expansion reproduces the exact shift to 0.2–1.1 %. The largest residual (5 MeV, $\theta=0.05$) is the $O(\varepsilon/E)$ remainder: $3.6\times10^{-6}$ against a shift of $1.7\times10^{-4}$.

## (d) Forward fixed-level limit (Baier–Katkov)

For $\theta=0$ and $\langle T_\perp\rangle_f=0$, (a) is linear in $x$:

$$
\hbar\omega=\frac{E\Delta\varepsilon}{E-p_zc-\varepsilon_f},\qquad
E-p_zc=\frac{m^2c^4+2E\varepsilon_i}{E+p_zc}=\frac{E}{2\gamma^2}+\varepsilon_i+O\!\Bigl(\frac{\varepsilon}{\gamma^2},\frac{\varepsilon^2}{E}\Bigr).
$$

So $\hbar\omega=\Delta\varepsilon/[1/2\gamma^2+\Delta\varepsilon/E]\,(1+O(\varepsilon_i/E))$. With $1-\beta=1/2\gamma^2+O(\gamma^{-4})$ this is $\omega_D/(1+\hbar\omega_D/E)$ to $O(\varepsilon/E)$. Numerically (50 digits), the relative gap is $3.3\times10^{-6}$, $1.1\times10^{-6}$, $1.8\times10^{-7}$ and $9.3\times10^{-9}$ at 5, 17, 54 MeV and 1 GeV. In each case it equals $\lvert\varepsilon_i\rvert/E$ to two digits, so the remainder is exactly the absolute-level term.

**Depends on Baier–Katkov (not inspected).** The identification relies on the standard textbook form of the recoil substitution, $\omega\to\omega E/E'$ applied to the classical line, which gives $\omega=\omega_D/(1+\hbar\omega_D/E)$. Applied at any angle, that substitution has no $\phi$ dependence. In (c), the $s$ term is the same order as the recoil "1" at $\gamma\theta\sim1$ ($s/2D\to\tfrac12\cos^2\phi$). Off axis, the per-level line derived here therefore differs from the quasi-classical BK line by a term of BK's own order. The sum-rule remark in (a) is a plausible reconciliation: BK's operator $e^{-i\mathbf k\cdot\mathbf r}$ kicks $p_x$, so its line is the analogue of an intensity centroid. I have not checked this against the source. The ledger claims BK agreement only in the forward limit, where $s=0$, and that claim is correct. The off-axis anisotropy should not be described as a BK result, and the research note does not do so.

## (e) Interpretation

Group velocity in the model: differentiate the shell $E^2-m^2c^4-2E\varepsilon(E)=p_z^2c^2$ at fixed level $n$ and use (b):

$$
v_z=\frac{\partial E}{\partial p_z}=\frac{p_zc^2}{E-\varepsilon+\langle T_\perp\rangle}
\quad\Rightarrow\quad 1-\beta_z\simeq\frac{1}{2\gamma^2}+\frac{\langle T_\perp\rangle}{E}.
$$

The absolute level $\varepsilon$ cancels at this order and only the transverse kinetic energy slows the drift. This is the undulator relation $1-\bar\beta_z=(1+K^2/2)/2\gamma^2$ with $K^2/2\leftrightarrow2\gamma^2\langle T_\perp\rangle/E=\gamma^2\langle\beta_x^2\rangle$. Then $\Delta\varepsilon/(D+\langle T_\perp\rangle/E)\simeq\omega_D[1-(\hbar\omega_D/E)\langle T_\perp\rangle/\Delta\varepsilon]$ reproduces the $\langle T_\perp\rangle$ term of (c). The note's reading is confirmed: the term is classical longitudinal slow-down, $\hbar$-free as $2\gamma^2\langle T_\perp\rangle/E$ in the forward direction, and absent from `cr-doppler-line-energy`. At this order the correct value is $\langle T_\perp\rangle_f$ and not the initial or mean value. The difference $\langle T_\perp\rangle_i-\langle T_\perp\rangle_f=O(\Delta\varepsilon)$ is the same order as the recoil term, so the classical picture fixes this term only up to the recoil order. The quantum equation (a) resolves it, because the final level is evaluated at $E'$.

The note's claim that "absolute level offsets enter only at $O(\varepsilon/E)$" is confirmed by the remainder in (c) and by the numbers in (d).

## Filters

| Check | Result |
| --- | --- |
| Units | Every term of (a) is eV²; $\langle T_\perp\rangle/\Delta\varepsilon$, $s$ and $D$ are dimensionless. Pass. |
| Forward fixed-level limit | BK form to $O(\varepsilon_i/E)$. Pass. |
| No recoil ($\hbar\omega_D/E\to0$) | $\omega\to\omega_D$ (`cr-doppler-line-energy`). Pass. |
| Anisotropy sign | $\phi=0$ gives a higher energy than $\phi=\pi/2$, because the electron does not pay the $x$ recoil. Pass. |
| Raising branch | $\Delta\varepsilon<0$ gives a negative root, so NaN. Pass. |
| Root selection | The equation is quadratic in $x$ (with weak $\log$ nonlinearity), $\sigma x^2-2Bx+2E\Delta\varepsilon\simeq0$ with $\sigma=s+2\langle T_\perp\rangle_f/E$. Small root $\to\omega_D$. Large root $\simeq2B/\sigma\ge2E(1-\cos\theta)/\sin^2\theta\ge E$ when $s$ dominates, using $1-\cos\theta\ge\tfrac12\sin^2\theta$, and larger still otherwise. It exceeds $T$ and is unphysical. A 2000-point scan of $(0,E)$ finds only the small root in every case. Pass. |

## Diff against the implementation

Body read after the derivation. The code computes, literally:

```text
total = kinetic + M_E_EV
pz2 = _momentum_ev(kinetic) ** 2 - 2.0 * total * eps_i
doppler = (M_E_EV**2 + 2.0 * total * eps_i) / (total + pz)
doppler += 2.0 * pz * np.sin(0.5 * theta_a) ** 2
a = (np.sin(theta_a) * np.cos(phi_a)) ** 2
...  # 64-step fixed point
    level_f = eps_f - t_f * np.log1p(-photon / total)
    c = 2.0 * total * (eps_i - level_f)
    b = 2.0 * (doppler - level_f)
    updated = 2.0 * c / (b + np.sqrt(b * b - 4.0 * a * c))
valid = (pz2 > 0) & (photon > 0) & (photon <= kinetic) & np.isfinite(photon)
```

- `doppler` is $E-p_zc\cos\theta=(E^2-p_z^2c^2)/(E+p_zc)+2p_zc\sin^2(\theta/2)$ with $E^2-p_z^2c^2=m^2c^4+2E\varepsilon_i$. This is exact and cancellation-free. `_momentum_ev` gives $pc=\sqrt{T(T+2mc^2)}$.
- At fixed `level_f`, (a) reads $a x^2-bx+c=0$. `2c/(b+sqrt(b²-4ac))` is the rationalized small root $(b-\sqrt{b^2-4ac})/2a$: stable, and correct at $a=0$. It is the physical root by the root-selection filter.
- `level_f` is the linearized Hellmann–Feynman level of (b), with `log1p` for accuracy.
- The fixed-point map contracts by about $\langle T_\perp\rangle\hbar\omega/(E^2D)\ll10^{-6}$, so it converges in two or three steps. If 64 steps did not converge, the code would return the last iterate without flagging it. This is unreachable in practice and does not count as a finding.
- Raising transitions, $\Delta\varepsilon=0$, negative discriminants and $p_z^2\le0$ all give NaN.

Numerical comparison (`channeling_recoil_line_energy_eV` against my 50-digit root of (a), 8 cases, 1 MeV–1 GeV, $\theta$ up to 3 rad, assorted $\phi$): maximum relative difference $6.9\times10^{-16}$. **Matches.**

Anchors: `uv run pyrite-dev test tests/montecarlo/quantum/test_longitudinal_coherence_reference.py` passes 34/34. The test's mass-shell reference is an independent mpmath residual of the full shell, not of (a). The 5e-6 BK tolerance is justified by $\lvert\varepsilon_i\rvert/E=1.0\times10^{-6}$ at 5 MeV, and the 1 % first-order tolerance holds for the test's cases. `checks/longitudinal_coherence_regimes.py` exits 0 ("limits: ok").

Regime table (research note §5): I checked each row from its printed $\Delta\varepsilon$, $\langle T_\perp\rangle_f$ and $\hbar\omega_D$. The BK column is $\hbar\omega_D/E$ (Si 5 MeV: $2.402/5511=4.36\times10^{-4}$). Total forward shift $\approx\mathrm{BK}\,(1+\langle T_\perp\rangle_f/\Delta\varepsilon)$ (Si 54 MeV: $2.882\times10^{-3}$, exact $2.873\times10^{-3}$). The anisotropy at $\theta=1/\gamma$ is $\hbar\omega_D(0)/4E$, because $\omega_D$ halves and $s/2D=\tfrac12$ there (Si 5 MeV: $1.09\times10^{-4}$). All rows agree to the printed digits, and the ledger's 0.06–0.37 % range matches.

## Findings (wording and convention, no physics discrepancy)

1. **Level zero in the regime check.** The docstring and ledger Notes say levels are "referred to the vacuum potential zero". `checks/longitudinal_coherence_regimes.py` passes `transverse_states(...).energies_eV`, which are referenced to $U_0=0$ (`cr-planar-continuum-potential`, `cr-mean-inner-potential` Notes), not to vacuum. Offsetting by $U_0\approx-13.8$ eV (Si) or $-20.8$ eV (diamond) changes the forward shift by 0.46 % of itself at Si 5 MeV ($-5.514\to-5.489\times10^{-4}$), 0.05 % at Si 54 MeV and 0.05 % at diamond 20 MeV. No printed table value changes at its precision. This is exactly the $O(\varepsilon/E)$ effect the claim allows. Fix: either add $U_0$ in the check, or state in the check docstring and note §5 that the table uses the $U_0=0$ reference, with the offset effect below 0.5 % of the shift.
2. **Ledger "=" for Hellmann–Feynman.** The ledger Claim writes `ε_f(E′) = ε_f − ⟨T⊥⟩_f ln(E′/E)`. It is the first-order (linearized) relation, as the docstring says. Use "≃", or add "linearized". The research note §3.3 has the same "=" in prose.
3. **"leaves the matrix element at $O(k_xa)$"** (note §3.3 and ledger Notes "Matrix elements stay at transverse dipole order (`O(k_x a)`)"). Clearer: "matrix elements are kept at transverse dipole order, neglecting $O(k_xa)$ corrections; line positions are exact".
4. **Off-axis BK.** Recommend adding to the note that the $\sin^2\theta\cos^2\phi$ term is a per-level kinematic result absent from the quasi-classical BK substitution, which is $\phi$-independent. Its relation to BK off axis is unverified until Baier & Katkov (1968) is inspected. The forward claim is unaffected.

## Verdict

`rederived`. (a)–(e) re-derive independently. The implementation solves (a) with its linearized Hellmann–Feynman level, returns the physical small root, and agrees with an independent 50-digit solve to $7\times10^{-16}$. The forward BK identification depends on the standard form of the recoil substitution. Baier & Katkov (1968) itself was not inspected.

## Owner source check (2026-11-08, after the verdict)

Baier & Katkov, Sov. Phys. JETP 26, 854 (1968), checked by the claim owner:

- Eq. (17) gives the squared matrix element the phase
  $\exp\{i[\omega\tau+E(\mathbf k\cdot\boldsymbol\rho-\omega\tau)/E']\}$ with
  $E'=E-\hbar\omega$. Recoil enters only through $\omega\to\omega E/E'$ on the
  classical $\boldsymbol\rho(\tau)$. For periodic motion the harmonic condition
  $(E/E')\omega(1-\bar\beta\cos\theta)=k\omega_0$ gives
  $\omega=\omega_D(\theta)/(1+\hbar\omega_D(\theta)/E)$ at every angle. That
  confirms (d) in the forward limit.
- BK reach this form after replacing the trajectory operators by classical
  values (text after Eq. 16, terms $\sim1/\gamma$ and $\sim\hbar\omega_0/E$
  dropped). It is a classical-trajectory result, not a bound-bound line.
  Along a free transverse direction, the $(\hbar k_\perp)^2$ term cancels
  exactly in the mass-shell identity. Only the bound plane-normal component
  survives as $\sin^2\theta\cos^2\phi$. The off-axis anisotropy is therefore
  outside BK's scope, not in conflict with it. The centroid/sum-rule
  reconciliation in (d) remains unshown.
