# Channeling-zone incoherent-emission flux weights

Validation: `cr-zone-incoherent-flux-weight`.

Fresh-context verifier; implementation inspected only after recording the independent expression. Reviewed source at `46d546cbeb91e20c54226cf23696a62c78acf169`; the task owner subsequently corrected an extra factor of $Z$ in the ledger and inventory. Scientific verdict: **rederived within the stated localization model**. Human sign-off is separate.

## Source, quantity and assumptions

The diagonal rate-operator expectation defines the relative incoherent yield. Its localization follows the Gaussian local-density convention of [the electronic jump kernel](cr-lindblad-jump-operators.md). Shell strengths use [PENELOPE-2024](https://doi.org/10.82155/1vk5-0513), §§3.2.2–3.2.3, Eqs. 3.104–3.105 with moment order zero. Bremsstrahlung uses the soft-photon screened-Born transfer shape associated with [Tsai (1974)](https://doi.org/10.1103/RevModPhys.46.815), supplemented by the explicitly chosen Gaussian localization and $Z^2:Z$ nuclear/electron mixture.

Assumptions: planar, first-Born, local diagonal approximation; transfer $q$ gives Gaussian width $1/q$; entrance-energy kernels; shell charge densities represented by isotropic Slater orbitals; distant-transverse ionization uniform over a cell; EEDL sets ionization magnitudes and GOS sets only relative localization. Bremsstrahlung uses the soft-photon limit and independently normalizes its nuclear and electronic transfer shapes before combining them. Thus its $Z^2:Z$ mixture is a model choice: exact screening-log strengths and photon-energy dependence are not retained. Axial rows and unresolved shells remain unweighted.

The output $F_c(z)$ and row mean $W_c(L)$ are dimensionless. Coordinates and depth are in Å; retained spatial density is in Å$^{-1}$; unnormalized local rate kernels can be in Å$^{-1}$.

## Independent derivation

For a positive diagonal rate operator $k_c(x)$ and retained density $\rho(x;z)$, its conditional expected rate is the ratio of its weighted integral to the retained population. Dividing by the rate of a uniform cell distribution gives

$$
F_c(z)=\frac{\langle\rho(x;z)k_c(x)\rangle_{\rm cell}}
{\langle\rho(x;z)\rangle_{\rm cell}\langle k_c\rangle_{\rm cell}},
\qquad
W_c(L)=\frac1L\int_0^L F_c(z)\,dz.
$$

The retained trace cancels; it must not be multiplied into $F$ a second time because sampled row lengths already account for the zone survival probability. At zero retained population the code assigns one by convention. A uniform density gives one for every process.

Convolving a density with a normalized Gaussian of width $1/q$ multiplies harmonic $G$ by $\exp[-G^2/(2q^2)]$. Atom-site density, thermal displacement and shell density therefore give

$$
k_{s,G}=n_{{\rm el},G}e^{-B(G/4\pi)^2}
\left[f_s(G)T_s(G)+\delta_{G0}\sigma_{t,s}\right],
\qquad
T_s(G)=\int d^2q\,w_s(q)e^{-G^2/(2q^2)}.
$$

The distant-transverse term contributes only to the zero harmonic. At $G=0$, $f_s=1$ and $T_s(0)+\sigma_{t,s}$ is the full zeroth GOS moment. Cross-section units Å$^2$ times site density Å$^{-3}$ give a local rate in Å$^{-1}$. Shell normalization removes all common strength factors, including per-electron versus per-formula-unit conventions.

For a Slater orbital density proportional to $r^N e^{-a r}$, with $N=2n-2$ and $a=2\zeta$, its normalized isotropic Fourier transform is

$$
f_s(G)=\frac{\int_0^\infty r^{N+2}e^{-ar}\,\sin(Gr)/(Gr)\,dr}
{\Gamma(N+3)/a^{N+3}}
=\frac{\sin[(N+2)t]\cos^{N+3}t}{(N+2)\sin t},
\qquad t=\arctan(G/a).
$$

This follows by taking the imaginary part of $\Gamma(N+2)/(a-iG)^{N+2}$. The prescribed binding-energy model is $\zeta=\sqrt{2U_s/E_h}/a_0$. Checks are

$$
f_s(0)=1,\qquad
f_{n=1}(G)=[1+(G/2\zeta)^2]^{-2},\qquad
f_s(G)=1-\frac{G^2\langle r^2\rangle}{6}+O(G^4),
\quad \langle r^2\rangle=\frac{(N+3)(N+4)}{4\zeta^2}.
$$

Within the stipulated soft-photon bremsstrahlung model, define normalized transforms

$$
D_{n,e}(G)=\frac{\int^{mc/\hbar} w_{n,e}(q)e^{-G^2/(2q^2)}\,dq}
{\int^{mc/\hbar} w_{n,e}(q)\,dq},
\qquad
w_n(q)=\frac{[1-F(q)/Z]^2}{q},
\quad w_e(q)=\frac{1-[F(q)/Z]^2}{q}.
$$

The numerical integral starts at $10^{-3}$ Å$^{-1}$. Nuclear and electron localization then combine as

$$
k_{b,G}=\sum_{\rm el} n_{{\rm el},G}e^{-B(G/4\pi)^2}
\left[Z^2D_n(G)+f_0(G)D_e(G)\right].
$$

The electron term is $Z[f_0(G)/Z]D_e(G)$, with no extra $Z$. Consequently $k_{b,0}=\sum n_{{\rm el},0}Z(Z+1)$. The previously written extra factor was a ledger notation discrepancy, corrected by the task owner; code already used this expression.

Normalize each kernel by its zero harmonic and synthesize with the same positive-exponential convention as the planar profile:

$$
K_c(x)=1+2\operatorname{Re}\sum_{m>0}\frac{k_{c,m}}{k_{c,0}}e^{imgx}.
$$

For samples linear between $z_i$ and $z_{i+1}$, the partial interval integral of length $h$ is

$$
I_i(h)=F_i h+\frac{F_{i+1}-F_i}{2(z_{i+1}-z_i)}h^2.
$$

Summing complete trapezoids and this partial integral yields the implemented row mean. The last sample is held beyond the recorded range, and $W(0)=F(0)$.

## Implementation comparison and evidence

`flux_factors` computes the conditional expectation; `row_flux_weights` computes the exact integral of its piecewise-linear samples. `solve_zone` records cell densities on the shared profile coordinate. Native rows receive unit columns; zone rows receive means over their own lengths. Bremsstrahlung multiplies mapped row lengths once. Characteristic emission multiplies shell-ionization cross sections before the cascade response, preserving the distinction between the initially ionized shell and the emitted line.

Independent numerical checks, separate from the implementation formula:

- Direct radial quadrature of the Slater density for $n=1,2,3$, $G=0.3,1,4,20$ Å$^{-1}$ agreed to maximum absolute error $1.75\times10^{-12}$.
- For $K=1+2\operatorname{Re}[(0.2+0.1i)e^{ix}]$ and $\rho=1+0.4\cos x+0.3\sin x$, the independently integrated factor is $1.05$; code returned $1.0499999999999998$. This checks the complex phase convention.
- For $z=[0,2,5]$, $F=[1,3,0]$, lengths $[1,3,6]$, independent exact means $[1.5,6.5/3,8.5/6]$ matched.
- The focused flux-weight module and `tests/dev/test_docs.py` passed together: 21 tests in 1.14 s, using the worktree's bundled catalog explicitly. An initial run during the concurrent demo-profile edit failed collection on a temporary unsupported `line_quadrature` profile key; the corrected profile uses `[cxr_grid_policy].quadrature`.

## Filter verdicts and limits of evidence

| Filter | Result | Scope |
| --- | --- | --- |
| Dimensions and normalization | pass | Dimensionless relative yields; common GOS conversion cancels. |
| Uniform density, zero harmonic, hydrogenic and small-$G$ limits | pass | Analytic derivation and focused tests. |
| Phase/sign convention | pass | Shared profile convention and independent complex-harmonic check. |
| Notation across code, ledger and inventory | pass after correction | Extra electron-field factor $Z$ removed by task owner. |
| Positive kernels | pass for tested Si case; concern outside it | Fourier truncation needs convergence checks for other materials/bases. |
| Universal ordering of shell enhancement | concern | The listed ordering is a Si characterization, not a theorem for all crystals or shells. |
| Source-exact bremsstrahlung mixture | concern, stated approximation | Independent normalization discards distinct nuclear/electron screening logarithms. |
| Reference transcription and tolerances | pass within scope | No newly transcribed measurement table. GOS-strength tolerance is numerical quadrature tolerance, not measurement uncertainty. |
| External-code, hardware-parity, measured-yield evidence | not asserted | No independent absolute orientation-dependent yield is established here. |

Recommend ledger status **rederived** for this explicit model. The analytic and integration tests anchor mathematical subclaims, but this report does not claim an external measurement anchor or human sign-off. Rendered-math verification is performed by the task owner with the repository docs checks.
