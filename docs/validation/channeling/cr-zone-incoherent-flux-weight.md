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

## Axial extension (#1148)

Fresh-context verifier of the axial fibre extension, reviewed on the uncommitted working tree over `e606ac427`. The derivation below was recorded from the ledger row before reading `axial_flux_weights.py` or the fibre/runner diffs. Scientific verdict: **rederived within the stated localization model**. Human sign-off is separate. This section supersedes the earlier statement above that axial rows remain unweighted.

### Independent derivation

**Yield along a history.** A local incoherent process with native macroscopic rate $n\sigma_c$ and diagonal localization $k_c(\mathbf r)$ has, for a normalized transverse state $\psi_h(\mathbf r;z)$ of history $h$, conditional rate $n\sigma_c F_h(z)$ with

$$
F_h(z)=\frac{\langle\lvert\psi_h\rvert^2 k_c\rangle_{\rm cell}}
{\langle\lvert\psi_h\rvert^2\rangle_{\rm cell}\langle k_c\rangle_{\rm cell}}.
$$

Along a straight row of direction cosine $v_z$, $dl=dz/v_z$, so the expected yield is $n\sigma_c\int_0^{L}F_h\,dl=n\sigma_c L\,\bar F_h$ with $L=z_h/v_z$ and $\bar F_h=z_h^{-1}\int_0^{z_h}F_h\,dz$. The weight is therefore a **depth** average multiplying the **path** length. Let $w_h$ be the history weight and $a_h(z)=1(z<z_h)$ its retained indicator. The ensemble factor

$$
F(z)=\frac{\sum_h w_h a_h(z)F_h(z)}{\sum_h w_h a_h(z)}
$$

satisfies $\sum_h w_h a_h(z)F(z)=\sum_h w_h a_h(z)F_h(z)$ pointwise, hence

$$
\sum_h w_h\int_0^{z_h}F(z)\,dz=\sum_h w_h\int_0^{z_h}F_h(z)\,dz
$$

exactly in continuous depth. Two qualifications apply: the identity is in depth, while the path carries $1/v_{z,h}$; and on the recorded depth grid both sides are piecewise-linear quadratures, so they agree only to that quadrature. The first is harmless in expectation because the fibre states use one carrier independent of the beam direction $v_{z,h}$. For normalized states, $\sum_h w_h a_h F_h$ equals $\operatorname{Re}\sum_{\mathbf Q}\rho_{\mathbf Q}K^*_{c,\mathbf Q}$ with $\rho_{\mathbf Q}=\sum_h w_h a_h\rho_{h,\mathbf Q}$ and $\rho_0=\sum_h w_h a_h$, which is the ledger form.

**Harmonic overlap.** With $\psi(\mathbf r)=e^{i\boldsymbol\kappa\cdot\mathbf r}\sum_{\mathbf G}\psi_{\mathbf G}e^{i\mathbf G\cdot\mathbf r}$,

$$
\lvert\psi\rvert^2=\sum_{\mathbf Q}\rho_{\mathbf Q}e^{i\mathbf Q\cdot\mathbf r},
\qquad
\rho_{\mathbf Q}=\sum_{\mathbf G}\psi^*_{\mathbf G}\psi_{\mathbf G+\mathbf Q}.
$$

The carrier cancels. For a real kernel $k_c=\sum_{\mathbf Q}k_{c,\mathbf Q}e^{i\mathbf Q\cdot\mathbf r}$, $k_{c,-\mathbf Q}=k^*_{c,\mathbf Q}$, and the cell mean is

$$
\langle\lvert\psi\rvert^2k_c\rangle_{\rm cell}=\sum_{\mathbf Q}\rho_{\mathbf Q}k_{c,-\mathbf Q}
=\sum_{\mathbf Q}\rho_{\mathbf Q}k^*_{c,\mathbf Q},
\qquad
F=\frac{\operatorname{Re}\sum_{\mathbf Q}\rho_{\mathbf Q}K^*_{c,\mathbf Q}}{\rho_0},
\quad K_{c,\mathbf Q}=\frac{k_{c,\mathbf Q}}{k_{c,0}}.
$$

The full sum is already real because $\rho_{-\mathbf Q}=\rho^*_{\mathbf Q}$; taking the real part only removes rounding. The Hamiltonian convention $\langle\mathbf G_i\rvert V\lvert\mathbf G_j\rangle=V_{\mathbf G_i-\mathbf G_j}$ with $V(\mathbf r)=\sum V_{\mathbf G}e^{i\mathbf G\cdot\mathbf r}$ is the same positive-exponential convention.

**String density.** Strings at $\mathbf R_s$ with linear density $\lambda_{s,\rm el}$ in a cell of area $A$, each smeared by an isotropic in-plane Gaussian with variance $a^2_{\rm el}$ per axis, give

$$
n_{\rm el}(\mathbf Q)=\frac1A\int_{\rm cell}n_{\rm el}(\mathbf r)e^{-i\mathbf Q\cdot\mathbf r}\,d^2r
=\sum_s\frac{\lambda_{s,\rm el}}{A}e^{-i\mathbf Q\cdot\mathbf R_s}e^{-a^2_{\rm el}Q^2/2},
$$

in Å$^{-3}$. The planar factor $e^{-Bs^2}$ with $s=G/4\pi$ and $B=8\pi^2\langle u^2\rangle$ equals $e^{-\langle u^2\rangle G^2/2}$, so $a^2_{\rm el}$ is the same one-axis mean-square displacement; the Debye–Waller factor belongs in $n_{\rm el}$ once. The shell and bremsstrahlung transforms are isotropic, so they enter at $\lvert\mathbf Q\rvert$, with the transverse-ionization and $f_0(0)=Z$ terms at $\mathbf Q=0$ only.

**Truncation and sampling.** State orders $\lvert G_i\rvert\le M_i$ give density harmonics $\lvert Q_i\rvert\le 2M_i$ only. Truncating the kernel at $2M_i$ is therefore exact for the in-window state. On a DFT grid of $N_i$ points per axis, $\lvert\psi\rvert^2$ has $4M_i+1$ distinct frequencies, so $N_i\ge 4M_i+1$ avoids aliasing; $4M_i+2$ suffices.

**Limits.** A single order gives $\rho_{\mathbf Q}=\delta_{\mathbf Q0}\lvert\psi_{\mathbf G}\rvert^2$ and $F=1$. A zero state has $\rho_0=0$ and is assigned $F=1$ by convention. A uniform density gives $F=1$ for every kernel.

### Implementation comparison

| Item | Code | Result |
| --- | --- | --- |
| Overlap and conjugation | `AxialFluxKernels.factors`: places window orders at $-M_i..M_i$ modulo $N_i=4M_i+2$, `ifft2`, squares, `fft2`, extracts $-2M_i..2M_i$, $\operatorname{Re}\sum\rho_{\mathbf Q}K^*_{\mathbf Q}/\rho_0$ | matches; `fft2` coefficient at index $q$ is that of $e^{+i\mathbf Q\cdot\mathbf r}$ |
| Order layout | `reshape(2Mx+1, 2My+1)` versus `axial_states` and `_Engine` orders from `meshgrid(..., indexing="ij")`; `chi.reshape(self.shape)` in the engine | consistent C order, $x$ outer |
| Fibre state | `fiber.vectors @ (u / norm)`; `vectors` are retained `eigh` columns, so the window state is unit norm | matches; `factors` also divides by its own $\rho_0$ |
| Kernel unflattening | stable `argsort` of $m^2+n^2$ puts the unique $\mathbf Q=0$ first; `flat[order] = kernels[key]` is the inverse permutation | matches; `kernels[key][0]` is $\mathbf Q=0$ |
| String density | `_string_densities`: $\lambda/A\,e^{-i\mathbf Q\cdot\mathbf R}e^{-a^2Q^2/2}$, $a^2=\tfrac12{\rm tr}$ of the in-plane block | matches; no second Debye–Waller factor in the shared helpers when `profile=None` |
| Ensemble | `_Companions.segment` adds $F_h$ at recorded depths where the history is retained; `_primaries` scales by `event.weight`; `fiber_on_axis_source` divides by $\sum w_h a_h$ | matches; the entry weight is the common projection of one entry state |
| Row weight | `_axial_zone_rows` returns terminal `depth`; rows have `L = depth / v0[:, 2]`; `_attach_axial_flux` averages $F$ over `[0, depth]` | depth average times path length, as derived |
| Row/history pairing | record depths come from `sample_physical_handoff(events=source.events)`, the same terminal events that set $a_h$ | consistent |

### Independent numerical checks

Scratch scripts `scratch/axial_flux_check.py` and `scratch/axial_kernel_check.py` gave the following results:

- Random 35-order state, $M=(2,3)$, $g=(1.3,0.9)$ Å$^{-1}$, nonzero carrier, random Hermitian kernel: direct $64\times64$ real-space cell quadrature $1.0119359750869323$ versus `factors` $1.0119359750869321$. Unequal $M_x,M_y$ would expose a transposed layout. A single order and a zero state gave exactly 1.
- Si $\langle001\rangle$, 300 K, 3.5 MeV, $M=(2,3)$: per-$\mathbf Q$ kernels from separately evaluated two-point calls ($\mathbf Q=0$ prepended, independent string-density sum) matched the unflattened array to $1.0\times10^{-15}$. Hermitian symmetry held to $2.8\times10^{-17}$. Summed string densities matched `nuclear_density` to $1.2\times10^{-14}$ (scale $0.050$ Å$^{-3}$).
- `test_axial_flux_weights.py` passed (4 tests). `test_channeling_axial_fiber_regression.py -k flux` passed (2 tests, including the dense-GKSL oracle). `test_axial_fibre_runner.py -k flux_weights` with `--slow` passed. Its printed zone means were brem 3.2994, K 2.0418, L1 1.1545, L2 1.1063 and L3 1.1063, matching the ledger's rounded values.

### Filter verdicts

| Filter | Result | Scope |
| --- | --- | --- |
| Dimensions | pass | $F$, $W$ dimensionless; $n_{\rm el}$ in Å$^{-3}$; normalization by the $\mathbf Q=0$ harmonic removes strengths. |
| Limits | pass | Single order, zero state, uniform density, and exact no-jump expectation. |
| Signs/conventions | pass | $e^{+i\mathbf G\cdot\mathbf r}$ synthesis, $e^{-i\mathbf Q\cdot\mathbf R}$ string phase, $K^*$ conjugation; brute-force check. |
| Depth vs path | pass | Depth-averaged $W$ times $L=z/v_z$. |
| Truncation | pass | Kernel at $2M_i$ exact for in-window states; DFT $4M_i+2$ non-aliasing. Window truncation of the state itself is inherited from `cr-axial-persisted-channel`. |
| "Summed weighted path identical" | concern, wording | Exact in continuous depth for the depth integral. On the recorded grid it holds only to piecewise-linear quadrature. With per-history $1/v_{z,h}$ it holds in expectation, not sample by sample. |
| Notation | minor | The ledger quotes "L2,3"; the code keys `14:L2` and `14:L3` separately, which have equal values here. |

Recommend ledger status **rederived** for the axial extension within the stated model. Suggested wording: replace "(the summed weighted path is identical)" with "(the summed depth-weighted path equals the per-history sum exactly in continuous depth; on the recorded depth grid to the piecewise-linear quadrature, and with per-history direction cosines in expectation)".

## Real-space reference anchor (2026-11-13)

Fresh-context verifier. The reference was built from the ledger Claim/Assumptions fields and recorded before `flux_weights.py` was read. It is evaluated in position $x$ across the (110) cell, not from Fourier harmonics. Test: `tests/montecarlo/channeling/test_channeling_flux_weight_reference.py` (slow, about 6 s).

**Inputs owned by other claims.** These are the admitted `channeling_demo` zone solve (Si(1-10), 17 MeV, $H=16$, $L=2.0027$ µm), its retained $\rho(x;z)$ on the 128-point cell grid, the plane period $d=1.9201$ Å with one plane per period at $x=0$, Debye–Waller $B_{\rm Si}=0.4817$ Å$^2$, the catalog PENELOPE oscillators with their zeroth GOS moments at $E_0$, and the Waasmaier–Kirfel $f_0$ coefficients of Si.

**Real-space kernels.** Write $e^{-Bs_G^2}=e^{-u^2G^2/2}$ with $u^2=B/8\pi^2$. Also write $e^{-G^2/2q^2}$ as a normalized 1D Gaussian of variance $1/q^2$. Then every localized term is a periodized Gaussian of variance $u^2+1/q_\perp^2$, averaged over the process's transfer weights and convolved with the projected source density:

$$
K_c(x)=\sum_j\int d\mu_c(q_\perp)\,\bigl[P_c\ast\mathcal N(0,u^2+q_\perp^{-2})\bigr](x-jd)\;+\;\frac{\sigma_{t,c}}{d}.
$$

- **Shells.** $P_s(y)=\zeta\,\Gamma(N+2,2\zeta\lvert y\rvert)/[(N+2)\Gamma(N+2)]$ is the 1D projection of the normalized Slater density $r^Ne^{-2\zeta r}$, with $N=2n-2$ and $\zeta=\sqrt{2U_s/E_h}/a_0$. The distant-longitudinal weight is $dQ/[Q(Q+2mc^2)]$ on $[Q_-,U_s]$, with $(\hbar cq_\perp)^2=Q(Q+2mc^2)-(cp_0-cp_1)^2$ at $W_s$. The close weight is Møller $W^{-2}F^{(-)}(E_0+U_s,W)$ on $[U_s,(E_0+U_s)/2]$, with free-target $Q=W$ and $q_\perp$ the same way. Their strengths come from the zeroth moments. The distant-transverse moment is the uniform term.
- **Brem.** The nuclear term is $Z^2$ times a thermal point source. The electron term is $\sum_i a_i\,\mathcal N(0,b_i/8\pi^2)+c\,\delta$, the real-space form of $f_0(s)=\sum_ia_ie^{-b_is^2}+c$. Each term is averaged over its normalized $[1-F/Z]^2\,dq/q$ or $[1-(F/Z)^2]\,dq/q$ transfer on $(0,mc/\hbar]$.
- $F_c(z)=\langle\rho K_c\rangle/(\langle\rho\rangle\langle K_c\rangle)$ uses the periodic rectangle rule on the 128 cell points. $W_c$ uses the trapezoid in $z$, which is the exact integral of the piecewise-linear samples.

**Quadrature convergence.** The reference uses Gauss–Legendre in $\ln q$ or $\ln W$ with $n_q=128$–$800$ nodes, a periodic convolution grid of 4096–16384 points, Gaussian image sums to $12\sigma$, and uniform periodization for $\sigma>3d$. All $W$ agree to better than $10^{-8}$. Halving the cell grid to 64 points leaves $W_{\rm brem}$ unchanged to $10^{-16}$. The retained density is therefore band-limited, and the harmonic truncation at $2H$ is exact.

| Key | Reference $W$ | Production $W$ | Relative difference |
| --- | --- | --- | --- |
| brem | 1.3406491 | 1.3406469 | $-1.7\times10^{-6}$ |
| 14:K | 1.1755646 | 1.1755641 | $-4\times10^{-7}$ |
| 14:L1 | 1.0569468 | 1.0569468 | $<10^{-7}$ |
| 14:L2 | 1.0417790 | 1.0417790 | $<10^{-7}$ |
| 14:L3 | 1.0417962 | 1.0417961 | $-1\times10^{-7}$ |

**Tolerance.** The test uses $10^{-5}$ relative. The reference quadrature contributes $<10^{-8}$. Production-side effects of order $2\times10^{-6}$ cover the residual brem difference of $-1.7\times10^{-6}$. These are its panel quadrature, its lower transfer limit $q=10^{-3}$ Å$^{-1}$ instead of $\delta\to0$ ($+1.9\times10^{-6}$ when applied to the reference), and $f_0(0)=Z$ (below). The reference values are also pinned to $\pm10^{-6}$ absolute.

**Adjudicated convention.** The first reference pass took the catalog $f_0$ literally. It extrapolated the Waasmaier–Kirfel fit past $s=6$ Å$^{-1}$, where the fit tends to $c=0.145$ rather than 0, and gave $W_{\rm brem}=1.3403964$, $1.9\times10^{-4}$ below production. Shells agreed to $5\times10^{-7}$ in that pass. Inspection showed that production zeroes $f_0$ beyond the fit range in the brem transfer weights only (`_screened_transform`; `F0_S_MAX_INV_ANG`). The kernel-harmonic $f_0(G)$ is not cut. With that convention, the reference agrees as tabulated. Zeroing is the better physical limit because $F\to0$ at large $q$. The ledger does not state this convention, so recording it in Assumptions is recommended. The test pins the $-1.9\times10^{-4}$ effect so that the tolerance demonstrably resolves it. A second, smaller convention is that production sets $f_0(0)=Z=14$, while the fit sums to 13.9989. A uniform correction for this changes $W_{\rm brem}$ by $-1.3\times10^{-6}$, inside tolerance.

**Scope.** This anchors production against an independent evaluation of the *stated localization model* at one admitted zone. It does not test the model against measurement, the diagonal approximation, or retained-density accuracy.
