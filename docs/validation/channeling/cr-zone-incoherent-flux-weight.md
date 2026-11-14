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

## Measured GaAs (400) comparison (#1178) — fresh-context verification 2026-11-13

Fresh-context verifier. Sources were read before the check body. These were Schowalter et al. (2009) Table 6 and Sec. 4 (rendered PDF page 11), Kundmann & Gronsky LBL-25204 (OSTI 6890669 full text) and Nüchter & Sigle (1995) Secs. 3, 6 and 7.2 with Fig. 6 (printed pp. 178–179, rendered at 220 dpi). Scratch scripts live under `scratch/verifier/`.

**Thermal rows.** The Table 6 (LDA) GaAs values are $\langle u^2\rangle_{\rm Ga}=3.397, 6.038, 11.658, 17.368\times10^{-3}$ Å$^2$ and $\langle u^2\rangle_{\rm As}=2.949, 5.201, 10.014, 14.909\times10^{-3}$ Å$^2$ at 100, 200, 400 and 600 K. With $8\pi^2=78.957$, these give the tabulated `B_ang2` to all four digits. Table 6 lists $\langle u^2(\nu;T)\rangle$ per element. The paper's Eq. 42, $\langle u^2\rangle=(\hbar/2M_\nu)\coth(\hbar\omega/2k_BT)/\omega$, is the one-dimensional oscillator mean square. That matches `thermal.py`'s $B=8\pi^2\langle u_x^2\rangle$. Sec. 4 quotes Stahn et al. (1998) at 287 K as Ga 0.00844 and As 0.00716 Å$^2$, against LDA values of 0.00845 and 0.00726 (0.1 % and 1.4 %). Linear interpolation of the 200 and 400 K rows gives 0.008483 and 0.007295 Å$^2$, which is +0.5 % and +1.9 % from the measurement. The TOML comment is correct.

**Conduction band.** LBL-25204 Table 1 gives GaAs $E_p=16.09\pm0.04$ eV and $\Gamma_p=4.0\pm0.1$ eV. The values come from a $\chi^2$ lineshape fit to Fourier-log-deconvolved single-loss spectra, and the text notes d-shell features on the high-loss side. The 8 electrons are Ga $4s^24p^1$ plus As $4s^24p^3$, which consumes the outermost shells as the file convention requires. The free-electron estimate for 8 e per $a^3/4$ ($a=5.653$ Å) is about 15.6 eV, consistent with that count. The OSTI record carries no DOI. `_persistent_id` requires exactly one of `doi`/`osti`, and its `ValueError` falls inside the existing `except`. This is correct, and the tests pass.

**Digitization.** Axis ticks were calibrated in pixels for each panel. A ring-template circle detector searched a $\pm9$ px window around every digitized point of panels (a)–(d), all 103 points, with a visual zoom check of the ambiguous panel (c) peak region. The located circles agree within 0.015 in ratio everywhere. In $k_x/g_{400}$ they agree within 0.02 except in the overlapping $\xi<0.08$ clusters and the dense 0.47–0.55 and 0.75–0.85 groups. There the detector cannot separate neighbouring circles (offsets up to 0.037), but visual inspection puts the markers on circles. No misreading exceeds 0.03 in ratio. Spot-checked half error bars (for example (a) 0.20, (c) 0.19, 0.31 and 0.36) agree within 0.01. The five low-tilt points are given evenly spaced $\xi=0.02$–$0.06$, whereas the drawn circles span about $-0.005$ to $0.08$. The model has zero slope at $\xi=0$, so this is immaterial. Panel (c) 0.36 sits about 0.02 left of the 0.38 used in the other panels for the same tilt. Experimental conditions match Sec. 6 and Table 2: 400 keV, 300 K, JEM 4000FX, Ge detector, 400 row, and wedge 0–300 nm estimated and 0–303 nm fitted. No convergence is stated; Sec. 8 recommends only that divergence be "small compared with the Bragg angle". The paper names only "L" lines and never specifies a subshell, so L3 ($L\alpha$) is a modelling choice.

**Observable.** For uniform illumination of a wedge $t\in[0,T]$, with $N(t)=\int_0^tI(z)\,dz$, the paper's Eq. 20 average is

$$
\bar N=\frac1T\int_0^T N(t)\,dt=\int_0^T I(z)\,\frac{T-z}{T}\,dz .
$$

Here $I=P\,F_c+(1-P)$. Sec. 3 treats inelastically scattered electrons as plane waves of homogeneous cell intensity (Hall 1966), so $W=1$ for $1-P$ is the stated model. `F_c` is the retained-density-normalized factor (`weighted / total`), so $P\,F_c$ is the coherent yield per incident electron. For tilt, $\xi=k_x/g_{400}$ maps to incidence $\psi=\xi g_{400}/p$. The solver's `entry_lattice` takes $k_x=p\psi$ with the same $p$, so the angle cancels exactly and refraction does not enter $k_x$. The profile period is 2.82685 Å, which is $a/2$, so $g_{400}=4\pi/d=2g_{200}$. The grid $k_x=l\,g_{200}/n$ is $\xi=l/2n$, which equals `tilt_grid`'s step $1/32$ at $n=16$, so no snapping occurs. Shell cross sections and detector efficiency are tilt-independent factors, so one scale per ratio is the paper's normalization. The scale is $s=\sum wMD/\sum wM^2$ with $w=\sigma^{-2}$, which a numerical minimization confirms. Using $\sigma$ as half the drawn bar assumes ±1σ bars, which the paper does not state.

**Independent recomputation.** This used the zone-solver outputs with its own Simpson quadrature of the $(T-z)/T$ form. At $\xi=0$ it gives Ga-K/Ga-L 1.05285, As-K/As-L 1.08284, Ga-K/As-L 0.80522 and Ga-K/As-K 0.74362 (unscaled). At $\xi=0.5$ it gives 0.99949, 0.97454, 1.11571 and 1.14485. Both rows are identical to `row_weights` to 5 digits, and so are $\xi=0.25$ and $0.34375$. The rerun default and `--ladder` arms reproduce the committed record exactly. The incidence relations are $p=382.20$ Å$^{-1}$, $g_{400}=4.4457$ Å$^{-1}$ and 11.632 mrad per unit $\xi$ (Bragg 5.816 mrad). This makes $\sigma_\xi=0.18$ and $0.21$ equal to 2.09 and 2.44 mrad rms. The smear was checked two ways.

- An independent continuous Gaussian quadrature on $\lvert\xi\rvert$ over a grid extended to $\xi=2.5$ reproduces the check's $\chi^2/N$ within 0.01. The $\xi_{\max}=1.75$ edge truncation is negligible.
- The solver's native `divergence_rad` gives an incoherent Gaussian mixture of grid momenta in one propagation. At $\xi=0$, 0.25 and 0.375 it agrees with `smear` to $\le10^{-4}$. This confirms the mirror and the incoherent sum of counts before the ratio.

| Ratio | $\chi^2/N$, $\sigma_\xi=0$ | 0.18 | 0.21 | Flat-line null |
| --- | --- | --- | --- | --- |
| Ga-K/Ga-L | 0.50 | 0.73 | 0.81 | 2.07 |
| As-K/As-L | 0.96 | 1.13 | 1.29 | 3.91 |
| Ga-K/As-L | 10.79 | 2.81 | 2.61 | 6.10 |
| Ga-K/As-K | 13.13 | 3.18 | 3.90 | 19.82 |

**Sensitivities.** The ladder (basis 16, cells 32, dz 5 Å) moves $\chi^2/N$ by at most 0.28 (Ga-K/As-K at cells 32) and 0.14 in the same-element ratios. Varying the wedge to $T=200$, 250 and 350 nm gives cross ratios of 12.6, 12.0 and 9.5 (Ga-K/As-L) and 14.0, 14.1 and 11.9 (Ga-K/As-K), with same-element values 0.48–1.07. Using L2, L1 or L3 + ½L2 instead of L3 changes every ratio by ≤ 0.65 in $\chi^2/N$. Thickness uncertainty and L-subshell assignment therefore do not explain the cross-element failure.

**Adjudication.** The observable, normalization, tilt mapping, grid and smear match the paper's stated model, and the stated numbers reproduce. The same-element passes are discriminating: the plane-wave prediction beats a flat line ($\chi^2/N$ 0.50 vs 2.07 and 0.96 vs 3.91) and places the 400 dip at 0.594. Ga-K/As-K involves two strongly localized K shells and fails at $\chi^2/N=13.1$, which isolates the sublattice flux partition near the 200 position as the discrepancy. This supports the ledger's interpretation. For Ga-K/As-L, the plane-wave prediction ($\chi^2/N=10.8$) is worse than a constant (6.1). The scaled contrast is max/min 2.31/1.33 = 1.74, against about 1.37 measured. The authors' own fitted curve in Fig. 6(c), read by eye, also rises steeply through 0.2–0.3 but reaches only about 2.03/1.54. Their "excellent agreement" is a visual judgement, not a $\chi^2$ against these bars.

| Filter | Result | Note |
| --- | --- | --- |
| Reference data (thermal, plasmon) | pass | Values, conversion and 1D convention verified against source. |
| Digitization | pass | ≤ 0.015 in ratio; $\xi$ ≤ 0.02 for isolated points; the low-tilt spread is idealized. |
| Units/tilt mapping | pass | 2.1–2.4 mrad conversion verified. |
| Observable/limits | pass | The wedge average equals N&S Eq. 20; $P\to1$ recovers the coherent limit and $F_c\equiv1$ gives a flat ratio. |
| Pass criterion | pass, wording | $\chi^2/N\le1.5$ against half bars, one scale; it assumes ±1σ bars. |

**Verdict:** confirmed with findings for the #1178 measured-comparison extension. The status stays `anchored`. Only the same-element shapes are measured anchors.
