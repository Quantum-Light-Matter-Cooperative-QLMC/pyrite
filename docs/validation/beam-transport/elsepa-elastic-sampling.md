# Validation: `elsepa-elastic-sampling`

## Scope and independence

Validation: `elsepa-elastic-sampling`. Fresh-context validation on 2026-09-25 in the `issue-89-elsepa-elastic-dcs` worktree. The expressions below were derived from the ledger row, the ELSEPA definitions (Salvat, Jablonski and Powell, *Comput. Phys. Commun.* **165** (2005) 157), and the signatures and docstrings of the ledgered symbols. The implementation bodies were read only afterwards. This validates PyRITE's consumption of ELSEPA tables: flight rate, element choice and polar-angle sampling. It does not validate the ELSEPA tables themselves; those are [`elsepa-vendor-reference`](../ledger-transport-background.md#elsepa-vendor-reference) and [`elsepa-muffin-tin-inputs`](../ledger-transport-background.md#elsepa-muffin-tin-inputs). It does not authorize human `signed-off` status.

## Source quantities

ELSEPA tabulates, for each element and incident kinetic energy $E_k$, the differential cross section $d\sigma/d\Omega(\theta)$ in cm$^2$/sr. It also tabulates the total and transport cross sections

$$
\sigma=2\pi\int_0^\pi\frac{d\sigma}{d\Omega}\sin\theta\,d\theta,\qquad
\sigma_l=2\pi\int_0^\pi\left[1-P_l(\cos\theta)\right]\frac{d\sigma}{d\Omega}\sin\theta\,d\theta .
$$

For unpolarized electrons on unoriented targets, the DCS does not depend on azimuth. The azimuth is uniform on $[0,2\pi)$ and independent of $\theta$.

## Independent derivation

### Angular variable and measure

Define $\mu=(1-\cos\theta)/2\in[0,1]$. Then $d\mu=\tfrac12\sin\theta\,d\theta$ and

$$
d\Omega=2\pi\sin\theta\,d\theta=4\pi\,d\mu,
\qquad
\sigma=4\pi\int_0^1\frac{d\sigma}{d\Omega}(\mu)\,d\mu .
$$

The polar density in $\mu$ is therefore

$$
p(\mu)=\frac{(d\sigma/d\Omega)(\mu)}{\int_0^1 (d\sigma/d\Omega)(\mu')\,d\mu'},
\qquad\int_0^1p\,d\mu=1 .
$$

The constant $4\pi$ cancels in the ratio. Neither the $4\pi$ nor the absolute unit of the DCS enters the angular shape. If the stored DCS is linear in $\mu$ between grid points, the normalization integral is exactly the trapezoid sum $\sum_j\tfrac12(D_j+D_{j+1})\Delta\mu_j$. The CDF at grid points is the cumulative trapezoid sum divided by that total, so $F(1)=1$ by construction.

The moments needed for the limit checks follow from $\cos\theta=1-2\mu$:

$$
\frac{\sigma_1}{\sigma}=\langle1-\cos\theta\rangle=2\langle\mu\rangle,\qquad
\frac{\sigma_2}{\sigma}=\left\langle\tfrac32\sin^2\theta\right\rangle=6\left(\langle\mu\rangle-\langle\mu^2\rangle\right).
$$

### Exact inversion on one panel

On panel $[\mu_j,\mu_{j+1}]$, let $h=\mu_{j+1}-\mu_j$, $t=(\mu-\mu_j)/h\in[0,1]$, and $p=p_j+(p_{j+1}-p_j)t$. Then

$$
F(t)=F_j+h\int_0^t\left[p_j+(p_{j+1}-p_j)s\right]ds
=F_j+h\left(p_jt+\tfrac12(p_{j+1}-p_j)t^2\right).
$$

Set $F=\xi$ and $r=\xi-F_j\ge0$. This gives $\tfrac12h(p_{j+1}-p_j)t^2+hp_jt-r=0$. The root in $[0,1]$ is

$$
t=\frac{-b+\sqrt{b^2+2h(p_{j+1}-p_j)r}}{h(p_{j+1}-p_j)}
=\boxed{\frac{2r}{b+\sqrt{b^2+2h(p_{j+1}-p_j)r}}},\qquad b=hp_j .
$$

The right-hand form comes from rationalizing the numerator. It stays finite as $p_{j+1}\to p_j$, where it gives $t=r/b$, and it avoids cancellation. The discriminant is non-negative on the whole panel. At the largest admissible $r=\tfrac12h(p_j+p_{j+1})$ it equals $h^2p_{j+1}^2$. At $p_j=0$, the root reduces to $t=\sqrt{2r/(hp_{j+1})}$, which is the correct inverse of a triangular panel. The denominator is zero only when $r=0$ and $p_j=0$, and the limit is then $t=0$. The panel index is the largest $j$ with $F_j\le\xi$. A panel of zero probability therefore cannot be selected unless $\xi$ equals its CDF value exactly.

### Energy interpolation

Rates are stored as $\ln(n_i\sigma_i(E_k))$, where $n_i$ is the atom number density in cm$^{-3}$. One element's partial macroscopic cross section is

$$
\ln\Sigma_i(E)=(1-f)\ln\Sigma_{i,k}+f\ln\Sigma_{i,k+1},\qquad
f=\frac{\ln E-\ln E_k}{\ln E_{k+1}-\ln E_k},
$$

and $\Sigma(E)=\sum_i\Sigma_i(E)$. This interpolation is log-log per element. For a compound, the summed $\Sigma$ is a sum of power laws, not itself a power law between nodes. Each element can carry its own energy grid, so only the per-element reading is well defined.

The flight length is $s=-\ln(\xi_1)/\Sigma$ in cm, which is $10^8/\Sigma$ times $-\ln\xi_1$ in Å. The target element is $i$ with probability $P(i)=\Sigma_i(E)/\Sigma(E)$. Here $\Sigma_i$ must be the same interpolants that make up the flight rate. Then $\sum_iP(i)=1$ exactly, and the joint hazard of an element-$i$ collision is $\Sigma_i$.

Across energy, one uniform $\xi$ is inverted on both bracketing nodes. The two quantiles are then combined as

$$
\mu(\xi,E)=(1-f)\,\mu_k(\xi)+f\,\mu_{k+1}(\xi).
$$

Each $\mu_k(\xi)$ is non-decreasing in $\xi$, and the weights lie in $[0,1]$. The mixture is therefore non-decreasing in $\xi$. At $f=0$ it is exactly node $k$'s inverse CDF. At $\xi=0$, every node gives $\mu=0$, so $\cos\theta=1$. Finally $\cos\theta=1-2\mu$.

### Limiting case: screened-Rutherford shape

For $d\sigma/d\Omega\propto(1-\cos\theta+2a)^{-2}\propto(\mu+a)^{-2}$,

$$
\int_0^1\frac{d\mu}{(\mu+a)^2}=\frac1{a(1+a)},\qquad
\int_0^1\frac{\mu\,d\mu}{(\mu+a)^2}=\ln\!\left(1+\frac1a\right)-\frac1{1+a},
$$

so

$$
\langle1-\cos\theta\rangle=2\langle\mu\rangle=2a\left[(1+a)\ln\!\left(1+\frac1a\right)-1\right].
$$

This agrees with the ledger's limit.

## Cheap filters

| check | result |
| --- | --- |
| units | $\sigma_i$ [cm$^2$] $\times\,n_i$ [Å$^{-3}\times10^{24}$ = cm$^{-3}$] gives $\Sigma$ [cm$^{-1}$]; $\lambda=10^8/\Sigma$ [Å]; the DCS unit and $4\pi$ cancel in $p$ — pass |
| energy units | table energies in eV, transport in keV; the log argument is $E_{\rm keV}\times10^3$ and the coverage check compares $E_{\rm eV}/10^3$ with keV bounds — pass |
| limits | $\xi=0\Rightarrow\cos\theta=1$; node reproduction at $f=0$; screened-Rutherford first moment — pass |
| signs/conventions | $\mu=(1-\cos\theta)/2$ and $\cos\theta=1-2\mu$; $F$ non-decreasing; quantile mixture monotone — pass |

## Source-to-code comparison

`elsepa_angular_pdf` builds the trapezoid increments $\tfrac12(D_j+D_{j+1})\Delta\mu_j$, divides by their sum, and pins `cdf[:, -1] = 1`. That is the normalization and CDF derived above. Non-positive or non-finite norms are refused.

`pack_elsepa_tables` stores `log(n_cm3 * sigma)`, with `sigma = total_elastic_cm2`, which is ELSEPA's own total rather than the trapezoid integral. It also stores `log(energy_eV)` and the normalized pdf/CDF rows. It refuses mismatched angular grids, fewer than two nodes, non-increasing energies and non-positive $\sigma$. `n_cm3` is `n_i * 1e24` from the Å$^{-3}$ layer density (`layer_tables.py`).

`_elsepa_bracket` returns the largest node with $\ln E_k\le\ln E$ and $f$ as defined above. An exact node therefore gives $f=0$, and the top node is returned as $(k_{\max}-1,1)$. Out-of-range energies are clamped, but they cannot occur. `check_elsepa_coverage` refuses any run whose $[\min E_{\rm cut},\max E_0]$ lies outside any element's table, and the core terminates electrons at $E_{\rm cut}$.

`_elsepa_rate_scalar` evaluates $\exp[\ln\Sigma_{i,k}+f(\ln\Sigma_{i,k+1}-\ln\Sigma_{i,k})]$, which is the per-element log-log interpolant.

`_elsepa_invert_row` uses the same bisection on `cdf[row, mid] <= xi`, the same $b=hp_j$, $r=\xi-F_j$ and discriminant, and the root $t=2r/(b+\sqrt{\rm disc})$. It adds three round-off guards: $\max(\text{disc},0)$, $t=0$ when the denominator is $0$, and clipping $t$ to $[0,1]$. None of them changes the result in exact arithmetic.

`_sample_cos_theta_elsepa` returns `1 - 2*((1-f)*mu0 + f*mu1)`, which is the quantile mixture above.

In `make_cpu_transport_core`, both cores compute `total_rate` as $\sum_i\Sigma_i(E_j)$ at the substep's starting energy `E_j`, and `lam_ang = 1e8 / total_rate`. Element choice draws `u * total_rate` and walks the cumulative $\Sigma_i(E_j)$. The lockstep core reuses `rate_arr`. The per-electron core recomputes each rate with the identical scalar call at `E_j`. Both use the same interpolants as the flight rate, which gives $P(i)=\Sigma_i/\Sigma$. The polar angle is sampled at the post-step energy `E_keV[e]`, i.e. the energy at the collision point. The `"mott"` and `"sr"` angular branches use the same convention. The element is chosen at the substep-start energy, the same convention every elastic model uses. The difference is $O(\Delta E)$ in the element ratio and is not specific to ELSEPA.

The CUDA kernel (`_jit_kernel.py`) evaluates the same per-element $\exp$ of the log-rate interpolant at `log(E_j*1e3)` for the flight. It recomputes the element CDF with the same expression. The grid helpers `_log_grid_lower` and `_log_grid_fraction` clamp exactly as `_elsepa_bracket` does. The kernel then uses the device `_elsepa_invert_row` (`_jit_device.py`), which is a line-for-line port on C-order flattened rows, at `log(E_keV[e]*1e3)`. The draw order on each electron's counter stream is: element uniform (only if $n_{\rm el}>1$), polar uniform, azimuth uniform. This is the same order as the per-electron CPU core. The launch passes `el_start`/`el_len` reshaped from `(n_layers, max_el)` with `max_el = L_Zs.shape[1]`, which is the same padding as the packer.

No divergent factor, sign, exponent, unit, interpolation order or draw order was found in the code.

## Numerical checks (independent of implementation helpers)

The following were run on 2026-09-25 with scratch scripts that do not use the implementation's inversion or moment code.

- **Inversion exactness:** on a screened-Rutherford table ($a=10^{-4}$, 801-point geometric $\mu$ grid), 20000 samples $\mu(\xi)$ inserted into an independently coded piecewise-linear CDF gave $\max|F(\mu(\xi))-\xi|=2.8\times10^{-17}$. The samples are monotone in $\xi$. $\xi=0$ gives $\cos\theta=1$.
- **Screened-Rutherford first moment:** the exact moment of the piecewise-linear density is $1.64258\times10^{-3}$, against the analytic $2a[(1+a)\ln(1+1/a)-1]=1.64227\times10^{-3}$, a difference of $1.9\times10^{-4}$ relative. With $2\times10^5$ draws the sampled mean is $1.606(42)\times10^{-3}$, within $0.9\,\sigma$.
- **Rate and mean free path:** with $n=0.05$ Å$^{-3}$ and $\sigma=2\times10^{-16}$ cm$^2$, the node rate is $10^7$ cm$^{-1}$ and $\lambda=10.000$ Å, both as expected. At the log midpoint of a two-node table the rate equals $n\sqrt{\sigma_k\sigma_{k+1}}$ to machine precision. For released Si and C tables in a SiC layer at 0.3, 5 and 31.4 keV, the per-element rates agree with an independent `np.interp` of $\ln\sigma$ versus $\ln E$ to $3\times10^{-15}$.
- **Quantile interpolation between nodes:** $\cos\theta$ is non-increasing in sorted $\xi$ at the log midpoint.
- **Node reproduction (released Si table, 10 keV node):** a Kolmogorov–Smirnov test of $2\times10^5$ samples against the independently evaluated node CDF gives $D=1.04\times10^{-3}$. The 99 % critical value is $3.64\times10^{-3}$. The $\ln(E_{\rm keV}\times10^3)$ round trip hits the stored node exactly.
- **Table conventions:** stored $\mu$ matches $(1-\cos\theta)/2$ from `theta_deg` to $5\times10^{-7}$, which is table rounding. `dcs_cm2_sr` matches `dcs_a0_2_sr` $\times\,a_0^2$ to $7\times10^{-6}$. The recomputed CDF matches the stored `angular_cdf` to $2.6\times10^{-15}$.
- **All 28 released tables:** exact moments of the normalized piecewise-linear density against ELSEPA's own ratios give the following maximum deviations. These reproduce the ledger's stated 0.56 % / 0.89 % / 0.59 %.

| band | $\max\lvert 2\langle\mu\rangle/(\sigma_1/\sigma)-1\rvert$ | $\max\lvert 6(\langle\mu\rangle-\langle\mu^2\rangle)/(\sigma_2/\sigma)-1\rvert$ | $4\pi\int D\,d\mu/\sigma-1$ (min, max) |
| --- | --- | --- | --- |
| $E\le1$ MeV | 0.56 % | 0.55 % | $+0.003$ %, $+0.77$ % |
| $1<E\le10$ MeV | 0.89 % | 0.87 % | $+0.45$ %, $+1.37$ % |
| $E>10$ MeV | 0.57 % | 0.59 % | $+0.72$ %, $+1.42$ % |

For the 4 muffin-tin tables, the largest $\lvert4\pi\int D\,d\mu/\sigma-1\rvert$ is 0.39 %.

- **Anchors:** `tests/montecarlo/test_elsepa_elastic.py`: 11 passed, 2 skipped. The skips are the CUDA-gated tests; no GPU was used here.

## Finding in the ledger prose (not the code)

The row's **Stated tolerance** says that above about 10 MeV "the trapezoid `4*pi*integral(DCS dmu)` falls 1.0-1.4 % short of ELSEPA's total". The measured sign is the opposite. On every released table and at every energy, the trapezoid integral **exceeds** `total_elastic_cm2`: by $+0.72$ % to $+1.42$ % above 10 MeV, and by up to $+1.37$ % at 1–10 MeV. An overestimate is also what the trapezoid rule gives when a convex forward peak is under-resolved. The magnitude, the muffin-tin bound (0.39 %) and the conclusion are correct. Only the angular shape carries this error, because the flight rate uses ELSEPA's total. The word "short" should read "over".

A second, minor wording point: "the rate is exact at nodes and geometric at the log midpoint" and "$\Sigma(E)$ interpolated log-log" hold per element, $\Sigma_i=n_i\sigma_i$. For a multi-element layer, the summed $\Sigma$ is a sum of per-element geometric interpolants. It is still exact at shared nodes. This is the only consistent definition when element grids differ, and it is exactly what element choice uses.

## Evidence and verdict

- **Filters:** units pass; limits pass; signs and conventions pass.
- **Re-derivation:** matches. There is no divergent term in the code. The ledger's stated-tolerance sign ("short" should be "over") is a documentation error.
- **Verdict:** `rederived`.
- **Human sign-off:** pending.

## Adjudication

Accepted 2026-09-25. The ledger row now states that the trapezoid integral exceeds ELSEPA's total by 0.7–1.4 % above 10 MeV (the sign was reversed), and that a compound's rate is the sum of per-element log-log interpolants. The status moved to `rederived`; no code changed.
