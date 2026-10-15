# `cr-debye-b-of-t` — site Debye–Waller $B(T)$ re-derivation

Fresh-context verification. Source: Willis & Pryor (1975); Peng, Ren, Dudarev & Whelan, Acta Cryst. A52, 456 (1996), Eq. 8, 9 and Table 1. Quantity: isotropic $B=8\pi^2\langle u_x^2\rangle$ in Å$^2$ per site; linear interpolation of phonon-DOS rows inside the tabulated range, Debye model with $\Theta$ fitted to the nearest tabulated end outside it.

## Independent derivation (written before reading the implementation bodies)

Debye density of states $g(\omega)=3\omega^2/\omega_D^3$, $\hbar\omega_D=k_B\Theta$. The mean-square displacement of a monatomic cubic crystal (mass $m$), per Cartesian direction:

$$
\langle u_x^2\rangle=\frac{1}{3}\cdot\frac{3\hbar}{2m}\int_0^{\omega_D}\frac{g(\omega)}{\omega}\coth\frac{\hbar\omega}{2k_BT}\,d\omega
=\frac{3\hbar^2}{m k_B\Theta}\left[\frac14+\left(\frac{T}{\Theta}\right)^2\int_0^{\Theta/T}\frac{x\,dx}{e^x-1}\right],
$$

using $\coth(x/2)=1+2/(e^x-1)$. With $y=\Theta/T$ and $\int_0^y x/(e^x-1)\,dx=y\,\phi(y)$:

$$
\langle u_x^2\rangle=\frac{3\hbar^2 T}{m k_B\Theta^2}\left[\phi(y)+\frac{\Theta}{4T}\right].
$$

Multiplying by $8\pi^2$ and using $\hbar^2=h^2/4\pi^2$, $m=A\,m_u$:

$$
B=\frac{6h^2}{A\,m_u k_B}\,\frac{T}{\Theta^2}\left[\phi\!\left(\frac{\Theta}{T}\right)+\frac{\Theta}{4T}\right].
$$

This is the ledger form and Peng Eq. 8 with the prefactor $11492/A$ replaced by its CODATA value.

### Filters

- Units: $h^2/(m_u k_B)$ is J·s·s/(kg·J/K) $=$ m$^2$·K; CODATA gives $6h^2/(m_u k_B)=11490.29$ Å$^2$·K (computed from `scipy.constants`). Peng quotes 11492 (a 0.015 % rounding difference, negligible). $A$ is on the $^{12}$C$=12$ scale, consistent with $m_u=m(^{12}\mathrm{C})/12$.
- $\phi(0)=1$: numeric $\phi(10^{-3})=0.99975$ (series $1-y/4+y^2/36$).
- $\phi(y\to\infty)=\pi^2/(6y)$ since $\int_0^\infty x/(e^x-1)dx=\pi^2/6$: $\phi(200)\cdot 6\cdot200/\pi^2=1.0000000$.
- $\phi(10)=0.164443$ (own quadrature) vs Peng's stated large-$y$ approximation $1.6454/y=0.16454$ (the exact constant is $\pi^2/6=1.644934$, so Peng's is a rounded approximation). The ledger's citation of a Peng Table 3 value 0.16444 could not be confirmed: Table 3 is not legible in the extracted text, but my independent value agrees with it.
- Classical limit $T\gg\Theta$: $\phi\to1$, $B\to(6h^2/Am_uk_B)\,T/\Theta^2\propto T$.
- Zero-point limit $T\to0$: $B\to(6h^2/Am_uk_B)/(4\Theta)$, $T$-independent. For C with $\Theta\approx1890$ K this gives $0.1248$ Å$^2$, matching the plateau seen in the numbers below.
- Monotonic decrease of $B$ with $\Theta$ at fixed $T$ (each term decreases), so the fit root is unique.

## Comparison with implementation

My own reference (`scipy.integrate.quad` for $\phi$, `scipy.optimize.brentq` for $\Theta$, CODATA prefactor) versus `debye_b_factor`, `fit_debye_temperature`, `thermal_b_factor`:

| case | $\Theta$ mine [K] | $\Theta$ code [K] | $B(T)$ agreement |
| --- | --- | --- | --- |
| Si, anchor 170 K, 0.2814 | 561.3401 | 561.3401 | identical to 16 digits at 10–1000 K |
| Si, anchor 280 K, 0.4634 | 519.6988 | 519.6988 | identical (e.g. $B(500\,\mathrm K)=0.77987$, $B(1000\,\mathrm K)=1.52610$) |
| C, anchor 170 K, 0.1336 | 1885.838 | 1885.838 | identical |
| C, anchor 280 K, 0.1422 | 1916.805 | 1916.805 | identical |

Site-level `thermal_b_factor`: Si 50 K $=0.191717$ (Debye from 170 K anchor, equals my 0.191717), Si 200 K $=0.33105$ (linear interpolation between 190 K 0.3145 and 210 K 0.3476), Si 300 K $=0.491229$ (Debye from 280 K anchor, equals mine), C 100 K $=0.129166$, C 300 K $=0.144729$. Rows reproduced exactly; continuous at both ends because $\Theta$ is fitted to the end value.

The derivation and code agree symbolically (same prefactor, same $T/\Theta^2$ and $\Theta/4T$ structure) and numerically.

## Data check against Peng Table 1 (extracted text)

The extracted layout is a multi-column table (Li, Be, C, Na, Mg, Al, Si, ...). Column positions for C (3rd) and Si (7th) were read from the 170–280 K rows:

- Confirmed: all ten Si rows $T=170,180,190,210,220,230,240,260,270,280$ K ($0.2814,0.2979,0.3145,0.3476,0.3641,0.3807,0.3972,0.4303,0.4469,0.4634$) and all ten C rows ($0.1336,0.1342,0.1349,0.1363,0.1370,0.1378,0.1386,0.1403,0.1412,0.1422$) match the TOML `[silicon.Si]` and `[diamond.C]` exactly.
- Cannot confirm / not used: rows at 85–160 K. The extracted Si series (0.2511 at 130, 0.2607 at 140, 0.2648 at 160 K) has a slope break at 160 K inconsistent with its neighbours, and the table skips 150, 200, 250 K; the ledger's decision not to use the sub-170 K rows is supported. The 280 K upper end matches the paper's last printed row.
- Atomic weights 28.0855 and 12.011 are standard values (the paper's Table 1 does not list them in the extracted portion; Table 2 covers other elements), and are not load-bearing beyond the fitted $\Theta$ (since $B$ depends on $\Theta$ and $A$ only through the fit anchored to tabulated $B$).

## Residual caveats

- Linear interpolation of a tabulated curve is a convention; the maximum error is of order $\tfrac18 h^2 B''$ for 20 K spacing, negligible vs the tabulated-data uncertainty.
- Extrapolation below 170 K for Si inherits Debye-model error (Peng: the Debye model fails for Si); that is already recorded in the ledger notes.

## Verdict

`rederived`. No divergence in factor, sign, exponent or unit.
