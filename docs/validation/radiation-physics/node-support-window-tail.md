# Support-windowed node reduction with an analytic far tail

Validation: `node-support-window-tail`. Status: **unverified** (issue #1191).

## Need

The incoherent CXR node reduction evaluates every kept line on every grid node.
In the flight-grouped (substepped) route it evaluates every flight's coherent
row sum on every node. Cost is therefore lines × nodes, although a line's mass
sits within a few dozen first-zero widths of its pole. Truncating the profile
at $|v| > x_c$ would drop $1/(\pi x_c)$ of each line's mass (0.3 % at
$x_c = 100$); holding the #1143 yield margin that way needs $x_c \approx 3\times10^4$.
The human decision on #1191 is to keep the tail analytically.

## Derivation

**Endpoint form.** A flight's rows $j$ (linear escape pieces of contiguous
substeps) carry the formation field of `coherent-formation-absorption`:

$$
S(E)=\sum_j c_j e^{i\phi_j}\,\frac{b_j e^{iv_j}-a_j e^{-iv_j}}{2w_j},\qquad
w_j=iv_j-q_j=i a_j\,(E-p_j),\qquad p_j=E_{k,j}-\frac{i q_j}{a_j},
$$

with $a_j=e^{-\tau_{\mathrm{start}}/2}$, $b_j=e^{-\tau_{\mathrm{end}}/2}$ and
$E_{k,j}=E_{\mathrm{vac},j}+\tfrac12\Delta L_j\,\delta\omega(E_{\mathrm{vac},j})/a_j$.
This is exact, so $S$ is a sum of endpoint amplitudes
$\gamma_k e^{i\psi_k(E)}/(E-p_k)$:

- end: $\gamma=c\,b/(2ia)$, $\psi=\phi+v$;
- start: $\gamma=-c\,a/(2ia)$, $\psi=\phi-v$.

$\psi_k$ is the emission phase $\omega t-\mathbf g\cdot\mathbf r-\omega\delta L_{\mathrm{esc}}$
at that endpoint. Its energy slope is the endpoint's observer arrival time.

**Cancellation at junctions.** Two contiguous rows share a junction point, its
time and its escape distance. The two terms there therefore have the same
$\psi$, the same slope and the same attenuation ($b_j=a_{j+1}$). With
$c\propto t_L$ and $a\propto t_L$ they cancel exactly for an unchirped
straight flight; energy loss along the flight leaves an $O(\Delta\beta)$
residual. This is the split invariance of `substep-radiation-invariance`
restated at the endpoints. Far from the poles only the flight's two outer
endpoints survive, so the far field follows the *flight's* width
$a_F=\sum_j a_j$, not a substep's.

**Cycle average.** Endpoints with different slopes beat at their slope
difference. Averaged over that beat their cross terms vanish, so

$$
\langle|S|^2\rangle=\sum_{\text{clusters}}\Big|\sum_{k\in C}\frac{g_k}{x-e_k}\Big|^2,\qquad
g_k=\gamma_k e^{i\psi_k(E_n)},\; x=E-E_n,\; e_k=p_k-E_n .
$$

A cluster is a set of endpoints with equal slope (tolerance $10^{-3}\cdot 2a_F$).
Expanded to second order in $e_k/x$ about the flight's node $E_n$:

$$
\langle|S|^2\rangle=\frac{|A_0|^2}{x^2}+\frac{2\,\mathrm{Re}(A_0A_1^*)}{x^3}+\frac{|A_1|^2+2\,\mathrm{Re}(A_0A_2^*)}{x^4},
\qquad A_m=\sum_k g_k e_k^m .
$$

Absorption enters through the complex poles; chirp and refraction enter through
the spread of $\mathrm{Re}\,e_k$. **Limit:** for one undamped row,
$A_0=\gamma$, $|\gamma_{\mathrm{end}}|^2+|\gamma_{\mathrm{start}}|^2=|c|^2/(2a^2)$,
which is the cycle average $\langle\sin^2\rangle/(a x)^2$ of $|c|^2\mathrm{sinc}^2(ax)$.
The incoherent per-row route is the same expansion with a real pole:
$m_2=w/(2a^2)$, $m_3=2\varepsilon m_2$, $m_4=3\varepsilon^2m_2$, $\varepsilon=E_r-E_n$.

**Window.** A flight is evaluated exactly on nodes with $|E-E_n|\le \mathrm{spread}+W$, where

$$
W=x_c\max\!\Big(\frac1{a_F},\;\max_j\frac{|q_j|}{a_j},\;\frac{2}{g_{\min}}\Big).
$$

Here $g_{\min}$ is the smallest slope gap between *significant* clusters
(intensity $\ge10^{-4}$ of the flight's largest). The first term makes the
flight's ends beat $x_c$ times across the window. The second keeps the complex
pole offset at $\le 1/x_c$ of $x$. The third covers non-contiguous rows, whose
uncancelled endpoints may beat slowly; such flights fall back toward the full
node sum rather than being approximated.

**Bookkeeping.** $K(x)=m_2/x^2+m_3/x^3+m_4/x^4$ (with $K(0)=0$) is added to
every node by one float64 FFT convolution per call. Inside its window each
flight adds (exact − $K$), so each node holds the exact value inside the window
and the tail outside it. Poles beyond a band-wide margin are aggregated into
geometric bins (re-centred by $\le 1/32$ of their distance). A far pole whose
window still reaches the band is evaluated directly.

## Assumptions

- Rows of one flight are contiguous linear escape pieces on one straight path.
  This is what transport substeps and `expand_escape_pieces` produce; the
  slope-gap term makes other inputs safe but slow.
- The grid is uniform. Other grids keep the full-node reduction.
- The reduction samples the profile at nodes. The tail replaces $\sin^2$ samples
  by their mean $\tfrac12$, which matches the reference only where nodes resolve
  the beat ($a\,\Delta E\ll\pi$). That holds on the #1143 reference grids by
  construction.

## Error

Plain truncation would lose $1/(\pi x_c)$ of the mass. The tail leaves:

- the $\cos 2v/(2v^2)$ oscillation, $O(1/x_c^2)$ of line mass;
- the expansion remainder, $O((\mathrm{spread}/W)^3)$;
- cross terms of residual junctions, $O(\sqrt{10^{-4}}/(\pi x_c))$.

Measured at $x_c=100$ against the full-node kernels (anchors below):

| case | yield | mass-L1 |
|---|---|---|
| 1500 random lines, in/out of band, 3 targets | $<5\times10^{-5}$ | $<3\times10^{-4}$ |
| 120 contiguous flights, 1–32 rows, chirp $\le1\%$, absorption, refraction | $<5\times10^{-5}$ | $<2\times10^{-3}$ |
| straight flight split 1 vs 16 rows | identical to $10^{-9}$ | — |
| random non-contiguous rows (guard active) | $<10^{-5}$ | $<10^{-4}$ |

The centroid error is below $10^{-5}$ relative. All of these lie well inside
the #1143 margins (yield and centroid $10^{-3}$, peak/FWHM and mass-L1 $10^{-2}$).
With grid-wide windows both routes reproduce the reference to FFT roundoff.

## Device implementation and replay qualification

NumPy and float32 CuPy share the host window/moment plan. On CuPy, one block
owns a line or flight and adjacent lanes visit adjacent support nodes. Formation
fields use float32; corrections, atomic accumulation, and the FFT use float64
to preserve cancellation near each pole. The correction stays in float64 until
the tail is added. FP64 CuPy and nonuniform grids retain the full-node reference.

`tests/montecarlo/spectrum/test_node_support_window_cuda.py` compares both
backends with identical quantized inputs: multiple polarization/weight targets,
split flights with absorption/refraction, direct and far poles, and full-band
windows. The peak tolerance is $3\times10^{-5}$ plus $3\times10^{-4}$ relative,
and the relative L1 tolerance is $3\times10^{-4}$. All four tests passed on an
RTX 5080 (job 12496, source `c95480569`). This is implementation parity evidence;
the ledger claim remains unverified.

`checks/node_support_window_1191.py` replays a trusted transport pickle through
both runner routes using `Case.node_support_cutoff` (`None` for the reference,
100 for support windows). It alternates arm order, reports the first call
separately from three warm calls, and applies the #1143 yield, centroid, peak,
FWHM and normalized 100 eV mass-L1 margins. A fixed reference peak window also
checks line/background against the same directly evaluated continuum. Reports
include transport digests, synced code identity, backend, grid, population,
cap, memory, and timing spread. These replays establish numerical/performance
qualification; they do not provide independent physics validation.
