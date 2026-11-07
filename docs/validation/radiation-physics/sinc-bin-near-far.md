# Near/far reduction of the bin-mean line spectrum

Validation: sinc-bin-near-far

## Claim and assumptions

The incoherent `bin-mean` spectrum is a nonnegative weighted sum of line
profiles. For a line with resonance $E_l$, positive inverse width $a_l$ in
eV$^{-1}$, and nonnegative weight $w_l$ in photons/eV, the #192 hybrid uses
the exact sinc-squared bin mean within $K=64$ first-zero widths and the
oscillation-averaged envelope beyond them. Only the reduction of those far
envelopes changes here. The energy edges are strictly increasing and the same
float32 line inputs are used by both device routes.

For a far line, the envelope is $w_l/[2a_l^2(E-E_l)^2]$. With
$c_l=w_l/(2a_l^2)$, its mean over $[L,H]$ is

$$
\frac{1}{H-L}\int_L^H\frac{c_l\,dE}{(E-E_l)^2}
=\frac{c_l}{(L-E_l)(H-E_l)}
=\frac{f_l(L)-f_l(H)}{H-L},\qquad f_l(E)=\frac{c_l}{E-E_l}.
$$

$c_l$ has units photons·eV and the mean has units photons/eV. Each far
pair's two factors have the same sign, so the result is nonnegative. The
product form evaluates a narrow bin without subtracting two nearly equal
Cauchy values. An edge at a resonance is always a near pair and uses the
exact sinc-squared mean; the far evaluator never forms that pole.

## Tree separation and error

Sort lines by $E_l$ and group at most 32 per leaf. Each internal tree node
stores its resonance center $C$, half-span $h$, guarded near radius
$R=\max_l\{(K\pi/a_l)(1+2\times10^{-6})
+10^{-12}\max(|E_l|,K\pi/a_l)\}$, and moments
$M_j=\sum_l c_l((E_l-C)/h)^j$, $0\le j\le10$. A node is accepted only when
the whole interval $[C-h-R,C+h+R]$ lies strictly to one side of the bin and
$h/\min(|L-C|,|H-C|)\le0.2$. Otherwise traversal descends; a leaf evaluates
each line with the #192 host/device near/far classifier and exact near
mean. Overlapping near windows are handled independently. An accepted node
contains only far pairs, including on nonuniform bins.
The radius guard covers the float32 scale and split-edge rounding in #192's
near classifier. Without it, an edge only $10^{-9}$ eV beyond the FP64
64-width boundary at $E_l=8000$ eV and $a_l=3$ eV$^{-1}$ is still classified
near by #192; an unguarded tree would replace its exact near mean with the
far envelope. Both sides of this boundary are pinned by regression.

Put $p=h/(L-C)$ and $q=h/(H-C)$. Expanding the product gives

$$
\sum_{l\in N}\frac{c_l}{(L-E_l)(H-E_l)}
=\frac{1}{(L-C)(H-C)}
  \sum_{j=0}^{\infty}M_j\sum_{m=0}^{j}p^m q^{j-m}.
$$

The implementation retains $j\le10$. For $r=\max(|p|,|q|)\le0.2$ and
$c_l\ge0$, the absolute omitted series, divided by the exact positive
node sum, is bounded by
$r^{11}(12-11r)(1+r)^2/(1-r)^2 < 4.6\times10^{-7}$.
The far envelope outside $K$ widths carries at most
$1/(\pi^2K)$ of the whole-line mass. Thus the additional integrated-yield
allowance is $4.6\times10^{-7}/(\pi^2\cdot64)<7.3\times10^{-10}$ per
line in exact arithmetic. The production float32 leaf envelope and final
array cast add ordinary rounding; the matched #192 comparison measures that
separately. This allowance is in addition to the #192 oscillatory-envelope
allowance of $1.2\times10^{-5}$.

For one line, the tree reduces to the #192 leaf formula. With $K\to\infty$
the existing all-exact route is retained. If all bins are near, no node is
accepted and the result is the #192 near reduction.

## Evidence and status

Host tests compare the tree with the #192 hybrid on irregular edges,
clustered and overlapping resonances, and edges coincident with resonances.
The CUDA regression compares the device tree, host tree, and device all-pairs
kernel on the same inputs. A matched RTX 5080 case benchmark remains pending.
Only a human may mark this claim signed off.

## Independent verification

A fresh-context verifier first checked commit `5f4cd571` on 2026-09-28 and
found a discrepancy: the unguarded tree could accept a node just beyond the
FP64 64-width threshold even though the #192 float32 classifier kept its
pairs near. With 33 identical lines at $E_l=8000$ eV, $a_l=3$ eV$^{-1}$,
and a bin starting $10^{-9}$ eV beyond that threshold, the exact near mean
was about $7.42\times10^{-11}$ per line while the tree used a far mean near
$1.2368\times10^{-5}$. The radius guard and left/right regressions correct
this at commit `bb5d80a8`.

The verifier re-checked `bb5d80a8` from fresh context and returned
**rederived** for the partial-fraction identity, moment truncation, units,
positivity, single-line and all-near limits, and the guarded boundary. This
is a source-to-code mathematics verdict. CUDA execution, production-case
accuracy, and the speed criterion are separate pending measurements.

## Fixed-point near accumulator

The fused route (`_bin_sparse.py`, #1003) keeps the near/far partition above and changes how near pairs are summed. Each pair adds `llrint(scale * w * mean)` to a per-bin 64-bit integer, with `scale` a power of two and $2^{60} \le \text{scale}\sum|w_l| < 2^{61}$. Integer addition commutes, so the sum is bitwise independent of launch order. One pair's error is at most half a unit, so at most $2^{-61}\sum|w_l|$.

The rounding is one-sided in practice. Weights span about 30 decades, and $\operatorname{sinc}^2$ means fall and have zeros across the near window, so 29-38% of pairs round to exactly zero and lose their whole value. That gives a systematic negative integral offset of $1.5\times10^{-13}$ to $2.0\times10^{-12}$ on the measured flushes, about 1% of the a-priori bound. A host replay with `np.rint` reproduces the measured offsets to about 0.2% (for example $-1.6194\times10^{-12}$ against $-1.6228\times10^{-12}$), and the agreement also indicates the device `llrint` rounds to nearest.

Fresh-context check (2026-11-06, `0db76d387`): `rederived`, no code defect. The verifier read the implementation before deriving, so this is not code-blind, and device behaviour rests on job 4059 and the kernel source. Near-classified pairs that the float32 test sends to the envelope get the same value as the tree.

Limits:

- Against the tree, per-bin relative deviation is at most $3.0\times10^{-9}$ with a floor of $10^{-6}$ of peak. Without the floor, a sample of about 700 bins per flush reaches $6.5\times10^{-8}$ on HOPG (at $2.6\times10^{-8}$ of peak) and $2.3\times10^{-6}$ on h-BN (at $6$ to $7\times10^{-11}$ of peak, the far tail near 280 keV). So agreement to float32 rounding holds above the floor, not at every bin.
- The launch grid reaches its $2^{31}-1$ block limit near $5.5\times10^{11}$ pairs, unguarded, and out of reach at the current batch target and grid.
- A zero-width bin would make `llrint` undefined where the tree gives NaN. Fixture edges are strictly increasing; not every grid path was audited.

