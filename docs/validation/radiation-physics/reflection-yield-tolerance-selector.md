# Reflection-yield tolerance selector

Validation: `reflection-yield-tolerance-selector`.

Independent fresh-context verification for issue #1173 (P3), 2026-11-13. The
derivation sections were written from the ledger row, the sibling rows
`line-hkl-batch`, `sinc-bin-integration`, `sinc-bin-far-envelope` and
`structure-factor`, and the cited sources, before the bodies of
`montecarlo/runner/reflection_cutoff.py` were read. The code comparison and the
numerics come after them.

## Claim, sources and conventions

The selector partitions the non-forbidden reflections into shells
$\lvert\mathbf g\rvert\in(G_k-\Delta,G_k]$ and measures each shell's in-band
incoherent line yield $s_k$. It stops at the first $k$ where at least two
consecutive shells have decayed and the geometric tail estimate is at most the
tolerance:

$$
\tau_k=\frac{s_k\,r_k/(1-r_k)}{\sum_{j\le k}s_j},
\qquad r_k=\frac{s_k}{s_{k-1}}\in(0,1).
$$

The rationale cites a shell ratio $\rho(G)=\bigl((G+\Delta)/G\bigr)^2
\exp\!\bigl(-2B\,\Delta(g^2)/16\pi^2\bigr)$, which falls with $G$.

Conventions: $\lvert\mathbf g\rvert=2\pi/d$,
$s=\sin\theta/\lambda=\lvert\mathbf g\rvert/4\pi$, amplitude Debye–Waller
factor $e^{-W}$ with $W=B s^2=B g^2/16\pi^2$ (`structure-factor`), so the
intensity carries $e^{-2W}$. The Waasmaier–Kirfel $f_0$ fit is valid for
$s\le6$ Å⁻¹.

## Independent derivation

### 1. Additivity over reflections

In the incoherent route (`line-hkl-batch`), the spectral density is a sum over
segments $n$, reflections $\mathbf g$ and polarizations $\lambda$ of
$\lvert A_{n\mathbf g\lambda}\rvert^2 L(E;E^{\rm res}_{n\mathbf g},a_n)$.
Nothing squares a sum over $\mathbf g$. The density for a list
$\mathcal H=\bigsqcup_k\mathcal H_k$ is therefore
$\sum_k\rho_{\mathcal H_k}(E)$, provided every per-line factor depends only on
its own $(n,\mathbf g)$ and on quantities that are the same in every call:

- the energy axis, so the bin edges and the line-keep window
  $E^{\rm res}\in(E_0-0.2\,\Delta E,\;E_1+0.2\,\Delta E)$;
- the tabulation grid for $\chi_{\mathbf g}$, $U_{\mathbf g}$, $\mu$ and
  $\delta$, which is built on the same padded window;
- the mosaic quadrature, which is deterministic in $(\mathrm{FWHM},N)$;
- the transport segments.

The tolerance claim needs no more than this. Shell yields then add exactly,
apart from float reassociation.

### 2. Integrated in-band yield on a tiling `bin-mean` axis

Nodes $E_i=E_0+h(i+\tfrac12)$, $i=0,\dots,N-1$, $h=(E_1-E_0)/N$, with
midpoint interior edges and reflected outer half-widths, give edges
$E_0+hi$. The first edge is $E_0\ge0$, so the 0 eV clamp is inactive. The
bins tile $[E_0,E_1]$ with width $h$. Each bin value is a bin mean
$\bar\rho_i=h^{-1}\int_{\rm bin}\rho$, so

$$
\int_{E_0}^{E_1}\rho\,\mathrm dE=h\sum_i\bar\rho_i
$$

exactly for every line handled by the exact route (`sinc-bin-integration`).

Beyond $K=$ `BIN_MEAN_EXACT_WIDTHS` $=64$ first-zero widths, production writes
the averaged envelope. By `sinc-bin-far-envelope`, each line's integrated
yield over any contiguous run of far bins then changes by at most
$3/(2\pi^3K^2)=1.2\times10^{-5}$ of that line's whole mass. Summed over
lines, the in-band error is at most $1.2\times10^{-5}$ times the total mass of
the kept lines. Kept lines also include those resonant in the $\pm20\,\%$
padding, which put only tails into the band. The pad adds at most 40 % of
span, so for a roughly flat line population the bound is
$\lesssim2\times10^{-5}$ of the in-band yield. The in-band yield is therefore
exact to $O(10^{-5})$ relative, not exactly. That is negligible against any
tolerance $\ge10^{-4}$. Float32 accumulation adds about $10^{-7}$
(`sinc-bin-integration`, CUDA). The in-band yield also omits the band tails
of lines resonant beyond the keep window. This is a production keep-window
property shared by every call, not an additivity defect.

### 3. Geometric remainder and its conservativeness

If $s_{k+j}=s_k\prod_{i=1}^{j}r_{k+i}$ with $r_{k+i}\le r_k<1$ for all
$i\ge1$, then

$$
T_k=\sum_{j\ge1}s_{k+j}\le s_k\sum_{j\ge1}r_k^{\,j}=\frac{s_kr_k}{1-r_k},
$$

with equality for an exactly geometric sequence. The estimate is an upper
bound **if and only if** the later ratios do not exceed the current one,
which is a property of the sequence and not of the formula.

**Smooth claim model.** For
$\ln\rho(G)=2\ln(G+\Delta)-2\ln G-2B(2G\Delta+\Delta^2)/16\pi^2$:

$$
\frac{\mathrm d\ln\rho}{\mathrm dG}
=-\frac{2\Delta}{G(G+\Delta)}-\frac{B\Delta}{4\pi^2}<0
\quad\text{for all }G>0.
$$

Both terms are negative. The multiplicity factor $((G+\Delta)/G)^2>1$ raises
$\rho$, but it *decreases* with $G$ toward 1, so it never makes $\rho$ grow.
The model ratio is strictly decreasing for every $B\ge0$, with no crossover.
The same holds for a linear (in-band, slab-restricted) multiplicity
$(G+\Delta)/G$. Values at $\Delta=2$ Å⁻¹:

| $G$ (Å⁻¹) | Si $B=0.46$, mult² | Si, mult¹ | MoS₂ $B=0.47$, mult² | MoS₂, mult¹ |
|---|---|---|---|---|
| 10 | 1.114 | 0.929 | 1.108 | 0.923 |
| 20 | 0.742 | 0.674 | 0.734 | 0.667 |
| 30 | 0.552 | 0.518 | 0.544 | 0.510 |
| 50 | 0.330 | 0.317 | 0.321 | 0.309 |
| 73 | 0.188 | 0.183 | 0.181 | 0.176 |

**What the model omits.** A shell's yield also carries $\lvert F\rvert^2$
through $f_0(s)^2$, the per-line kinematic factors, and the discrete lattice
count per shell.

- **$f_0(s)^2$.** It enters the PXR-like weight. For Si it is not
  ratio-monotone. Using the xraydb Waasmaier–Kirfel $f_0$, the model
  $\rho\cdot[f_0(G+\Delta)/f_0(G)]^2$ *rises* for $G=13$–$16$ Å⁻¹
  (0.642 to 0.648). This is the loss of the L-shell part leaving the
  slower-falling K-shell core. Mo shows no rise; S rises slightly at 18 Å⁻¹.
  A Mott–Bethe CBS-like weight $(Z-f_0)^2/g^2$ is monotone for all three.
- **Lattice counting.** Shells of a cubic lattice collect a few dozen
  discrete $h^2+k^2+l^2$ values with irregular multiplicities. The diamond
  extinctions ($h+k+l=4n+2$ and mixed parity removed, $\lvert F\rvert^2=32f^2$
  or $64f^2$ otherwise) are uniform in reciprocal space. They do not create
  systematic dips, but they reduce the count and so raise the relative
  fluctuation. The in-band selection restricts $g_\parallel=\mathbf g\cdot\hat
  v$ to a slab, so the shell count grows linearly in $G$ instead of
  quadratically. A zone-axis beam also puts the points on discrete Laue
  planes. Both increase the fluctuation.

## Numerical test of conservativeness (independent model)

The script uses its own Si (diamond, $a=5.431$ Å) and 2H-MoS₂ ($a=3.161$,
$c=12.295$ Å, $z_{\rm S}=0.621$) structure factors, xraydb $f_0$, $e^{-2W}$,
and shells of $\Delta=2$ Å⁻¹ out to $4\pi\cdot6$ Å⁻¹. It was written without
the PyRITE helpers, and the script lives in the verifier scratchpad. The shell
weight is $\lvert F\rvert^2e^{-2W}$, either over all of reciprocal space (3D)
or in an in-band proxy slab $0.05<\lvert\mathbf g\cdot\hat v\rvert\le6.5$
Å⁻¹. The 6.5 Å⁻¹ slab is $E=10$ keV at 90° emission for 300 keV. The beam
$\hat v$ is a generic direction or exactly [001]. This is a proxy for the
production kernels, not a substitute; it tests the *sequence* property the
claim relies on.

Shell-ratio increases, i.e. violations of "non-increasing", occur throughout
$G\ge10$ in every variant. Examples for Si 3D: 0.418 to 0.840 at 14 to 16;
0.575 to 0.663 at 24 to 26; 0.344 to 0.370 at 42 to 44. For the Si [001]
slab: 0.415 to 0.565 at 28 to 30, and 0.320 to 0.393 at 44 to 46. The
selector was emulated (two consecutive decays, $\tau_k\le$ tol) over a tol
sweep, with the true tail summed to the $f_0$ cap:

| variant | worst true/tol, tol $\le0.2$ | worst true/tol, tol $\le10^{-2}$ | worst true/tol, tol $\le10^{-3}$ |
|---|---|---|---|
| Si 3D | 2.99 ($G=14$, tol 0.048) | 1.00 ($G=24$) | 0.81 |
| Si slab, generic | 2.87 ($G=14$, tol 0.022) | 1.14 ($G=28$, tol $6.5\times10^{-4}$) | 1.17 |
| Si slab, [001] | 3.04 ($G=14$, tol 0.022) | **1.48** ($G=28$, tol $5.1\times10^{-4}$) | 1.48 |
| MoS₂ 3D | 1.21 ($G=16$, tol 0.10) | 0.92 | 0.92 |
| MoS₂ slab, generic | 1.26 ($G=20$, tol 0.012) | 0.94 | 0.91 |
| MoS₂ slab, [001] | 1.32 ($G=20$, tol 0.012) | 0.98 | 0.98 |

The smallest estimate-to-true ratio for single-shell estimates at $G\ge24$ is
0.67 (Si [001] slab, $G=28$). The two-decaying-shell guard does not stop the
premature $G=14$ Si stops: the Si shell sequence falls monotonically through
8–14 Å⁻¹ (the $f_0$ shoulder), and the following shell ratio rises to about
0.84.

A variant that uses $\max(r_k,r_{k-1})$ in place of $r_k$ was conservative in
all six variants for tol $\le10^{-2}$ (worst true/tol 0.93). For tol up to
0.2 its worst was 1.31, at the Si $G=14$ shoulder.

The proxy agrees with the ledger's prototype at the level the prototype
resolves. The Si 3D model ratio at $G=30$ is 0.53, against 0.51 measured, and
its single-shell estimates are 0.93–1.0 of truth near $G=22$–24, against the
prototype's $6.6\times10^{-3}$ vs $6.67\times10^{-3}$. The ledger's "DW
prediction 0.51" for g28→30 is the DW factor alone,
$\exp(-0.92\cdot116/16\pi^2)=0.509$. The stated model with multiplicity²
gives 0.584. The agreement therefore does not test the stated rationale;
$f_0$ and kinematic factors cancel the multiplicity growth.

## The cap and the shell width

- $F_{0}$ range: $s\le6$ Å⁻¹ means $\lvert\mathbf g\rvert\le4\pi\cdot6=
  75.398$ Å⁻¹. This is correct.
- The production set is narrower than this cap.
  `dominant_reflections` stops at the first family whose metric
  $\lvert S\rvert e^{-W}/g^2$ is below $10^{-9}$ of the strongest. For Si
  ($B=0.46$) the shells therefore end at $(64,66]$ Å⁻¹, with 1328 members
  against 10124 in $(62,64]$. The last shell is partial, so its $r$ is biased
  low. The yields there are negligible (about $10^{-12}$ relative), so this
  matters only for the reported `tail_fraction` at an unconverged cap. A
  non-multiple cap such as 75.4 likewise ends in a narrower last shell.
- $\Delta=2$ Å⁻¹. The code comment says "one Debye–Waller e-fold". The DW
  decrement per shell is $2B(2G\Delta+\Delta^2)/16\pi^2=0.70$ e-folds at
  $G=30$ and 0.47 at $G=20$. "Halving" (0.69 e-folds) is right near
  $G\approx30$; "one e-fold" is loose. With 0.4–0.7 e-folds per shell, two
  decaying shells span only about 1–1.4 e-folds, which is why lattice
  fluctuations of 10–40 % in $r$ matter.

## Comparison with the implementation

`geometric_tail_fraction(yields)` returns
`last * ratio / (1 - ratio) / kept` with `ratio = last / previous` and
`kept = yields.sum()`. This equals $\tau_k$, including $s_k$ in the
denominator, and matches the derivation. Its guards:

- It returns `inf` for fewer than two shells, nonfinite or negative entries,
  `kept <= 0`, `previous <= 0`, or `ratio` outside $(0,1)$.
- **It returns `0.0` when `last == 0`.** That is $r=0$, outside the claim's
  $r\in(0,1)$. In `select_reflection_cutoff`, a zero shell after a decaying
  one counts as the second consecutive decay, so the selector stops with
  `tail_fraction = 0` regardless of the shells beyond. Reproduced: yields
  `[1, 0.5, 0, 0.4, 0.3]` at tol $10^{-3}$ stop at the third shell,
  `converged=True`, tail 0. A zero in-band yield needs a nonempty shell whose
  every line lies outside the keep window. That is implausible above the
  sparse low-order shells, but possible there, for example a low shell of only
  $g\perp\hat v$ reflections whose $E^{\rm res}=0$ lines are dropped.

`select_reflection_cutoff` increments `decaying` on a strict decrease and
resets it otherwise. It stops when `decaying >= 2` and `tail <= tol`, and
reports `converged=False` with every shell when the list runs out. This
matches the ledger's stopping rule.

`reflection_shells` takes edges $\Delta,2\Delta,\dots$ and uses
`searchsorted(edges, g - 1e-9 g_max)`, i.e. upper-inclusive shells
$(G-\Delta,G]$. Members are exactly `dominant_reflections(..., n_families=1e9)`
at the cap, so the union up to $G$ is that function's set at $G$. Shells are
disjoint and cover the set, which additivity needs. Si (111) has
$\lvert\mathbf g\rvert=2.004$, so the first nonempty shell is $(2,4]$ with 44
members: {111} 8, {220} 12, {311} 24. This matches.

`band_bin_grid` builds nodes at $E_0+h(i+\tfrac12)$, at least four of them.
This is the tiling of §2. `bin_axis` uses
`characteristic._energy_bin_edges_and_widths`, the same convention.

Additivity check of the incoherent route:

- The keep mask in `_batched.py` is
  `cxr_electron & (E_res > lo_keep) & (E_res > 10) & (E_res < hi_keep)`, with
  the pad equal to 0.2 of the axis span. It depends on the axis, not on the
  reflection list.
- The tabulation window in `cxr/_setup.py` uses the same padded axis, and
  `_cxr_tabulation_grid` depends only on the elements.
- The mosaic quadrature is deterministic.
- The `_flat_energy_keep` union-over-rows truncation is coherent-only.
- `_admit_line_tables` can refuse a long list but does not change values.

No reflection-list-dependent normalization was found, so additivity holds.
The gate's `reflection_chunks` relies on the same property. The layered-case
pilot in `checks/defaults_spectral_gate_642.py` scores one radiator with the
others set to `None`. Whether a `None` radiator still contributes its layer's
absorption was not checked here; that is harness scope, not this claim.

## Tests

`tests/montecarlo/runner/test_reflection_cutoff.py` passed (10 passed,
2026-11-13). The tests pin:

- the band tiling;
- exactness on a geometric sequence;
- refusal before decay;
- the shell partition against `dominant_reflections`;
- the unconverged cap.

They do not pin the following:

- **The two-decaying-shell guard.** In
  `test_selection_stops_at_the_first_converged_shell_and_skips_low_order_dips`,
  no single-decay shell ever has $\tau\le$ tol. The tail estimates at the
  dips are 0.099 and 0.136 against tol 0.02, so the test passes with the
  guard removed. A sequence with a lone deep dip, such as
  `[1, 1.2, 0.001, 1.0, ...]`, would pin it.
- the `last == 0` path;
- additivity, i.e. that shell spectra sum to the one-call spectrum on the
  same axis;
- the conservativeness rationale.

No test exercises production kernels.

## Adjudication

- Formula, guards (except $r=0$), shell partition, band tiling and additivity
  match the independent derivation.
- In-band bin-mean yield is exact to $O(10^{-5})$ relative under the
  production far envelope, not exactly.
- The stated smooth model $\rho(G)$ is strictly decreasing for all $G$. That
  is proven above, and the multiplicity factor does not oppose it.
- The claim's conclusion, "the geometric estimate is conservative once the
  sequence is DW-dominated", does **not** follow for measured shell
  sequences. Real $s_k$ carry $f_0(s)^2$, which is non-monotone for Si at
  13–16 Å⁻¹, and shell-to-shell lattice-count fluctuations of 10–40 % in $r$.
  The two-shell guard does not exclude either. In an independent proxy the
  selector stops with the true omitted fraction up to 1.48× tol at tol
  $\le10^{-2}$ ($G=28$, Si [001]). At tol of a few percent it reaches 3×,
  at the Si $f_0$ shoulder at $G=14$.
- The ledger's "DW prediction 0.51" omits the multiplicity factor that the
  rationale names.

First divergent item: the word "conservative" in the claim. The estimate is
an unbiased-to-optimistic single-ratio extrapolation with typical error
$\lesssim1.5\times$ above $G\approx24$ Å⁻¹, not a bound. The second item is
code behavior outside the claim: $r=0$ returns tail 0 and can stop.

Verdict: **discrepancy** (claim wording plus the $r=0$ edge). No numerical
fault was found in the estimator formula.
