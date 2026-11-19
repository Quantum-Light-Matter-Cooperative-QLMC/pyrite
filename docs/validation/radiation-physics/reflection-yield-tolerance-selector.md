# Reflection-yield tolerance selector

Validation: `reflection-yield-tolerance-selector`.

Independent fresh-context verification for issue #1173 (P3), 2026-11-13. The
derivation sections were written from the ledger row, the sibling rows
`line-hkl-batch`, `sinc-bin-integration`, `sinc-bin-far-envelope` and
`structure-factor`, and the cited sources, before the bodies of
`montecarlo/runner/reflection_cutoff.py` were read. The code comparison and the
numerics come after them.

The original single-ratio derivation and adjudication below are historical.
The current max-of-two implementation and superseding review appear under
[Current re-verification](#current-re-verification-2026-11-14). The historical
missing-test and zero-shell findings do not describe the current code.

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
the kept lines. Kept lines include resonances in the $\pm20\,\%$ padding,
which can contribute only tails to the band. Converting this whole-line-mass
allowance to a relative in-band allowance requires the ratio of complete
kept-line mass to retained in-band mass. That ratio is not bounded for an
arbitrary narrow band. Bin-width weighting integrates the represented means;
accuracy inherits route-specific approximations and accumulation errors. The
production keep window also drops tails of lines outside its padded span,
a shared approximation rather than an additivity defect.

### 3. Geometric remainder and its conservativeness

If $s_{k+j}=s_k\prod_{i=1}^{j}r_{k+i}$ with $r_{k+i}\le r_k<1$ for all
$i\ge1$, then

$$
T_k=\sum_{j\ge1}s_{k+j}\le s_k\sum_{j\ge1}r_k^{\,j}=\frac{s_kr_k}{1-r_k},
$$

with equality for an exactly geometric sequence. The estimate is an upper
bound **if** the later ratios do not exceed the current one,
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

## Production scoring (#1173 P4)

For explicit fixed bands, the spectrum phase applies the selector on the production line axis rather
than a separate coarse pilot. Each shell's incoherent density is evaluated
with the case's own line quadrature and integrated against the axis weights:
trapezoid weights for `node`, bin widths for `bin-mean`. By additivity
(section 1), the sum of the scored shells is the line spectrum of the selected
set on that axis. The selection therefore costs about one shell beyond its
stop. A pilot followed by a second scoring pass would cost about twice as
much.

### Shared automatic selection band (#1243, v5)

Implementation-context change, independently rederived below. Automatic
`kinematic-ceiling` and `resonance-population` payloads carry the same start
and clamped ceiling even when their realized production stops differ. The
selector now integrates shell lines on 500 eV `bin-mean` bins tiling that
common band. The existing closed-form sinc-bin mass supplies the integral;
the bins do not approximate the line shape at their centers. Continuum share
and background use the same band's endpoints, with the existing logarithmic
continuum quadrature. Thus all three terms of the net metric refer to one
band, and the selected set no longer depends on the measured stop or its
fallback extension.

The selected set is then evaluated in admitted reflection chunks on the
production axis. Only this pass contributes to its truncation audit; reference
evaluation has no production audit accumulator. The returned spectrum and
audit therefore refer to the same reflections. The selection record adds
`reference_band_eV` and `reference_quadrature`; its shell integrals describe
the reference band, while `quadrature` still names production scoring.
Explicit fixed axes and routes with grouped flights, a truncated sinc
profile or explicit numeric node-support windows retain direct shell scoring.
The bin reducer's route-specific approximations still apply; exact integration
of a represented line is not a new accuracy guarantee. No new transport or
random draws occur.

Regression: `test_automatic_bandwidths_select_the_same_reflections` uses one
six-history Si transport. Before the change, production stops of 2 and 8 keV
selected 238 and 944 reflections; afterward their reference shell records
are identical. This pins band independence, not a universal tail bound.
The automatic-axis generation is `measured-shell-cutoff-v5`; historical
qualification of earlier generations below does not certify this change.

Fresh-context read-only review (2026-11-18): **rederived**, conditional on the
existing line-kernel, ceiling and tail-bound claims. The verifier derived
$L_k(B)=\sum_i\Delta E_i\bar\rho_{ki}$ and $s_k=L_k(B)-S_k(B)$ with common
$C_r(B)$ before inspecting the implementation. Shared transport, material
support, quadrature and $B=[a,C]$ give the same shell sequence for either
production stop. Reference bins tile $B$ within roundoff (observed endpoint
deviation at most $5.7\times10^{-14}$ eV); their padded keep window contains
the entire band. A layered in-memory probe confirmed one active radiator per
pilot, separate admitted chunk sizes and exactly one production evaluation
per selected reflection. Reference evaluations clear the audit accumulator;
production retains its own ceiling window and audit. The review found that
explicit numeric `node_support_cutoff` cannot run bin-mean pilots; its new
exclusion and three compatibility anchors close that regression. Four focused
physics/selection anchors and the three compatibility anchors passed in the
review. New-anchor CI is unobserved; no human sign-off is inferred.

Checked on CPU: splitting one call into halves, or into the selector's shells,
reproduces the single-call spectrum to $3\times10^{-15}$ of peak. The
selector's own sum matches the single call of its selected set to $10^{-9}$
relative (`test_reflection_selection_spectrum.py`). The automatic line axis
has to exist before the selected set is known, so its bandwidth uses the
$\lvert g\rvert$ cap (`reflection-selection-cap-v1` in the policy sources).
Diffracted bremsstrahlung and Kossel terms are a separate two-beam model and
keep the case's build-time reflections. Coherent emission, line roulette,
temporal profiles, channeling, two-beam transition radiation and streamed
block scoring are refused when selection is explicitly requested. Since the
user-approved #1173 default flip, an automatic request falls back to the
four-family set on those routes. Catalog-pinned sets remain pinned. The
`CXR_REFLECTION_SELECTION_MODEL` generation is `measured-shell-cutoff-v3`;
both dataset and case identities carry it. v3 (#1200) sizes each shell's line
calls from the radiator's tabulation length (2 GiB of tables at the backend
precision, 64--8192 reflections) instead of a fixed 512, and refuses a call
the device budget cannot hold before the first shell is scored. Line spectra
add over reflections, so this moves only the float summation order (at most
4e-8 of the peak on the frozen #642 scenes, with identical selected sets
and tail estimates). It does not alter the shell estimator or upgrade this
claim's validation status.

### Electron-prefix selection (#1281, v6)

Implementation-context change; fresh-context review below (`rederived`). Every shell
quantity the stop reads is an incoherent sum over histories: the line yield
$L_k=\frac1N\sum_{i<N}\ell_{ki}$, the continuum share $S_k$ and the
background $C_r$ (per primary, same band $B$). The stop is a deterministic
function of the sequence of ratios $L_{k+1}/L_k$ (or $s_{k+1}/s_k$) and of
$\sum_j s_j/C_r$, so it is invariant to a common normalization of the three
terms. v6 evaluates all three on the prefix $i<n$, $n=\min(N,
\texttt{SELECTION\_HISTORIES})$, of the line population (and $\min(N_e^{\rm
brem},\texttt{SELECTION\_HISTORIES})$ of the continuum population), with
the matching per-primary normalization. On counter-addressed transport cores
(per-electron, CUDA) the prefix is the same set of independent histories for
every $N\ge n$, so the selected set is identical for every $N\ge n$ at fixed
seed and fixed $\min(N_e^{\rm brem},n)$. An energy spread builds transport
tables over an $N$-dependent range, so there the agreement is only to
interpolation round-off. The lockstep core (CPU default, grooved runs) shares
one step-major generator, so its prefix histories depend on $N$.
The production spectrum then sums the selected reflections over all $N$
histories, so conditional on the selected set it is the same estimator as v5.

Consequences:

- **Cost.** Selection costs $O(n)$ line evaluations per scored shell, not
  $O(N)$; the production pass is unchanged and remains $O(N)$ (#1295, #1296).
- **Noise.** The cutoff inherits the sampling noise of $n$ histories, which no
  longer shrinks as $N$ grows. The tail estimate was already an extrapolation,
  not a bound; with a prefix it is also a finite-sample estimate. A cutoff
  near the tolerance can therefore differ from the full-population cutoff;
  this is a documented difference, not a bias of the scored spectrum.
- **Limit.** Exactly v5 when both $N_e$ and $N_e^{\rm brem}$ are within the
  budget (no prefix is taken); otherwise only the over-budget population is
  restricted.

Regression: `tests/montecarlo/runner/test_reflection_selection_spectrum.py::test_selection_scores_an_electron_prefix_then_every_history`
pins that a prefix selection equals the full selection on the prefix
population, and that the returned spectrum is the selected set scored on
every history. It pins the `elec_id` cut, the normalization and the
full-history rescoring on one transport, not cross-$N$ invariance or unequal
line/continuum prefixes.

Streamed runs (live blocks and artifact replay, `streamed_selection.py`) hold
the leading whole-electron blocks until a row of an electron at or past the
prefix arrives, select on those rows through the same
`select_batch_reflections`, then score every block with the selected set and
remove its lattice share from the continuum, as the retained route does.
Live streaming refuses the lockstep core, so its prefix is always
counter-addressed. `tests/montecarlo/transport/test_streamed_selection.py`
pins the streamed record equal to the retained one and spectra within
summation order for blocks of 1, 3 and 1000 electrons, and a replayed held
block committing once.

Evidence (`qlmc:~/pyrite-1281/out-cal`, `out-stab`; RTX 5080, code
`e52bd7465`, profile `cleo_sub_1281`, intrinsic spectra):

| Case | N | Seeds | Prefix arms | Cutoff / reflections | Same as full |
|---|---|---|---|---|---|
| MoS₂ 30 keV, 15° | 8192 | 0 | 1024, 4096 | G20 / 11,924 | yes |
| MoS₂ 30 keV, 15° | 4096 | 0, 1 | 1024 | G20 / 11,924 | yes |
| MoS₂ 100 keV, 57° | 4096 | 0, 1 | 1024 | G22 / 15,930 | yes |
| MoS₂ 1000 keV, 15° | 4096 | 0, 1 | 1024 | G20 / 11,924 | yes |
| WS₂ 30 keV, 15° | 4096 | 0, 1 | 1024 | G16 / 6,018 | yes |
| WS₂ 100 keV, 57° | 4096 | 0, 1 | 1024 | G16 / 6,018 | yes |

Tail estimates agree within about 3 % between arms. At v6, HOPG and h-BN
pinned catalog (00l) sets and never ran the selector; since v7 they select
on a fiber-texture domain (below). Measured cost on MoS₂ 30 keV
(709,999 segments at N = 8192): selection ≈ 12 s + 31 ms per scored history;
production ≈ 28 ms per history. The automatic-axis generation is
`measured-shell-cutoff-v6`.

### Fiber-texture candidate domain (#1301, v7)

**Claim.** For a catalog crystal with `texture = "fiber"` and primitive
reciprocal axis $a$ (reduced `surface_hkl`), the selector's candidate set is
$\{na : n\in\mathbb Z\setminus\{0\},\ \lvert naA\rvert\le G_{\max}\}$, filtered
by the existing structure-factor metric floor. Shells, the geometric tail,
the net-metric continuum shares and the reconstructed continuum-share set all
use that one domain; the estimator and tolerance are unchanged.

**Assumptions.** Uniform random grain azimuths about $a$ (HOPG; pyrolytic,
turbostratic h-BN by user decision 2026-11-18), so only $g\parallel aA$ remains
a coherent line in the catalog's axial model. The scalar `B_ang2` is then the
axial projection (B33 for $a=(001)$), applied only to $g\parallel c$
(`hopg-debye-waller-00l`, `hbn-debye-waller-00l`). Mosaic FWHM is unchanged.

**Derivation.** With nonsingular reciprocal basis $A$,
$hA=\lambda aA \iff (h-\lambda a)A=0 \iff h=\lambda a$. For primitive integer
$a$, integer $h$ on that line are exactly $h=na$, $n\in\mathbb Z$: if
$h=\lambda a$ with $\lambda=p/q$ reduced, $q$ divides every $a_i$, so $q=1$.
Collinearity is therefore exact in Miller space for any (non-orthogonal)
cell; no Cartesian angle tolerance enters. $\lvert naA\rvert=\lvert n\rvert\lvert aA\rvert$
gives the count $\lfloor G_{\max}/\lvert aA\rvert\rfloor$.

**Limits.** (i) Single-crystal texture: the enumeration is the historical
full grid, unchanged. (ii) HOPG/h-BN (P6₃/mmc): odd $l$ is extinct by the
structure factor, so at the default 2 Å⁻¹ shell width the first two shells are exactly the retired
$(00\bar2),(002)$ and $(00\bar4),(004)$ pins. (iii) A cap below $\lvert aA\rvert$
gives no candidates; the metric-floor maximum defaults to zero instead of
raising.

**Identity.** `CXR_REFLECTION_SELECTION_MODEL = measured-shell-cutoff-v7`.
Each radiator record carries `texture`, `texture_axis_hkl` and
`selected_set_sha256` over the sorted selected set and texture domain;
dataset/case identities carry `crystal_texture` for fiber crystals.

**Evidence (job 13676, qlmc RTX 5080, code `ec0365c5c`, intrinsic spectra,
4096 histories, tilt 15°).** Frozen pre-change transports scored over every
allowed basal shell to the form-factor cap; the existing net-metric stop
replayed on the measured rows:

| Case | Selector stop | Tail | Retired-pin omission of full basal yield |
|---|---|---:|---:|
| cleo_sub HOPG 30 keV | (0010), G = 10 Å⁻¹ | 1.02e-4 | 0.758 % |
| cleo_sub HOPG 1 MeV | (008), G = 8 Å⁻¹ | 1.25e-4 | 0.102 % |
| bells_45deg HOPG 30 keV | (006), G = 6 Å⁻¹ | 3.34e-4 | ≈100 % (1 − 1.6e-9), but full basal yield is 2.3e-18 against a 5.7e-7 in-band background: negligible |

These are paired finite-population measurements on these scenes, not an
omitted-yield bound.

**Anchors.** `tests/montecarlo/runner/test_fiber_reflection_selection.py::test_frozen_fiber_shells_select_and_reconstruct_the_same_basal_set`
replays the selector on the job-13676 shell rows
(`tests/data/reflection_selection_fiber_1301.json`) and pins the stop, tail,
texture record and reconstructed set.
`tests/materials/test_crystal_texture.py` pins the (00l)-only domain, the
retired pins as the first two shells, a non-orthogonal axis, and an unchanged
single-crystal enumeration.

### Fiber-texture re-verification (v7, #1301, 2026-11-18)

This is a fresh-context review. The verifier derived the result below from the
claim statement before reading the `_reflection_families` fiber branch.

**1. Collinearity in Miller space.** Write $\mathbf g(h)=hA$, where $h$ is an
integer row vector and the rows of $A$ form the reciprocal basis, with
$\det A\neq0$. Then $hA\parallel aA$ means $hA=\lambda aA$ for some real
$\lambda$, which is equivalent to $(h-\lambda a)A=0$. Because $A$ is
invertible, this holds exactly when $h=\lambda a$. The argument uses neither
orthogonality nor a metric, so an oblique or triclinic cell needs no angle
tolerance.

**2. Primitive axis, integer multiples.** Let $a$ be primitive, so
$\gcd(a_1,a_2,a_3)=1$. By Bézout's identity there are integers $c_i$ with
$\sum_i c_ia_i=1$. Hence $\lambda=\sum_i c_i(\lambda a_i)=\sum_i c_ih_i$ is an
integer, and the integer points on the line are exactly
$\{na:n\in\mathbb Z\}$. Since
$\lvert naA\rvert=\lvert n\rvert\,\lvert aA\rvert$, the nonzero candidates
inside the cap are $0<\lvert n\rvert\le\lfloor G_{\max}/\lvert aA\rvert\rfloor$.
The set is symmetric in $n$, so $\pm a$ give the same domain. If the axis
were not primitive, for example $2a$, only even multiples would remain and
half the line would be lost. The reduction step is therefore essential.

**3. P6₃/mmc $(00l)$ extinctions.** Take the catalog sites of graphite (C at
$z=0,\tfrac12,0,\tfrac12$) and of h-BN (B and N each at $z=0,\tfrac12$). For
$(00l)$, each species has phase sum
$\sum_j e^{2\pi i l z_j}=1+e^{i\pi l}$, which is $0$ for odd $l$ and $2$
for even $l$. With anomalous $f',f''$ included, a species' form factor still
multiplies a vanishing sum. Odd $l$ is therefore extinct for every energy.
The fiber domain $\{(0,0,n)\}$ thus keeps only $l=\pm2,\pm4,\dots$ after the
metric floor. The orders sit at $\lvert\mathbf g_{00l}\rvert=2\pi l/c$, which
is $0.936\,l$ Å⁻¹ for HOPG ($c=6.711$ Å) and $0.943\,l$ Å⁻¹ for h-BN
($c=6.661$ Å). With the default $2$ Å⁻¹ shell width, the first two non-empty
shells are therefore exactly $\{(00\bar2),(002)\}$ and
$\{(00\bar4),(004)\}$, the retired pins. One shell holds one order until
$l\approx30$.

**4. Thermal factor.** All candidates satisfy $\mathbf g\parallel\mathbf c^*$.
The scalar factor
$\exp[-B(\lvert\mathbf g\rvert/4\pi)^2]=\exp(-B\,l^2/4c^2)$ then equals the
$B_{33}$ form in `hopg-debye-waller-00l` and `hbn-debye-waller-00l`. Neither
material ever receives $B_{33}$ on an in-plane vector. This holds because the
domain is fixed before the Debye–Waller weight is applied.

**5. Comparison with the implementation.** The comparison used the
uncommitted diff on `issue-1301-fiber-reflections`.

| Item | Derived | Code | Result |
|---|---|---|---|
| Axis | primitive $a$ from `surface_hkl` | `load_crystals`: `texture_axis_hkl = spec.cut`; `cut` reduces `surface_hkl`, which fiber requires; `_reflection_families` reduces again | match; `(0,0,2)` gives `(0,0,1)` |
| Domain | $na$, $0<\lvert n\rvert\le\lfloor G_{\max}/\lvert aA\rvert\rfloor$ | `np.arange(-count, count+1)[:, None] * axis`, then the same `0<\lvert g\rvert\le G_{\max}` keep | match |
| Order | domain before DW and metric | `hkl` is built before `g_vec`; DW and `metric` follow unchanged | match |
| Empty cap | no candidates, no error | `max(..., default=0.0)` and the `m > 0.0` filter in `iter_reflection_shells` | match; for single crystals, `m > 0` is redundant when the floor is positive, so behavior is unchanged |
| One domain | shells, tail, shares and reconstruction | all go through `iter_reflection_shells` and then `_reflection_families`, keyed by `CRYSTALS[crystal]`: `select_case_reflections` (shell yields and `continuum.share`), `lattice_share.scored_reflections` (rebuild) and `reflection_shells`/`dominant_reflections` | match |
| Identity | texture and axis in identities | `_selected_set_record` hashes the sorted `hkl_list` with texture and axis; `_structure_factor_identity` adds `crystal_texture`; `CXR_REFLECTION_SELECTION_MODEL` is `v7` | match |
| Validation | fiber needs a surface axis, and pins must be parallel | `_crystal_texture` and `CrystalSpec.__post_init__` use `np.cross(hkl, surface) == 0`, which is exact for integers and equivalent by step 1 | match |

**6. Independent probes.** These live in
`scratch/verify1301/probe.py` and `probe2.py`, which are not committed. For
the comparison, the probe builds its own reciprocal basis from the lattice
parameters. It scans the full integer grid, keeps vectors whose Cartesian
cross product with $aA$ vanishes, and computes its own phase sums.

- HOPG and h-BN at $G_{\max}=12$: the probe finds 24 parallel vectors. Its
  even-$l$ subset equals the union of `iter_reflection_shells`. The first
  three shells are $\pm2$, $\pm4$ and $\pm6$ at uppers 2, 4 and 6.
- Hypothetical axes on the HOPG cell: for `(2,2,0)`, `(1,0,0)`, `(1,1,2)`,
  `(2,-1,3)` and `(0,0,2)`, every vector the implementation returns lies in
  the probe's parallel set. The only vectors it omits are $(00l)$ with odd
  $l$, $\pm(2,\bar1,3)$ and $\pm(3,6,\bar3)$. Each of these has phase sum
  $0$, so it is extinct.
- Triclinic cell ($3.1, 4.3, 5.7$ Å; $81^\circ, 97^\circ, 112^\circ$) with
  axis `(1,2,-1)`: the implementation returns $\pm1,\pm2,\pm4$ multiples. The
  probe's set adds $\pm3$, which is extinct for this basis.
- Frozen job-13676 rows: the probe re-implemented the net-metric stop from
  §3 and the net-metric section, using
  $r=\max(\lvert s_k/s_{k-1}\rvert,\lvert s_{k-1}/s_{k-2}\rvert)$,
  denominator $\sum s_j+$ background, and two decaying shells. It reproduces
  the recorded cutoffs, 10, 8 and 6 Å⁻¹, with bit-identical tails
  `1.0236768e-4`, `1.2542166e-4` and `3.3350115e-4`. The retired-pin
  omissions recomputed from the rows are 0.7581 % and 0.1016 %, matching the
  table.

**7. Findings. None changes the claimed physics.**

1. *Markers.* The bullet names three functions as code owners, but they
   carry no `Validation: reflection-yield-tolerance-selector` marker:
   `materials/_parse.py::_crystal_texture` (line 134),
   `montecarlo/runner/reflection_cutoff.py::iter_reflection_shells` (line
   124, which also carries the new `m > 0` domain filter) and
   `::_selected_set_record` (line 600). Of the named owners, only
   `crystal.py::_reflection_families` (line 944) has one.
   `tests/dev/test_validation_ledger.py` passes because it reads the row's
   `**Code:**` field and not this bullet. The task owner should either add
   the markers or move the names into the `**Code:**` field.
2. *bells_45deg row.* In the table, the omission column for this row gives
   only the absolute yield. The frozen record
   (`omitted_fraction_by_pins = 0.99999999839`) shows that the retired pins
   omitted essentially **all** of that scene's basal yield, because the
   $(002)$ in-band yield is $0$ and the $(004)$ yield is $3.7\times10^{-27}$.
   The yield is still negligible next to the $5.7\times10^{-7}$ background.
   The wording should give both numbers.
3. *Shell-width dependence.* "The first two shells are exactly the retired
   pins" holds at the default $2$ Å⁻¹ width. That is the width used by the
   anchors and by production, but the statement could say so.
4. *Fixed-count routes.* For routes outside the selector, `crystal_params`
   and `dominant_reflections(n_families=4)` now give
   $(002),(004),(006),(008)$ for HOPG and h-BN, where the pins gave two
   families. This is consistent with "candidate domain before selection",
   but no page states it.
5. *Robustness (direct construction only).* `CrystalSpec(surface_hkl=(0,0,0),
   texture="fiber")` passes `__post_init__`. The catalog decoder rejects a
   zero vector, but this construction would overflow in
   `int(np.floor(g_max / 0))`.
6. *Identity sign.* `texture_axis_hkl` records the signed axis. The spellings
   $(0,0,1)$ and $(0,0,\bar1)$ produce the same set under different digests.
   This can only over-invalidate a cache.

**8. Tests run.** The command was
`uv run pyrite-dev test tests/materials/test_crystal_texture.py tests/montecarlo/runner/test_fiber_reflection_selection.py tests/dev/test_validation_ledger.py tests/campaign/test_numerics_groups.py`,
with 45 passed.

**9. Verdict (v7).** `rederived`. The fiber-texture candidate domain, its
reduction to the first two shells for P6₃/mmc, the B33 scope and the single
shared domain match the independent derivation. The frozen-transport replay
reproduces the anchored stops exactly. The anchors are green locally, but no
CI run was observed for them. Once that CI run is observed, the row can move
to `anchored`. Findings 1 and 2 should be addressed in wording or markers
before the task owner applies the status. No human sign-off is implied.

## Historical adjudication (2026-11-13)

- Formula, guards (except $r=0$), shell partition, band tiling and additivity
  match the independent derivation.
- The bin-width sum integrates represented bin means. The whole-line-mass
  far-envelope allowance does not imply a universal relative in-band bound
  (corrected in section 2 and the current re-verification below).
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


## Current re-verification (2026-11-14)

A fresh read-only verifier derived the expression from the source conventions
before inspecting implementation bodies. For fixed transport, direction, axis,
material/absorption factors and quadrature weights,

$$
S_k=\sum_{j\le k}s_j,\qquad
r_k=\max\!\left(\frac{s_k}{s_{k-1}},\frac{s_{k-1}}{s_{k-2}}\right),\qquad
\widehat T_k=\frac{s_kr_k}{1-r_k},\qquad
\tau_k=\frac{\widehat T_k}{S_k}.
$$

Shell yields have units photons per primary per steradian; $\tau_k$ is
dimensionless. The remainder is exact for a genuinely geometric continuation.
The sufficient condition that every future ratio is at most $r_k<1$ is not
established by the last two observed ratios. Independently,

$$
\frac{d\log\rho}{dG}
=-\frac{2\Delta}{G(G+\Delta)}-\frac{B\Delta}{4\pi^2}<0
$$

for $G,\Delta>0$, $B\ge0$. This smooth-envelope result does not establish
monotonicity of actual lattice-shell yields.

The formula, kept-yield normalization, zero/invalid-yield guards and cap
exhaustion match the current code. Shell scoring preserves the complete
absorption stack while changing only the active radiator and reflections.
No cross-reflection normalization was found. The original discrepancy is
superseded for the fixed estimator; no numerical implementation fault was
found in this review.

The review required the wording corrections now applied above: whole-line
error versus relative in-band error, a sufficient rather than necessary ratio
condition, and extrapolation rather than a bound. The ledger's old assertion
that every tail estimate was within 3% of measured omission was also removed:
film/substrate estimates refer to their own yields, whereas the recorded
layered gap refers to their sum. For seed 642501 the weighted combined estimate
is $8.21362\times10^{-4}$, versus measured g40 omission relative to kept yield
$7.19300\times10^{-4}$ (estimate about 14.2% above the measured gap).

The verifier checked raw report hashes and every fixture shell yield/count,
tail value and scene gap. Jobs 12672/12673 at `110c92aae` supply the production
sequences in `tests/data/reflection_selection_1173.json`. The new regression
pins G30 and selected-HKL digests for Si slab, MoS₂ film and Si substrate,
with 13,688 / 40,198 / 13,688 reflections and tail fractions
$3.679296\times10^{-4}$ / $6.230099\times10^{-4}$ / $9.557996\times10^{-4}$.
It replays measured scalar yields rather than rerunning GPU kernels or proving
physical accuracy independently.

| Filter | Result |
|---|---|
| Units, estimator limits, signs/conventions | Pass |
| Formula, guards, normalization, fixed-axis additivity | Match the independent derivation |
| Reference/provenance | Raw report hashes and measured fixture values match; source commit exists |
| Tolerance | Empirical finite-g40 qualification for these scenes, not a universal error certificate |
| Numerical accuracy | Inherits node/bin-mean production approximation and reduction route |
| Backend/axis coverage | Anchors use uniform ROCm bin-mean/Fourier; Si CPU cost is measured (job 12677) but layered CPU cost is not (time limit), and nonuniform ROCm remains outside this evidence; automatic-axis qualification is the Si GPU run only; the final rebase includes the upstream #1196 near/far repair |

Independent focused checks: cutoff and production-spectrum modules, 23 passed;
documentation static checks, 9 passed. The owner separately ran all 19 cutoff
tests, with each new measured anchor below 0.4 s. Human sign-off is not granted.


The verifier's final verdict is **rederived** after these wording corrections.
The owner promotes the narrowed algorithmic claim to **anchored** using the
local green reference-value regressions named in the ledger. New-anchor CI
has not been observed; this status does not imply CI success or human sign-off.

## Additional qualification at the approved default

Job 12678 at `80c03a680`, qlmc-ace ROCm, replays seed 642501 of the same
frozen transports on a 0.5–3 keV uniform bin-mean axis at 0.09375 eV
(26,667 coordinates). Both scenes pass every #1143 margin against g40:

| Scene | Selected cutoff / reflections | Yield gap vs g40 | Peak gap vs g40 | Selector call |
|---|---|---|---|---|
| Si | G28 / 11,180 | −4.77e-4 | −1.15e-4 | 13.0 s |
| MoS₂ / Si | G28 / 32,832; G30 / 13,688 | −4.94e-4 | −3.75e-4 | 113.4 s |

The narrower band reduces the Si slab and film sets; the layered substrate
keeps G30. These are ordered first-call latencies, not a controlled cold/warm
speedup comparison. Calls are bounded at 512 reflections. Reports, source
scripts, sync/replay hashes and allocation/device metadata are archived at
`qlmc:~/pyrite-1173-flip/evidence/flip/12678-narrow/`.

Job 12679 measures the Si 300 keV automatic axis with production node
quadrature on the same frozen transport. It resolves 420,113 coordinates over
50–300,000 eV at 0.713976 eV spacing in 0.030 s. Incremental selection takes
194.0 s, keeps G30 / 13,688 reflections and estimates a 7.06e-4 tail. This
measures the automatic-axis budget and cost, not its independent spectral
accuracy. Evidence: `qlmc:~/pyrite-1173-flip/evidence/flip/12679-auto/`.
CPU cost is not established by these GPU measurements; see job 12677 below.

Job 12677 (qlmc-queen CPU, 2 threads, 16 GiB, 90 min cap, seed 642501) replays
both frozen transports on the 0.5–10 keV node axis at 0.09375 eV (101,335
nodes), ladder `4, sel1e-3, g30`:

| Scene | n=4 first call | `sel1e-3` first call | Selected set |
|---|---|---|---|
| Si 300 keV / 10 µm | 24.0 s | 2,184.5 s | G30 / 13,688; tail estimate 3.68e-4 |
| MoS₂ 2000 Å / Si 30 keV | 156.4 s | not finished (cancelled at the time limit) | none |

For Si the selection lands on the `g30` set, so the harness scores it once and
both arms share the 2,184.5 s figure (peak host RSS 0.95 GiB). The selected
spectrum equals the g30 spectrum to 1e-12 relative on every observable, so the
selection changes nothing but the set. The four-family arm misses the same
#1143 gate margins: yield −61.5 %, centroid −34 %, peak −9.0 %, FWHM
−13.6 %. For layered MoS₂/Si no selected-arm result exists on CPU: the job was
cancelled at the limit with its report holding only the four-family arm. This is
a bounded failure, not a pass or a measured cost; it says only that the CPU
selection costs more than the roughly 14 min it ran. Only first-call wall times
were taken, so these are not cold/warm speedup claims. CPU cost at the full
automatic axis (420,113 nodes) was not measured. Cost reduction is tracked in
#1200. Evidence: `qlmc:~/pyrite-1173-flip/evidence/flip/12677-cpu/`.


The same-axis g40 reference (job 12680, 452.8 s scoring) measures a
−6.4471e-4 yield gap for the automatic selection, within the 1e-3 yield
target. Peak and FWHM gaps are −2.43e-6 and −2.86e-6. The full-support
centroid gap is −1.2214e-3, exceeding the harness's 1e-3 centroid diagnostic
margin; this full-support observable is not qualified by the passing
0.5–10 keV comparisons. No claim that every full-support diagnostic passes
follows from the yield tolerance. The original declared-band qualification
remains unchanged. Reference and selection use the same 420,113-node axis
and frozen transport; evidence is archived at
`qlmc:~/pyrite-1173-flip/evidence/flip/12680-auto-reference/`.
These jobs used pre-rebase source `80c03a680`; the later upstream #1196
repair affects the nonuniform ROCm near/far route, which these node and
uniform Fourier measurements do not use.

## Net-yield metric (#1250 P5c)

Owner's statement for fresh-context re-verification; not yet independently
verified. `CXR_REFLECTION_SELECTION_MODEL` is `measured-shell-cutoff-v4`.

**Why the metric changed.** A crystalline BremsLib continuum now has the
lattice-coherent share $S$ of the scored reflections removed
(`brem-lattice-coherent-fraction`). The spectrum a run reports is therefore
the consistent sum $\sum_{k\le K}L_k+C-\sum_{k\le K}S_k$, where $L_k$ is shell
$k$'s in-band line yield, $S_k$ the in-band continuum share its reflections
remove and $C$ the uncorrected continuum. Raising the cutoff by one shell
changes that total by $s_k=L_k-S_k$, not by $L_k$. On the frozen #642
transports, about 82 % of the g16 → g30 line increment is CBS, which $S_k$
cancels to within 0.1–0.8 % (Si 300 keV) and 4–5 % (MoS₂/Si 30 keV, before
the line-cutoff floor). A tail of $L_k$ measures omitted line yield, most of
which the corrected continuum already restores. That forced G30.

**Claim.** For a radiator whose layer continuum is corrected:

- $s_k=L_k-S_k$, with $S_k$ the in-band yield of the production continuum
  scorer evaluated with tables holding only shell $k$'s share
  ($\text{base}-\text{corrected}$ for exactly those reflections, same
  line-cutoff floor), on the case's continuum electrons, segments, layer
  view and estimator (coupled or uncoupled).
- The scorer is linear in the table, so the $S_k$ add to the share of the
  union, up to DDCS nodes clamped at zero.
- $S_k$ and $C$ are integrated by the trapezoid rule on 96 log-spaced nodes
  per decade across the line band (the bin edges for `bin-mean`, first and
  last node for `node`); $L_k$ keeps the production line weights.
- Ratios and the decay count use $|s_k|$. The tail is
  $\tau_k=|s_k|\,r_k/[(1-r_k)(\sum_{j\le k}s_j+C_r)]$, with $C_r$ the
  radiator layer's own uncorrected in-band continuum; a zero $|s_k|$ or a
  non-positive denominator gives $\infty$. The stop rule (three shells, two
  consecutive decreases, $\tau_k\le\mathrm{tol}$) is unchanged.
- Per-radiator denominators partition the layered case: summed over
  radiators, the estimated omission is at most tol of
  $\sum_r(\text{net}_r+C_r)$, which is no larger than the case's consistent
  total (amorphous layers' continua are not assigned).
- Radiators without a corrected continuum keep the v3 line metric.

**Not claimed.** $|s_k|$ extrapolation bounds the omitted *net* magnitude
only under the same unobserved-ratio assumption as v3, and a zero crossing
of $s_k$ (where the share approaches the line yield at large $G$) can stop
the search early. $S_k$ uses the continuum histories and $L_k$ the line
histories; these overlap (identical when $N_e=N_e^{\rm brem}$) and the share
is floored at the line cutoff, so common fluctuations largely cancel. The
remaining noise on non-shared histories is amplified by $L_k/s_k$, so
low-history runs pick a noisier $G$. The coarse continuum grid integrates absorption-edge
steps with a trapezoid error far below tol.

**Local probes** (2 histories, 0.5–10 keV, 100 eV bin-mean, tol 1e-3,
`scratch/p5c/sel_probe.py` at `9140efe58`):

| Scene | Radiator | Selected | Reflections | $s_k/L_k$ at stop | Tail |
|---|---|---|---|---|---|
| Si 300 keV / 10 µm | slab | G20 | 4,016 | 0.21 | 6.8e-4 |
| MoS₂ 2000 Å / Si 30 keV | MoS₂ film | G22 | 15,930 | 0.15 | 3.4e-4 |
| | Si substrate | G22 | 5,408 | 0.16 | 6.6e-4 |

**Remote selection and total-spectrum margins** (64 histories, seeds
642501–642503, qlmc jobs 13552–13555 and 13556–13558). Remote runs select the
same arms as the local probes: Si G20 (estimated net tail 4.2–5.0e-4) and
MoS₂/Si G22 on both radiators. The G20 total spectrum, lines through
$|g|\le20$ plus the continuum corrected for that set, was compared with G30
under the #1143 margins, on the same transports and 100 eV bins (worst of
three seeds):

| Arm | Reflections | Yield | Centroid | Peak | Dominant FWHM | Mass L1 |
|---|---|---|---|---|---|---|
| G20 | 4,016 | 4.4e-4 | 1.8e-4 | 4.3e-4 | 7.7e-5 | 3.5e-4 |
| G22 | 5,408 | 1.8e-4 | 8.4e-5 | 2.6e-4 | 4.9e-5 | 1.7e-4 |
| margin | | 1e-3 | 1e-3 | 1e-2 | 1e-2 | 1e-2 |

G20 passes every margin. Its measured yield gap (4.4e-4) agrees with the
selector's own tail estimate, so the default tolerance stays at 1e-3. The
uncorrected continuum at G20 misses the yield margin (2.7e-3), which is the
double count the correction removes.

The share pass cost 4.6 s (Si) and 18 s (MoS₂/Si) against 11 s and 28 s of
line scoring at 2 histories. It is dominated by the per-shell table builds,
which do not grow with the history count.

**Tests.** `tests/montecarlo/runner/test_reflection_cutoff.py`
(`::test_net_tail_runs_on_magnitudes_against_the_consistent_total`,
`::test_net_metric_stops_once_the_net_tail_is_small`),
`tests/montecarlo/runner/test_lattice_share.py::test_share_only_tables_hold_what_the_correction_removes`,
`tests/montecarlo/runner/test_reflection_selection_spectrum.py::test_selected_spectrum_is_the_spectrum_of_the_selected_set`
(production record: `metric="net"`, background, $0<S_k<L_k$, selected
spectrum still equals the single-call spectrum of the selected set).

## Net-metric re-verification (2026-11-17)

Fresh-context verification of the v4 net metric (`measured-shell-cutoff-v4`).
Sections 1–5 were written from the ledger rows
`reflection-yield-tolerance-selector` and `brem-lattice-coherent-fraction`
and the owner's statement above. They were written before the bodies of
`reflection_cutoff.py`, `lattice_share.py` and `brem_lattice_share.py` were
read. The code comparison follows.

### 1. What the cutoff changes

Fix one radiator $r$, its transport, observer direction, absorption stack and
line band $[E_a,E_b]$. Write $\Lambda_g(E)$ for the incoherent line density of
reflection $g$ and $b(E)$ for the uncorrected continuum density of the layer.
Write $\sigma_g(E)$ for the continuum density that reflection $g$'s
lattice-coherent share contributes. The correction removes that share for
exactly the scored set $\mathcal G$. For a cutoff $G$, with
$\mathcal G(G)=\{g:\lvert g\rvert\le G\}$, the reported in-band yield is

$$
Y(G)=\int_{E_a}^{E_b}\Bigl[\sum_{g\in\mathcal G(G)}\Lambda_g
+b-\Sigma_{\mathcal G(G)}\Bigr]\,\mathrm dE ,
$$

where $\Sigma_{\mathcal G}$ is the continuum density the corrected table
removes. If the correction is linear in the reflection set,
$\Sigma_{\mathcal G}=\sum_{g\in\mathcal G}\sigma_g$ (section 3), then with
shells $\mathcal G_k$,

$$
Y(G_K)=\sum_{k\le K}\underbrace{\bigl(L_k-S_k\bigr)}_{s_k}+C_r,\qquad
L_k=\int\sum_{g\in\mathcal G_k}\Lambda_g,\quad
S_k=\int\sum_{g\in\mathcal G_k}\sigma_g,\quad C_r=\int b .
$$

So $s_k$ is additive over shells because both $L_k$ and $S_k$ are, and
$Y(G_{K+1})-Y(G_K)=s_{K+1}$ exactly. The omitted part of the converged answer
is $Y(\infty)-Y(G_K)=\sum_{k>K}s_k$. The consistent relative omission is
therefore

$$
\frac{\sum_{k>K}s_k}{\sum_{k\le K}s_k+C_r}.
$$

The denominator is the line plus corrected continuum yield of the kept
spectrum in the band. Using $L_k$ instead, as v3 did, measures omitted *line*
yield. Most of that yield is CBS that the uncorrected part of the continuum
already contains, so v3 over-states the omission whenever the continuum is
corrected. $s_k$ is the correct stop quantity for a corrected radiator. For
an uncorrected radiator $\sigma_g\equiv0$, so the reported spectrum is
$\sum L_k+C_r$ and the increment is $L_k$. The v3 line metric with the
line-only denominator remains the stated v3 quantity; including $C_r$ there
would be a different, looser choice and is not claimed.

**Units and normalization.** $L_k$, $S_k$ and $C_r$ must be integrals of
densities in the same convention: photons per incident electron per sr per eV,
integrated over the same band in eV, at the same observer direction, and after
the same self-absorption. They must also use the same per-incident-electron
normalization, so a per-history average over two different populations is
legitimate only when both divide by their own number of primaries. The
quadratures may differ, for example production line weights for $L_k$ and a
log-node trapezoid for $S_k$ and $C_r$. Both still approximate the same
integral, and the difference is quadrature error. The trapezoid error on an
absorption-edge step is bounded by the step height times the node spacing:
at 96 nodes per decade, $\Delta E/E=10^{1/96}-1=2.4\%$. The edge error is
therefore at most about $\tfrac12\times2.4\%$ of the local continuum density
times one interval, a few $10^{-4}$ of $C_r$ per edge in a 0.5–10 keV band.
This error enters $C_r$ and the denominator, not the shell ratios, so it moves
$\tau$ by a relative amount of that size.

### 2. Geometric tail on a signed sequence

For a corrected radiator, $s_k\approx\mathrm{PXR}_k+I_k+\delta_k$, where
$I_k$ is the PXR×CBS interference and $\delta_k$ is the model mismatch between
the line route's CBS and the share. The ledger quotes $\delta$ as 0.1–0.8 % of
CBS for Si 300 keV and 4–5 % for MoS₂/Si before the floor. Each term can have
either sign in principle, although $\mathrm{PXR}_k\ge0$. If
$\lvert s_{k+j}\rvert\le\lvert s_k\rvert r_k^{\,j}$ for every $j\ge1$, then

$$
\Bigl\lvert\sum_{j\ge1}s_{k+j}\Bigr\rvert\le\sum_{j\ge1}\lvert s_{k+j}\rvert
\le\frac{\lvert s_k\rvert r_k}{1-r_k}.
$$

The magnitude extrapolation therefore bounds the signed remainder under the
same sufficient condition as v3. Sign changes in the tail only make it more
conservative, by the triangle inequality. The new failure mode is a
**zero crossing in the observed shells**. If $s_k$ passes through zero at shell
$k$ while the later $\lvert s_{k+j}\rvert$ grow again with the opposite sign,
the sequence $\lvert s_{k-2}\rvert>\lvert s_{k-1}\rvert>\lvert s_k\rvert\approx0$
satisfies the two-decrease guard, and $\tau_k\approx0$ stops the search.
The max-of-two ratio does not prevent this, because $\tau_k$ is proportional
to $\lvert s_k\rvert$. All terms carry the same Debye–Waller envelope
$e^{-2W}$, so a crossing needs $\mathrm{PXR}_k+I_k\approx-\delta_k$, an
accidental cancellation rather than a trend. It cannot be excluded in
general. It is plausible when the net fraction $s_k/L_k$ is comparable to the
relative mismatch $\delta_k/L_k$, which is not the case in the owner's probes
($s_k/L_k=0.15$–$0.21$ against $\delta/L\lesssim0.05$). This must be stated
as a limit of the claim.

A non-positive denominator $\sum_{j\le k}s_j+C_r\le0$ would mean the kept
spectrum has non-positive in-band yield, so no relative tolerance is defined.
Returning $\infty$ there, and for $\lvert s_k\rvert=0$, is the correct refusal.

### 3. Additivity of share-only tables

Let the continuum scorer map a DDCS table $\chi$ to an in-band yield
$\mathcal S[\chi]$. If $\mathcal S$ is linear in $\chi$, meaning that the
per-node values enter only through interpolation weights that do not depend on
$\chi$ and through a sum over segments, then

$$
\mathcal S[\chi_{\rm base}-\chi_{\rm corr}(\mathcal G)]
=\sum_k\mathcal S[\chi_{\rm base}-\chi_{\rm corr}(\mathcal G_k)]
$$

holds whenever $\chi_{\rm base}-\chi_{\rm corr}(\mathcal G)=\sum_k
(\chi_{\rm base}-\chi_{\rm corr}(\mathcal G_k))$ node by node. The
correction removes $kS_{\mathcal G}/(nZ^2)$ per node with
$S_{\mathcal G}=\sum_{g\in\mathcal G}\lvert U_g\rvert^2H$, which is linear in
the set. The identity fails only at nodes clamped at zero, where
$\chi_{\rm corr}=\max(\chi_{\rm base}-\Delta_{\mathcal G},0)$ makes the removed
amount $\min(\Delta_{\mathcal G},\chi_{\rm base})$, which is sub-additive. There,
$\sum_kS_k\ge S_{\mathcal G}$, so the net sum slightly under-states the kept
total. Linearity also requires conditions that must be checked in the code:

- interpolation linear in the table value (log-interpolation in $\chi$ would
  break it);
- the photon-energy sampling and the transport do not depend on the
  correction (transport keeps the full tables per the sibling row);
- the share-only pass uses the same line-cutoff floor and node set as the
  production correction.

### 4. Different electron populations

$L_k$ is scored on the line histories and $S_k$, $C_r$ on the continuum
histories. (The code comparison below finds that the two sets share histories
$[0,\min(N_e,N_e^{\rm brem}))$, which reduces the effect estimated here.)
Each is an unbiased per-primary estimator, so
$\mathbb E[s_k]=\mathbb E[L_k]-\mathbb E[S_k]$. The stop decision is not
linear in $s_k$, however. Within a population, all shells share the same
histories, so the dominant fluctuation is a common, nearly shell-independent
scale: $L_k\to L_k(1+\epsilon_L)$ and $S_k\to S_k(1+\epsilon_S)$. Then

$$
s_k\to s_k+\epsilon_LL_k-\epsilon_SS_k,\qquad
\frac{\delta s_k}{s_k}\approx\frac{\epsilon_L-(1-f)\epsilon_S}{f},\quad
f=s_k/L_k .
$$

With $f\approx0.2$ the relative noise of $\epsilon$ is amplified by about
$5\times$. If $L_k/S_k$ is nearly constant across neighbouring shells, the
error is again roughly proportional to $s_k$, so the ratios $r_k$ are
preserved. The relative change of $\tau_k$ is then about
$\delta s_k/s_k$ in the numerator, diluted by $C_r$ in the denominator. That
is a multiplicative error on $\tau$, not an offset on tol. Averaged over
fluctuations, $\mathbb E\lvert s_k\rvert\ge\lvert\mathbb E s_k\rvert$ biases
the magnitude, and so $\tau$, upward when $s_k$ is near its noise floor.
That is a conservative bias (a later stop). Shell-to-shell decorrelated noise
(different histories populating different shells) can instead fake a
decrease and stop early. This is the v3 risk, magnified by $1/f$. Low-history
runs (the owner's 2-history probes) therefore select a noisy $G$. This is a
statistical, not a systematic, defect, and it vanishes as the history count
grows.

### 5. Limits

- No reflections, or no corrected continuum: every $S_k=0$ and $C_r$ plays no
  role, so the metric must reduce exactly to v3 ($s_k=L_k$, denominator
  $\sum L_j$).
- $B\to\infty$ or $G\to0$: the share vanishes, the same reduction.
- Multiple radiators: each $\tau^{(r)}\le\mathrm{tol}$ against
  $\sum s^{(r)}+C_r$, so the summed estimated omission is at most
  $\mathrm{tol}\sum_r(\text{net}_r+C_r)$. That is at most tol of the case
  total if every per-radiator denominator is positive, because the remaining
  amorphous continua are non-negative.

### 6. Comparison with the implementation (head `9c72bd04c`)

| Derived requirement | Implementation | Result |
|---|---|---|
| Increment $s_k=L_k-S_k$; decay and ratios on $\lvert s_k\rvert$ | `select_reflection_cutoff`: `metric = value - share`; `decaying` compares `abs(metric)` | Match |
| $\tau_k=\lvert s_k\rvert r_k/[(1-r_k)(\sum_{j\le k}s_j+C_r)]$, $r_k$ max of two | `geometric_tail_fraction(..., background)` | Match |
| Refuse zero $\lvert s_k\rvert$ and a non-positive total | `np.any(size == 0.0) or not total > 0.0` gives `inf` | Match (any zero shell, not only the last: stricter, consistent with v3) |
| $S_k$ from tables holding $\chi_{\rm base}-\chi_{\rm corr}(\mathcal G_k)$ with the production floor | `share_only_tables` calls the production `_corrected_tables` with `_line_cutoff_keV(case)` and subtracts DDCS and SDCS | Match |
| Scorer linear in the table | `_evaluate`: linear in $\theta$, $k/T$ and $\ln T$ fractions, with weights independent of table values. `lattice_share_density`: `einsum` over $\lvert U_g\rvert^2$ weights. SDCS: $\chi^{\rm corr}_{\rm SDCS}=\chi^{\rm base}_{\rm SDCS}\cdot\mathrm{new}/\mathrm{old}$, a per-node constant times the corrected angular integral | Linear except at clamps |
| Clamp exception | `ddcs = np.maximum(ddcs, 0.0)`, so the share-only node is $\min(\chi_{\rm base},\Delta)$, which is sub-additive: $\sum_kS_k\ge S_{\mathcal G}$ | Match; the direction is a small under-statement of the kept total |
| Same estimator, layer view, composition, normalization | `_layer_brem_from_segments` uses `mc_brem_spectrum` (uncoupled; same `cross_section_model`, `bremslib_tables`, `groove`, `E_cut_brem_keV`, `electron_limit=Ne_brem`) or `mc_coupled_brem_spectrum` (coupled), on the same layer composition as `_uncoupled_brem_from_segments`/`_coupled_brem_from_segments` | Match |
| $C_r$ is the uncorrected in-band continuum of the layer | `background` scores the base tables with `tabled_only=False`, so untabled elements keep their EEDL fallback as in production | Match |
| Share omits untabled elements | `tabled_only=True` drops them; `share_only_tables` passes them through unchanged | Match |
| Band equals the line band | `axis_band_eV`: `bin-mean` outer edges, `node` end nodes; 96 log nodes per decade, trapezoid | Match |
| Correction set equals the selected set | `continuum_case` → `scored_reflections` rebuilds the shells to the cutoff and refuses a count mismatch | Match |
| Uncorrected radiator → v3 | `SelectionContinuum.applies` is `_uncorrected_reason(...) is None`; otherwise `select_reflection_cutoff` is called without `share_yield` (`background=None`, line form) | Match |
| Units | $L_k$: line density at the production weights; $S_k$, $C_r$: continuum density at trapezoid weights; both are photons per primary per sr per eV integrated over eV, at the same `n_hat` and absorption stack (`abs_layers`) | Match |

**Electron populations.** `electron_cutoffs` shows that the line histories are
$[0,N_e)$ and the continuum histories $[0,N_e^{\rm brem})$. They share
$[0,\min)$ and are identical when $N_e=N_e^{\rm brem}$. The continuum scorer
runs continuum histories down to `E_cut_brem_keV`, but the share is zero below
`E_cut_lines_keV` (the floor), which is exactly where the line route stops. So
$L_k$ and $S_k$ are estimated mostly on the same histories over the same
electron-energy range, and the common-scale fluctuations of section 4 largely
cancel. The owner's "Not claimed" wording ("different electron populations")
is conservative. They are distinct only when $N_e\ne N_e^{\rm brem}$, and then
only for the surplus histories. No systematic bias of the stop follows. The
residual statistical effect is the $1/f$ amplification of section 4 on
non-shared histories plus any history-level CBS-versus-share mismatch.

**Zero-crossing probe** (independent synthetic sequence, the production
`select_reflection_cutoff`). Take $\mathrm{CBS}_k=100\cdot0.8^k$,
$\mathrm{PXR}_k=60\cdot0.7^k$, $L_k=\mathrm{CBS}_k+\mathrm{PXR}_k$,
$S_k=1.3\,\mathrm{CBS}_k$ and $C_r=500$. The net yield falls through zero at
$k=5$ (0.254, then −0.805). The selector stops at $K=5$ with estimate
$3.2\times10^{-4}$, but the true omission is $2.9\times10^{-2}$ of the kept
total, 29× tol. The 30 % mismatch is exaggerated for illustration, but the
mechanism is generic. PXR and interference fall with $f_0(s)$ faster than CBS
falls with $(Z-f_0)$, so $s_k/L_k\to1-S/\mathrm{CBS}$ at large $G$. The ledger
gives $S/\mathrm{CBS}_{\rm MC}=0.999$–$1.008$ (Si 300 keV) and $1.04$–$1.05$
(MoS₂/Si 30 keV), so the net sequence can cross zero once $s_k/L_k$ reaches
the mismatch level. The owner's probes stop at $s_k/L_k=0.15$–$0.21$, well
before such a crossing, so these selections are unaffected. Tighter
tolerances or larger $G$ approach the crossing. This is a scope limit, not an
implementation fault. The existing "not a bound" disclaimer formally covers
it, because a crossing violates the future-ratio condition, but it is not named.

**Wording and notation findings (no code fault).**

1. The ledger row's net bullet calls the background $B$, which collides with
   the Debye–Waller $B$ used in the same row. The derivation page uses $C_r$;
   use $C_r$ in the ledger.
2. Name the zero-crossing limit in the ledger "Not a bound" bullet and in the
   page's "Not claimed". Asymptotically $s_k/L_k\to1-S/\mathrm{CBS}$, so a
   crossing of $s_k$ through zero can stop the search early. The magnitude tail
   is trustworthy only while $\lvert s_k\rvert/L_k$ is well above the
   share/CBS mismatch.
3. "Different electron populations" should read: the line and continuum
   histories overlap, $[0,\min(N_e,N_e^{\rm brem}))$, and the share is
   floored at the line cutoff.
4. `SELECTION_NODES_PER_DECADE`'s comment gives "~2.4 %" as the trapezoid
   error. 2.4 % is the node spacing $\Delta E/E$. The edge-step error is of
   order half a step times one interval, a few $10^{-4}$ of $C_r$ per edge.
   The conclusion (far below tol) is unchanged.

### 7. Tests run

`uv run pyrite-dev test tests/montecarlo/runner/test_reflection_cutoff.py
tests/montecarlo/runner/test_reflection_selection_spectrum.py
tests/montecarlo/runner/test_lattice_share.py` under an 8 GiB memory cap:
34 passed (16.7 s). Independent synthetic probe above:
`select_reflection_cutoff` with a sign-changing net sequence.

### 8. Verdict

| Filter | Result |
|---|---|
| Units / normalization | Pass: same per-primary, per-sr, in-band convention and observer |
| Limits | Pass: uncorrected radiator → v3 line form; $B\to\infty$, $G\to0$, empty set → zero share |
| Signs / conventions | Pass: $s_k=L_k-S_k$; magnitude extrapolation; refusal of a non-positive total |
| Additivity | Exact for share-only tables except clamped DDCS nodes (sub-additive, $\sum S_k\ge S_{\mathcal G}$) |
| Scope limit | Zero crossing of $s_k$ can stop early (shown); unnamed in the claim |

Re-derivation **matches** the implementation term for term. Verdict
**`rederived`** for the v4 net metric, conditional on wording edits 1–3 above
being applied to the ledger row (no code change required). Human sign-off is
not granted.

## Electron-prefix re-verification (v6, #1281, 2026-11-18)

Fresh-context review of the v6 electron-prefix change only; v3–v5 are taken
as given. Sections 1–3 were written from the ledger row and the v6 owner's
statement before the bodies of `reflection_cutoff.py`, `electron_blocks.py`
and `lattice_share.py` were read.

### 1. Normalization the stop can and cannot absorb

Write the per-primary prefix estimators

$$
\hat L_k=\frac{1}{n_L}\sum_{i<n_L}\ell_{ki},\qquad
\hat S_k=\frac{1}{n_B}\sum_{i<n_B}\sigma_{ki},\qquad
\hat C_r=\frac{1}{n_B}\sum_{i<n_B}c_{i},
$$

with $n_L=\min(N_e,n_{\max})$ and $n_B=\min(N_e^{\rm brem},n_{\max})$.
The line metric uses only ratios $\hat L_k/\hat L_{k-1}$ and
$\hat L_k/\sum_j\hat L_j$, so any common factor cancels; the normalization
of $\hat L$ is irrelevant there. The net metric is not invariant to separate
factors: if $\hat L\to a\hat L$ and $(\hat S,\hat C)\to b(\hat S,\hat C)$,
then $s_k\to a L_k-bS_k$, which changes both the ratios and the zero
crossing unless $a=b$. Consistency therefore requires each population to be
divided by its *own* prefix count ($n_L$ for lines, $n_B$ for continuum),
exactly as production divides lines by $N_e$ and continuum by
$N_e^{\rm brem}$. Dividing both by one count, or summing $n_B$ continuum
histories but dividing by $n_L$, biases $S_k$ by $n_L/n_B$ and is a
discrepancy. With the correct per-primary counts the two prefixes need not
be equal: both estimate the same per-primary expectations.

### 2. N-independence of the prefix

The selected set is a deterministic function of the prefix data. It is
identical for all $N\ge n_{\max}$ at fixed seed if and only if:

1. history $i$'s trajectory (primary and every secondary it produces) is a
   function of $(\text{seed},i)$ only — counter-addressed streams;
2. its transport settings do not depend on $N$. In particular, if electrons
   with $i\ge N_e^{\rm brem}$ are transported with a different cutoff
   (`E_cut_lines`), then $N_e^{\rm brem}$ must be $N$-independent, or at
   least $\ge n_{\max}$, for the prefix histories to keep the same settings;
3. $n_B$ itself is $N$-independent: $N_e^{\rm brem}$ fixed or
   $\ge n_{\max}$. If $N_e^{\rm brem}$ is a function of $N$ below
   $n_{\max}$, the continuum prefix grows with $N$ and (a) fails;
4. the prefix is selected by an electron-id mask on segment rows (or by an
   electron-sorted offset), so row order cannot change membership;
5. secondaries carry the primary's id (then they belong with their
   primary's history), or are excluded consistently in both the selection
   and the normalization.

### 3. Limits and consequences expected

- $N_e\le n_{\max}$ and $N_e^{\rm brem}\le n_{\max}$: no restriction, so v5
  exactly. If only one population exceeds $n_{\max}$, only that population is
  restricted; v5 is reproduced only when both are within budget.
- Conditional on the selected set, the production pass is the v5 estimator
  scored on all histories: unbiased with the same normalization.
- The cutoff is a random variable of $n_{\max}$ histories; its distribution
  does not narrow with $N$. That is a difference in which set is chosen, not
  a bias of the spectrum of a given set.

### 4. Comparison with the implementation (head `eab6be482`)

- **Prefix construction.** `electron_blocks.py::electron_prefix` keeps the rows
  with `elec_id < n` (a leading slice when rows are electron-sorted, an index
  mask otherwise) and sets `Ne = n`. Secondaries carry their primary's
  `electron_id` (`transport/secondaries.py`, `tracks["electron_id"]`), so
  whole histories are kept. Condition 2.4 and 2.5 hold.
- **Counts.** `reflection_cutoff.py::_batch_selection` (refactored from
  `selected_batch_spectrum`) takes $N_e$ from `batch.electron_limit` (the
  production batch always sets it to `Ne_lines`) and $N_e^{\rm brem}$ from the
  spectrum phase, falling back to $N_e$ when `Ne_brem` is `None`. It forms
  $n_L=\min(N_e,n_{\max})$, $n_B=\min(N_e^{\rm brem},n_{\max})$, slices the
  rows to $\max(n_L,n_B)$, and passes `electron_limit=n_L` to the line
  evaluation and `Ne=n_B` to `lattice_share.py::selection_continuum`.
- **Normalization.** The line kernel masks `elec_id < Ne` and divides by
  `Ne = electron_limit` (`cxr/_setup.py`). `mc_brem_spectrum` masks
  `elec_id < Ne` and divides by `Ne = electron_limit` (`spectrum/brem.py`).
  Each population is therefore divided by its own prefix count, as §1
  requires. This holds for $n_B<n_L$, $n_B>n_L$ and layered radiators, because
  the layer view of `_segments_in_layer` keeps the scalar `Ne` and the mask.
- **Limit.** The prefix is skipped iff $(n_L,n_B)=(N_e,N_e^{\rm brem})$. The
  unchanged batch and `Ne_brem` then reach exactly the v5 call. When only one
  population exceeds $n_{\max}$, only that one is restricted, and the result is
  not v5.
- **Production pass.** With a prefix, `separate` is true even without a
  reference band. `_selected_spectrum` then rescores the pruned set with
  `evaluate`, which closes over the unrestricted `batch`. This is the v5
  estimator conditional on the selected set.
- **Row fields.** `_SEG_ARRAYS` does not list `hard_radiative_k_eV`, `_Z`,
  `_direction` and `_target_momentum_eV_c`. The prefix leaves those arrays at
  full length. The current consumers (line kernel, expected-value coupled
  continuum) do not read them, and a probe matched a fully sliced prefix
  exactly. They remain a latent misalignment for any future prefix consumer of
  hard-event rows. `iter_electron_blocks` and `_segments_in_layer` have the same
  latent misalignment.

**N-independence (condition 2.1), the first divergent convention.** The
derivation's step "because streams are counter-addressed per electron, the
prefix is the same set of histories for every $N\ge n$" holds only on the
`per-electron` and `cuda` transport cores. `resolve_transport_core("auto")`
returns `lockstep` on any host without CUDA and for every grooved run. That
core draws from "one shared Generator in step-major order"
(`transport/api.py`), so history $i$'s trajectory depends on $N$. Two further
points:

- With a beam energy spread, `block_transport.py` builds the shell, radiative
  and LUT tables over `table_energy_range`, which spans
  $[\min E_{\rm cut},\max_{i<N}E_{0,i}]$. That range grows with $N$, so even
  the counter-addressed cores agree only to the level of table-interpolation
  round-off, not bit for bit.
- $n_B$ is fixed only when $N_e^{\rm brem}$ is held fixed, or is at least
  $n_{\max}$. `cxr_histories` and `brem_histories` are independent settings,
  and adaptive runs realize $N_e^{\rm brem}$ separately.

### 5. Probes (CPU, Si 30 keV fixture of the anchor test, `scratchpad` scripts)

| Probe | Core | Result |
|---|---|---|
| A: $N_e=6$, $N_e^{\rm brem}=3$, budget 4 ($n_L=4\ne n_B=3$) vs an unrestricted selection on the population (4, 3) | lockstep | identical cutoff; shell $L_k$, $S_k$ and $C_r$ equal to 0 relative |
| B: transport $N_e=6$ vs $N_e=10$ (fixed $N_e^{\rm brem}=3$), budget 4 | lockstep | same cutoff, but shell yields differ up to 43 % and $C_r$ 5.45e-5 vs 4.00e-5; electron 0 has 252 vs 227 rows |
| B on `PYRITE_MC_TRANSPORT_CORE=per-electron` | per-electron | shells and $C_r$ bit-identical |
| C: $N_e^{\rm brem}=2$ vs 3 at budget 4 | per-electron | $n_B$ follows $N_e^{\rm brem}$; $C_r$ changes (3.91e-5 vs 3.97e-5) |
| Unsorted rows, coupled radiative continuum | lockstep | prefix and fully sliced background agree |

Tests: `uv run pyrite-dev test
tests/montecarlo/runner/test_reflection_selection_spectrum.py
tests/montecarlo/runner/test_reflection_cutoff.py` under an 8 GiB cap: 33
passed, 2 slow tests skipped.

**Anchor coverage.**
`test_selection_scores_an_electron_prefix_then_every_history` pins the
mask, the normalization and the production rescoring on *one* transport. Its
"alone" reference reuses `electron_prefix(tp["segs"], 2)`, so it does not
test N-independence of transport. With $n_L=n_B=2$, it also does not
exercise unequal prefixes. Probe A covers unequal prefixes and probe B
covers N-independence; the owner may promote either to the test module.

### 6. Verdict (v6)

| Filter | Result |
|---|---|
| Units / normalization | Pass: lines on $n_L$ divided by $n_L$; continuum on $n_B$ divided by $n_B$; consistent per-primary $s_k=L_k-S_k$ and $\sum s_j+C_r$ |
| Limits | Pass: both populations within budget gives exactly v5. Only one within budget restricts only the other (the claim needs this wording) |
| N-independence | Holds on counter-addressed cores (per-electron, CUDA) with a monoenergetic beam and fixed $\min(N_e^{\rm brem},n_{\max})$. **Fails** on lockstep (CPU default, all grooved runs). Only to round-off with an energy spread |
| Line vs continuum prefixes | Pass, including $n_B<n_L$, $n_B>n_L$, `Ne_brem=None` and layered radiators |
| Scope limits | The noise floor of $n_{\max}$ histories is documented. On lockstep, the cutoff also varies with $N$ through the realized prefix, still with $n_{\max}$-history noise |

The estimator re-derivation **matches**. Consequences (b), (c) and (d) hold.
Consequence (a) diverges at the stated mechanism: "counter-addressed streams".
That is a property of the per-electron and CUDA cores, not of lockstep.
Overall verdict: **`rederived`** for v6, conditional on these wording edits
(no code change required):

1. The derivation and ledger should state that the selected set is identical
   for every $N\ge n_{\max}$ at fixed seed only on counter-addressed transport
   cores (per-electron, CUDA), for a fixed $\min(N_e^{\rm brem},n_{\max})$.
   They should add that an energy spread rebuilds the transport tables over
   an $N$-dependent range, so agreement there is only to round-off. On
   lockstep (CPU default, grooved runs) the prefix histories depend on $N$,
   and the cutoff remains an $n_{\max}$-history estimate of a different
   realization.
2. Replace "$N\le n$ reproduces v5 exactly" with: "v5 exactly when both
   $N_e\le n_{\max}$ and $N_e^{\rm brem}\le n_{\max}$. Otherwise only the
   over-budget population is restricted."
3. The regression sentence should say the anchor pins the mask, the
   normalization and the full-history rescoring on one transport. It does not
   pin cross-$N$ invariance or unequal line/continuum prefixes.

Human sign-off is not granted.

## Exact band-mass decision yield and single pass (v8, #1326)

This section is the implementer's derivation and regression record. It has not
had fresh-context verification, so it does not change the row's `rederived`
status.

**Claim.** A shell's decision yield may be taken as the closed-form in-band
mass of its line arrays,

$$
Y_{\rm band}=\sum_{\ell}w_\ell\,\frac{\pi}{a_\ell}\Bigl[R(x^{\rm hi}_\ell)-R(x^{\rm lo}_\ell)+\mathbf 1\{x^{\rm lo}_\ell<0\le x^{\rm hi}_\ell\}\Bigr],
\qquad x^{\rm lo,hi}_\ell=\frac{a_\ell(E_{\rm lo,hi}-E^{\rm res}_\ell)}{\pi},
$$

with $R=F-s/2$ the tail complement of the $\mathrm{sinc}^2$ antiderivative
(`sinc-bin-integration`). Bins that tile $[E_{\rm lo},E_{\rm hi}]$ telescope, so
$h\sum_i\bar\rho_i$ over the exact bin means equals $Y_{\rm band}$ to
rounding. The previous selector measured the same quantity through bin means
carrying the production far-envelope replacement, whose per-line difference is
at most $3/(2\pi^3K^2)=1.2\times10^{-5}$ of that line's mass
(`sinc-bin-far-envelope`). $Y_{\rm band}$ has no such term, so a marginal shell
whose yield sits within that allowance of a stop decision can flip.

**Single pass.** The selector keeps every shell it scores up to the stop, so
no scored shell is rejected. With a prefix that covers the population
($N\le n_{\max}$) and a production axis spanning the reference band, the line
keep window and tabulation of the production pass contain every line that
reaches the band. The production spectrum of each shell and $Y_{\rm band}$ come
from the same line arrays, the spectrum of the selected set is the sum of the
scored shells, and nothing is rescored. A production axis short of the band
cannot supply the band's lines, and a prefix smaller than the population scores
different electrons; both keep the reference pass, which now generates the lines on the
reference band and measures them with a `mass_only` audit, with no bin
reduction. The decision is on the reference band alone, as in v5.

**Normalization.** Audit sums are un-normalized; $Y_{\rm band}$ is divided by the
number of electrons the spectrum is divided by, so it is in the units of
$\rho\cdot w$ that the continuum shares and background already use.

**Regression anchors.**
`tests/montecarlo/spectrum/test_band_mass.py` pins the closed form against
summed `band_bin_grid` bin means (exact, $10^{-11}$) and against the
production far-envelope bins (within $3/(2\pi^3K^2)$ of the whole mass),
plus per-line limits and guards;
`tests/montecarlo/spectrum/test_sinc_bin_integration_cuda.py::test_band_mass_kernel_matches_the_host_closed_form`
pins the device reduction against the host;
`tests/montecarlo/runner/test_reflection_selection_spectrum.py::{test_axis_spanning_the_band_selects_in_one_pass,test_short_axis_measures_band_mass_without_reducing_lines}`
pin the selected set, shell yields ($10^{-9}$) and spectrum against the
bin-mean reference route, that no `_selected_spectrum` pass runs on the spanning
axis, and `tests/montecarlo/runner/test_reflection_cutoff.py::test_shell_mass_is_the_decision_yield_and_density_is_the_spectrum`.
