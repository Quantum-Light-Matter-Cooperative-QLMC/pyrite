# Pixel acquisition counting: electron count and native-bin partition

**Claim id:** `pixel-acquisition-counting`
**Code:** `src/pyrite/instrument/acquisition.py::electron_count`; `src/pyrite/instrument/acquisition.py::score_acquisition`
**Source:** SI definitions of the coulomb, picocoulomb and hertz; exact elementary charge $e = 1.602176634\times10^{-19}\,\mathrm{C}$ (2019 SI redefinition, `scipy.constants.elementary_charge`)
**Verifier:** fresh-context independent verification, issue #4 batch 2 (2026-11-11)

## Inputs recorded before reading the implementation

From the ledger row and the docstrings/signatures only:

| Symbol | Code identifier | Unit | Meaning |
| --- | --- | --- | --- |
| $t$ | `acquisition.exposure_s` | s | exposure time, positive |
| $f$ | `rep_rate_hz` | Hz $=\mathrm{s^{-1}}$ | bunch repetition rate, nonnegative |
| $Q$ | `bunch_charge_pc` | pC | charge per bunch, nonnegative |
| $m_k$ | native `NativeSpectrum.events` | events per incident electron | accepted measured event mass in native bin $k$ |
| $[a_k, b_k)$ | `NativeSpectrum.edges_eV` | eV | native half-open measured bins, strictly increasing, $a_0 \ge 0$ |
| $[e_j, e_{j+1})$ | `acquisition.measured_edges_eV` | eV | reporting bins, half-open |
| $E_{\rm th}$ | `acquisition.hit_threshold_eV` | eV | hit threshold, nonnegative |

The `electron_count` and `score_acquisition` docstrings carry only a one-line summary and the
`Validation:` marker; they state no source equation, assumption or limiting case (finding F1
below). The ledger states the conservative-rebin assumption: uniform event density within each
native measured bin. While reading the `electron_count` docstring the verifier's view also
included its five-line body; the derivation below does not depend on it.

## Independent derivation

### Electron count

One bunch carries charge $Q\,[\mathrm{pC}] \times 10^{-12}\,\mathrm{C\,pC^{-1}}$. Each electron
carries $e$ (exact). Electrons per bunch: $n_b = Q\cdot10^{-12}/e$. Over exposure $t$ at
repetition rate $f$ the number of bunches is $tf$ (dimensionless, $\mathrm{s}\cdot\mathrm{s^{-1}}$).
Hence

$$
\boxed{N_e = \frac{t\,f\,Q\cdot 10^{-12}}{e}},
\qquad
[N_e] = \frac{\mathrm{s}\cdot\mathrm{s^{-1}}\cdot\mathrm{C}}{\mathrm{C}} = 1 .
$$

The count is an expected (mean) number of incident electrons; $tf$ need not be an integer.

Limits:

- **One electron per bunch**: $Q = e/10^{-12}\,\mathrm{pC} = 1.602176634\times10^{-7}\,\mathrm{pC}$
  gives $N_e = tf$ exactly (up to one floating-point rounding).
- **Zero charge or zero rate**: $Q = 0$ or $f = 0$ gives $N_e = 0$ exactly.
- **Sign**: all factors nonnegative, so $N_e \ge 0$; negative $f$ or $Q$ must be rejected
  (electron charge sign is already absorbed by counting electrons, not charge).
- Reference values (exact rational arithmetic): $t=1\,\mathrm{s}$, $f=10\,\mathrm{Hz}$,
  $Q=1\,\mathrm{pC}$ gives $N_e = 62\,415\,090.744\,607\,63$; $t=2.5\,\mathrm{s}$,
  $f=1\,\mathrm{kHz}$, $Q=250\,\mathrm{pC}$ gives $N_e = 3.900\,943\,171\,537\,977\times10^{12}$.

### Expected counts

If $m_k$ is the accepted measured event mass per incident electron in native bin $k$
(events $\mathrm{e^{-1}}$), the expected number of events in that bin during the exposure is

$$
\mu_k = N_e\, m_k ,
$$

dimensionless (events). Any rebinned quantity is linear in $m_k$, so scaling by $N_e$ commutes
with the partition below; scaling can be applied before or after rebinning. In counting mode
each channel's realization is a Poisson draw with mean equal to its expected count; the mean
of the draw is the expected count, and Poisson additivity keeps disjoint channel sums
consistent in distribution.

### Conservative partition of native bins

Assume uniform density within native bin $[a_k, b_k)$: the mass in any measurable subset
$S\subset[a_k,b_k)$ is $m_k\,\lvert S\cap[a_k,b_k)\rvert/(b_k-a_k)$, where
$\lvert\cdot\rvert$ is interval length. Define the overlap fraction

$$
w_k(S) = \frac{\lvert [a_k,b_k)\cap S\rvert}{b_k-a_k} .
$$

Let reporting edges $e_0 < e_1 < \dots < e_M$ and threshold $E_{\rm th}\ge 0$. The threshold
acts first (a sub-threshold event is never a hit), so define the disjoint output channels as
subsets of $[0,\infty)$:

$$
\begin{aligned}
S_{\rm below} &= [0, E_{\rm th}),\\
S_{\rm under} &= [E_{\rm th}, e_0) \quad(\text{empty if } E_{\rm th}\ge e_0),\\
S_j &= [\max(e_j, E_{\rm th}),\, e_{j+1}) \quad(\text{empty if } E_{\rm th}\ge e_{j+1}),\ j=0,\dots,M-1,\\
S_{\rm over} &= [\max(e_M, E_{\rm th}),\, \infty).
\end{aligned}
$$

These sets are pairwise disjoint and their union is $[0,\infty)$: below threshold everything is
in $S_{\rm below}$; above threshold the half-open sets $[E_{\rm th},e_0)$, $[e_j,e_{j+1})$,
$[e_M,\infty)$, each intersected with $[E_{\rm th},\infty)$, tile $[E_{\rm th},\infty)$.
Since $[a_k,b_k)\subset[0,\infty)$ and length is additive over disjoint intervals,

$$
w_k(S_{\rm below}) + w_k(S_{\rm under}) + \sum_{j} w_k(S_j) + w_k(S_{\rm over}) = 1
$$

for every $k$, hence

$$
\boxed{C_{\rm below}+C_{\rm under}+\sum_j C_j + C_{\rm over} = N_e\sum_k m_k}
$$

with $C_X = N_e \sum_k m_k\,w_k(S_X)$. Every $w_k(S_X)\in[0,1]$, so no channel is negative.

Edge handling: a shared edge (native, reporting, or threshold) is a single point of measure
zero, so with interval-length weights no mass is double-counted or lost at coincident edges;
the half-open convention only matters for point masses, which the uniform-density model
excludes. When native and reporting edges coincide, $w_k(S_j)$ is the Kronecker delta and the
rebin is the identity. A native bin straddling $E_{\rm th}$ splits proportionally:
fraction $(E_{\rm th}-a_k)/(b_k-a_k)$ to below-cut and the remainder to the reporting or
under/overflow channels containing $[E_{\rm th}, b_k)$.

Worked hand example. Native edges $[0,10,20,30]$ eV with masses $m=(1,2,3)$; reporting
edges $[5,15,25]$ eV; $E_{\rm th}=8$ eV.

- Bin $[0,10)$, $m=1$: below $[0,8)$ $0.8$; under $[8,5)$ empty $0$; $S_0=[8,15)$ overlap
  $[8,10)$ $0.2$. Sum $1$.
- Bin $[10,20)$, $m=2$: $S_0$ overlap $[10,15)$ gives $1.0$; $S_1=[15,25)$ overlap $[15,20)$
  gives $1.0$. Sum $2$.
- Bin $[20,30)$, $m=3$: $S_1$ overlap $[20,25)$ gives $1.5$; over $[25,\infty)$ overlap
  $[25,30)$ gives $1.5$. Sum $3$.

Totals: below $0.8$, under $0$, reporting $(1.2, 2.5)$, over $1.5$; sum $6 = \sum m$.
With $E_{\rm th}=0$: below $0$, under $0.5$, reporting $(1.5, 2.5)$, over $1.5$.

## Filters and implementation comparison

Read after the derivation above was written.

### Electron count

```text
cadence = _nonnegative_real("rep_rate_hz", rep_rate_hz)
charge = _nonnegative_real("bunch_charge_pc", bunch_charge_pc)
if cadence == 0.0 or charge == 0.0:
    return 0.0
return acquisition.exposure_s * cadence * charge * 1.0e-12 / elementary_charge
```

Term-by-term identical to $N_e = tfQ\cdot10^{-12}/e$: same factors, exponent $-12$ for pC to C,
division (not multiplication) by $e$, and `scipy.constants.elementary_charge` equal to the exact
$1.602176634\times10^{-19}\,\mathrm{C}$. Negative and non-finite rate or charge raise
`ValueError`; zero rate or charge returns exactly `0.0`.

### Native-bin partition

`score_acquisition` calls `detector.native_score` (the detector response, a separate claim),
then `_partition_native`, then multiplies every channel by `electron_count`. The channel
intervals used by `_partition_native` are:

| Channel | Implementation interval | Derivation interval (threshold-first) |
| --- | --- | --- |
| reporting bin $j$ | $[\max(e_j,E_{\rm th}),\,e_{j+1})$ | $[\max(e_j,E_{\rm th}),\,e_{j+1})$ |
| underflow | $[0,\,e_0)$ | $[E_{\rm th},\,e_0)$ |
| below-cut | $[e_0,\,\min(E_{\rm th},e_M))$ | $[0,\,E_{\rm th})$ |
| overflow | $[e_M,\,\infty)$ | $[\max(e_M,E_{\rm th}),\,\infty)$ |

The implementation's sets are also pairwise disjoint and tile $[0,\infty)$: $[0,e_0)$,
then $[e_0,\min(E_{\rm th},e_M))$ plus the reporting sets that tile
$[\min(\max(E_{\rm th},e_0),e_M),\,e_M)$, then $[e_M,\infty)$. So the overlap fractions of
every native bin sum to one and the partition is conservative, with the same
interval-length weights $w_k(S)$ derived above.

The **reporting bins are identical** under both conventions. The two conventions differ only
in how out-of-window sub-threshold mass is labelled: the implementation is range-first
(underflow and overflow mean "outside the reporting window", regardless of threshold;
below-cut means "inside the window but below threshold"), whereas the derivation above was
threshold-first. Totals and the registered counts are unaffected. Neither the ledger row, the
function docstrings, nor the `Acquisition` docstring states which precedence applies; this is a
documentation finding (F2), not a factor discrepancy. The implementation's labelling is
self-consistent with `tests/instrument/test_acquisition_core.py`, which pins
underflow $=1.0$ and below-cut $=1.5$ for a threshold inside the reporting window.

Hand example under the implementation's range-first convention ($E_{\rm th}=8$ eV): underflow
$[0,5)$ takes $0.5$, below-cut $[5,8)$ takes $0.3$, reporting $(1.2, 2.5)$, overflow $1.5$;
sum $6$.

The `np.clip(..., 0.0, None)` on each scaled channel is inert because every overlap is
nonnegative by construction.

Counting mode draws independent Poisson variates with the expected channel values as means,
from a Philox stream keyed by observation digest, seed, pixel row/column and component.
Independence across channels and pixels is consistent with Poisson thinning of a single
Poisson process; the coordinate key gives selection/chunk invariance. `total_counts` sums
the registered bins rather than redrawing a total, and `combine_acquisitions` sums
component draws, both consistent with Poisson additivity.

### Numeric comparison

Scratch script with an independent loop reference (interval-length overlaps written from
the definitions above, no implementation helpers):

| Check | Result |
| --- | --- |
| $N_e(1\,\mathrm{s},10\,\mathrm{Hz},1\,\mathrm{pC})$ vs exact rational $62\,415\,090.744\,607\,63$ | relative difference $0$ |
| $N_e(2.5\,\mathrm{s},1\,\mathrm{kHz},250\,\mathrm{pC})$ vs exact rational $3.900\,943\,171\,537\,977\times10^{12}$ | relative difference $0$ |
| one electron per bunch, $t=3$ s, $f=7$ Hz | $N_e = 21.0$ |
| zero charge | $0.0$; all scored channels $0$ |
| negative / NaN rate | rejected |
| hand example, $E_{\rm th}\in\{0, 8\}$ eV | reporting $(1.5,2.5)$ and $(1.2,2.5)$ match; channel labels as tabulated |
| 2000 random native/reporting/threshold layouts | max $\lvert$impl $-$ range-first loop ref$\rvert = 8.9\times10^{-16}$; reporting bins match threshold-first ref equally; max relative conservation error $4.0\times10^{-16}$ |
| `score_acquisition` with `IdealPhotonCounter`, $N_e$ vs one-electron scaling | ratio equals $N_e$ to $2.2\times10^{-16}$; per-electron channel totals equal native event mass |

Anchors `tests/instrument/test_acquisition_core.py` and `tests/results/test_spatial_model.py`:
36 passed, 1 slow skipped.

## Findings

- **F1 (docstring contract).** `electron_count` and `score_acquisition` docstrings carry only a
  summary line and the `Validation:` marker; they omit the source equation, assumptions (uniform
  in-bin density) and a limiting case, which the contributor rules require.
- **F2 (convention undocumented).** Underflow/overflow take precedence over below-cut for
  sub-threshold mass outside the reporting window. Conservation and registered counts are
  unaffected, but the precedence should be stated in the ledger claim or the `Acquisition`
  docstring.
- Scope: the detector response `native_score` that produces the native event mass is not part of
  this claim and was not verified here.

## Verdict

`rederived`. The electron count and the conservative partition match the independent
derivation symbolically and numerically; the only divergence is the undocumented
channel-labelling convention F2, which does not change any registered count or total. No
hardware calibration claim is made or verified.
