# `coherent-line-grid-fringe-spacing`

> Ownership refresh (2026-10-20, #599): module paths below point to current
> owners. Dated implementation excerpts, line numbers and measurements retain
> their original revision and evidence scope; this does not revalidate physics.

Ledger row: [`coherent-line-grid-fringe-spacing`](../ledger-core-coherent-physics.md). Instrument: `montecarlo/spectrum/diagnostics.py::coherent_fringe_spacing` (estimator), `montecarlo/runner/cxr_grid.py::_refuse_coherent_resolution` (policy, non-windowed), `energy_grid/convergence_case.py` (ladder, now route-switchable). No kernel change: this row records the band limit of the existing coherent reducer and the evidence for refusing automatic resolution on that route (issue #117).

## Claim

1. **Band limit.** On the coherent route the line-grid Nyquist step is $h_\mathrm{coh} = \pi \hbar c / D$, where $D$ is the span of the per-segment retardation scalar $d_j = t_{\mathrm{mid},j} - \hat n \cdot \mathbf r_j$, not the per-segment sinc width of `sinc_feature_spacing`.
2. **Separation of principle.** Along one segment $d$ increments by exactly $(1 - \beta\, \hat v \cdot \hat n)\, t_L$, the quantity setting the incoherent first zero. The incoherent step therefore follows the *per-segment* increment and the coherent step the *total* span, so the incoherent estimator cannot bound the coherent route at any calibration.
3. **Measured.** At the spacing where the incoherent yield is converged to $3.1\times10^{-8}$, the coherent yield is still $5.0\times10^{-2}$ from convergence.
4. **Policy.** Automatic resolution without feature windows refuses `coherent_emission` rather than silently aliasing. A windowed policy resolves the fringes inside per-row coherent windows instead ([`coherent-line-grid-windowed-resolution`](coherent-line-grid-windowed-resolution.md), #350). An explicit `E_grid_line` is unaffected.

## Derivation

The coherent reducer (`montecarlo/spectrum/cxr/_per_hkl.py::_accumulate_reflection_coherent`, and the batched steps 5/7) builds one complex field per (reflection, orientation) row,

$$
A(E) = \sum_j c_j \,
\mathrm{sinc}\!\left(\frac{a_{\mathrm{width},j} (E - E_{r,j})}{\pi}\right)
e^{i \phi_j(E)},
\qquad
\phi_j(E) = d_j \omega(E) - \mathbf g \cdot \mathbf r_j
          - L_{\mathrm{esc},j}\, \Delta\omega(E),
$$

with $\omega(E) = E / \hbar c$ and $d_j = t_{\mathrm{mid},j} - \hat n \cdot \mathbf r_j$ in ångström at $c=1$ (`montecarlo/spectrum/cxr/_setup.py`). The estimator reads $d_j$ through `diagnostics.coherent_retardation`, in the reducer's own geometry: on an infinite slab with active decoherence each electron sits at its offset-free transverse position, as `montecarlo/spectrum/cxr/_setup.py` evaluates it; a finite footprint keeps the sampled positions. Every measurement below is on a finite footprint. The spectrum is $|A(E)|^2$, so it carries cross terms $j,k$ whose phase runs over the output grid at the **difference of per-segment phase slopes**

$$
s_j = \frac{\mathrm d \phi_j}{\mathrm d E}
    = \frac{d_j}{\hbar c}
      - L_{\mathrm{esc},j} \frac{\mathrm d (\Delta\omega)}{\mathrm d E}.
$$

The $-\mathbf g \cdot \mathbf r_j$ term is energy-independent and drops out. The in-medium term is smaller by $1 - n_\mathrm{re} \sim 10^{-5}$ for x-rays and is neglected. The fastest cross term therefore has period $2\pi / (s_\max - s_\min)$, giving the Nyquist step

$$
h_\mathrm{coh} = \frac{\pi}{s_\max - s_\min}
              \simeq \frac{\pi \hbar c}{D},
\qquad D = \max_j d_j - \min_j d_j .
$$

### Limiting case

For a single segment, $D \to 0$ and $h_\mathrm{coh} \to \infty$: the only remaining scale is the segment's own $\mathrm{sinc}^2$ envelope, and the coherent route collapses to the incoherent self-term $|A|^2 t_L^2 \mathrm{sinc}^2$ (the limit already pinned by `test_coherent_emission.py`). The band limit is then correctly the incoherent one.

### Why the incoherent estimator cannot be rescaled into this one

Along a segment of length $L$ at speed $\beta$, $t$ advances by $t_L = L/\beta$ and $\hat n \cdot \mathbf r$ by $L\, \hat v \cdot \hat n$, so

$$
\Delta d = t_L - L\,\hat v \cdot \hat n
         = (1 - \beta\, \hat v \cdot \hat n)\, t_L ,
$$

which is exactly the combination whose reciprocal sets the incoherent first zero $2\pi\hbar c / [(1 - \beta \hat v \cdot \hat n) t_L]$ in `sinc_feature_spacing`. The incoherent width is set by one increment; the coherent fringe by the sum of all of them. Their ratio grows with the number of coherently summed segments, so no fixed safety factor on the incoherent estimate bounds the coherent route.

## Decoherence selects which span applies

`decoherence_active` (`montecarlo/spectrum/cxr/_setup.py`) is true whenever the case carries nonzero bunch offsets or a finite footprint — including the standard `beam_fwhm_mm = 1.0` cases. The spectrum is then the blend $(1-F)\sum_e |S_e|^2 + F |\sum_e S_e|^2$ on the offset-free geometric phase:

- the $\sum_e |S_e|^2$ floor is band-limited by the widest **single-electron** span (`coherent_fringe_spacing(..., grouped=True)`);
- the $|\sum_e S_e|^2$ term is band-limited by the **all-electron** span.

The all-electron span is set by the transverse beam extent projected on $\hat n$, so it is ~2.5 mm in every case measured below, essentially independent of material and beam energy. Those inter-electron fringes are one realization of the sampled offsets — speckle, which `montecarlo/spectrum/cxr/_setup.py` already notes "does not shrink with electron count". Whether they are an observable at all is a modelling question outside this row; it is tracked separately as issue #155.

## Measurements

Transport: `build_ladder_case`, seed 7, tilt 5°, 1 mm, `bunch_length_fs = 100` (standard ladder cases carry none, and the coherent route refuses without it once decoherence is active).

Band limits at $N_e = 200$, against the incoherent estimate at $\varepsilon = 10^{-3}$:

| case | `sinc_feature_spacing` [eV] | $D$ all-electron [Å] | $h_\mathrm{coh}$ all-e [eV] | $h_\mathrm{coh}$ per-e [eV] | ratio all-e / per-e |
|---|---|---|---|---|---|
| hopg 30 keV | 0.974 | $2.53\times10^{7}$ | $2.45\times10^{-4}$ | $1.49\times10^{-2}$ | 3975× / 65× |
| hopg 100 keV | 0.481 | $2.65\times10^{7}$ | $2.34\times10^{-4}$ | $2.81\times10^{-3}$ | 2055× / 171× |
| wse2 30 keV | 7.85 | $2.50\times10^{7}$ | $2.48\times10^{-4}$ | $3.41\times10^{-2}$ | 31640× / 230× |

Refinement ladder, hopg 30 keV, $N_e = 40$, band 10–3700 eV, both routes on **identical trajectories** (one transport, injected into both ladders; `sinc_feature_spacing` estimate 1.504 eV):

| $h$ [eV] | points | incoherent yield $\Delta_\mathrm{rel}$ | coherent yield $\Delta_\mathrm{rel}$ | coherent centroid $\Delta$ [eV] |
|---|---|---|---|---|
| 4.0 | 923 | $2.49\times10^{-1}$ | $1.18\times10^{-1}$ | 0.36 |
| 2.0 | 1846 | $8.16\times10^{-2}$ | $5.32\times10^{-2}$ | 0.67 |
| 1.0 | 3691 | $3.08\times10^{-8}$ | $5.01\times10^{-2}$ | 1.38 |
| 0.5 | 7381 | $7.29\times10^{-9}$ | $4.72\times10^{-3}$ | 0.22 |

The incoherent route locks in at the sinc estimate, reproducing `line-grid-sinc-convergence`. The coherent route does not: it remains above the $10^{-3}$ intrinsic-source tolerance at three times finer spacing, and its centroid drift does not shrink monotonically, which is the signature of newly resolved fringes rather than converging quadrature.

## Policy consequence

Resolving the all-electron fringes uniformly over a keV band costs $\sim1.5\times10^{7}$ points against 2453 at the incoherent estimate; the grouped floor alone costs $\sim2.5\times10^{5}$. Neither is an affordable default, and silently spending either would misrepresent an unconverged spectrum as a converged one. `_refuse_coherent_resolution` therefore raises for `coherent_emission` on an automatic policy **without feature windows**, reporting both derived steps and their spans, and directs the caller to windows or an explicit `E_grid_line`. Since #350 a windowed policy spends the fine step only inside per-row coherent windows, under the envelope, step, and decoherence rules of [`coherent-line-grid-windowed-resolution`](coherent-line-grid-windowed-resolution.md), and refuses with its coordinate count only above the point budget. Explicit grids, stored catalog rows, and the whole incoherent route are unaffected.

## Reproduction

```python
from pyrite.energy_grid.convergence_case import CaseLadder, build_ladder_case
from pyrite.montecarlo.spectrum.diagnostics import coherent_fringe_spacing

case = dict(build_ladder_case("hopg", 30.0, 5.0, 0.0, n_electrons=40, seed=7))
case["bunch_length_fs"] = 100.0
incoherent = CaseLadder(case)
coherent = CaseLadder(case, transport=incoherent.transport, coherent=True)
coherent_fringe_spacing(incoherent.segments, incoherent.transport["n_hat"])
```

Heavier ladders (larger $N_e$, 100/300 keV, tilt 85°) belong on `pyrite remote`; the numbers above are the CPU-affordable subset and are sufficient to separate the two routes by six orders of magnitude at the policy spacing.

## Independent verification

Fresh-context verifier, 2026-11-11, issue #4 batch 3. The derivation below was
drafted from the ledger row and the signatures and docstrings of
`coherent_fringe_spacing`, `coherent_retardation` and
`_refuse_coherent_resolution` before this page or any implementation body was
read. The implementation was compared afterwards.

### Independent derivation

Units: $c=1$, times and lengths in Å, photon energy $E$ in eV,
$\omega = E/\hbar c$ in Å$^{-1}$, $\hbar c = 1973.27$ eV Å.

Take one straight segment $j$ at constant velocity $\beta\hat{\mathbf v}_j$,
duration $t_L$ (a time, so $t_L = L/\beta$ for path length $L$), midpoint time
$t_{\mathrm{mid},j}$ and midpoint position $\mathbf r_j$. Write
$t = t_{\mathrm{mid},j} + t'$ with $\lvert t'\rvert \le t_L/2$, so
$\mathbf r(t) = \mathbf r_j + \beta\hat{\mathbf v}_j t'$. The diffracted
far-field amplitude is

$$
A_j(E) = \int_{-t_L/2}^{t_L/2}
\exp\!\left\{ i\left[\omega\,\bigl(t - \hat{\mathbf n}\cdot\mathbf r(t)\bigr)
 - \mathbf g\cdot\mathbf r(t)\right]\right\} \mathrm dt'
= e^{i(\omega d_j - \mathbf g\cdot\mathbf r_j)}\,
  t_L\,\frac{\sin(\Delta_j t_L/2)}{\Delta_j t_L/2},
$$

with $d_j = t_{\mathrm{mid},j} - \hat{\mathbf n}\cdot\mathbf r_j$ and
$\Delta_j = \omega(1-\beta\hat{\mathbf v}_j\cdot\hat{\mathbf n}) - \beta\,\mathbf g\cdot\hat{\mathbf v}_j$.
The midpoint factorization leaves
a **real** envelope, so the whole energy-dependent phase of segment $j$ is

$$
\phi_j(E) = \frac{d_j}{\hbar c}E - \mathbf g\cdot\mathbf r_j
 - L_{\mathrm{esc},j}\,\delta\omega(E),
$$

where the last term is the in-medium propagation phase the ledger names. Then

$$
\lvert A\rvert^2 = \sum_j \lvert A_j\rvert^2
 + \sum_{j\ne k} \lvert A_j\rvert\lvert A_k\rvert
   \cos\bigl(\phi_j(E)-\phi_k(E)\bigr),
$$

and each cross term oscillates in $E$ at
$\mathrm d(\phi_j-\phi_k)/\mathrm dE = (d_j-d_k)/\hbar c$, the difference of
per-segment slopes. $-\mathbf g\cdot\mathbf r_j$ carries no $E$ and contributes
only a constant offset to each cross phase.

**In-medium term.** With $\delta\omega = (1-n_{\rm re})\omega$ and
$1-n_{\rm re} = \delta \propto \omega^{-2}$ away from edges,
$\mathrm d(\delta\,\omega)/\mathrm dE = -\delta/\hbar c$. The term therefore
shifts the cross-term rate by
$\delta\,(L_{\mathrm{esc},j}-L_{\mathrm{esc},k})/\hbar c$. It is negligible when
$\delta\,\Delta L_{\rm esc} \ll D$. With $\delta \lesssim 10^{-4}$ and the
smallest quoted span $D \sim 4\times10^{5}$ Å, this needs
$\Delta L_{\rm esc} \ll 4\times10^{9}$ Å, i.e. 0.4 m, which holds for every
crystal. The ledger's "smaller by $1-n_{\rm re}$" is correct once read as
$\delta\,\Delta L_{\rm esc}$ against $D$, not $\delta$ against 1.

**Per-segment increment.** From $d(t) = t - \hat{\mathbf n}\cdot\mathbf r(t)$,
$\mathrm dd/\mathrm dt = 1-\beta\hat{\mathbf v}\cdot\hat{\mathbf n}$ is constant
on a straight segment, so

$$
\Delta d_j = (1-\beta\hat{\mathbf v}_j\cdot\hat{\mathbf n})\,t_L
$$

exactly, provided $t_L$ is the flight *time* $L/\beta$. Since
$\beta<1$, $d$ is strictly increasing along a trajectory. For one contiguous
electron the span is therefore a sum of increments:
$\max d - \min d = \sum_j \Delta d_j - (\Delta d_{\rm first}+\Delta d_{\rm last})/2$,
which grows as $N_{\rm seg}\,\overline{\Delta d}$. The sinc first zero
$\Delta E = 2\pi\hbar c/\Delta d_j$ follows one increment, so the ratio between
the two scales grows with the segment count. A fixed safety factor cannot
connect them.

**Band limit.** In the conjugate variable $\tau$ (Å) of $E/\hbar c$, $A_j$ is the
Fourier transform of a box of width $\Delta d_j$ centred on $d_j$. The support
of $\lvert A\rvert^2$ is the difference set of the union of those boxes:
$\lvert\tau\rvert \le D_{\rm end} = \max_j(d_j+\Delta d_j/2)-\min_k(d_k-\Delta d_k/2)$.
This gives two sampling rules, a factor of 2 apart:

- the spectral-shape (Nyquist) step $h \le \pi\hbar c/D_{\rm end}$;
- exact uniform-trapezoid integrated yield (Poisson summation: aliases sit at
  $\tau = 2\pi\hbar c\,m/h$) for $h < 2\pi\hbar c/D_{\rm end}$.

When $D \gg \Delta d$,
$D_{\rm end} = D + O(\Delta d)$, giving $h_{\rm coh} = \pi\hbar c/D$ with
$D = \max_j d_j - \min_j d_j$. This is the shape-Nyquist step, which is
conservative by about 2 for integrated yield. Taken literally with midpoints,
it is larger than the strict $\pi\hbar c/D_{\rm end}$ by the factor
$D_{\rm end}/D = 1 + O(1/N_{\rm seg})$.

**Limits.** For a single segment, $D=0$, the midpoint estimator gives
$h\to\infty$, and the only band limit is the envelope's own
$D_{\rm end} = \Delta d$. The spectrum then reduces to the incoherent
$t_L^2\,\mathrm{sinc}^2$, whose shape-Nyquist step is $\pi\hbar c/\Delta d$:
half the first-zero distance $2\pi\hbar c/\Delta d$. As $D\to\infty$, $h\to 0$.
Units: $[\hbar c/D] = $ eV Å / Å $=$ eV.

**Refusal.** `sinc-nyquist` derives its step from $2\pi\hbar c/\Delta d_j$, one
increment, while the coherent cross terms need $\pi\hbar c/D$ with
$D \approx N_{\rm seg}\overline{\Delta d}$ per electron (more across electrons).
Without windows, the automatic step therefore aliases the coherent route by a
factor of about $2N_{\rm seg}$. Refusing is the correct consequence, and
refusing unconditionally, rather than comparing two steps, is also sound.

### Comparison with the implementation

| item | independent | implementation | result |
|---|---|---|---|
| phase slope | $d_j/\hbar c$ | `d_all_geom / HBARC_EV_ANG` (`_batched.py`), `arg = d*omega - g_phase - L_esc*dom` (`_per_hkl.py`) | match |
| $d_j$ | $t_{\rm start}+t_L/2-\hat{\mathbf n}\cdot\mathbf r_{\rm mid}$, $t_L=L/\beta$ | `(start_time + 0.5*t_L) - r_mid @ n_vec`, `t_L = length/beta` | match |
| step | $\pi\hbar c/D$ | `np.pi * HBARC_EV_ANG / span` | match (midpoint convention, see above) |
| grouped | widest single-electron span | max over `elec_id` groups of `part.max()-part.min()` | match |
| single segment | estimator $\infty$; envelope governs | returns `(inf, 0.0, 1)` | match; the estimator does **not** itself return the incoherent width |
| infinite slab, active decoherence | reducer subtracts $(x_{0,e},y_{0,e})$ from $\mathbf r$, so $d \mathrel{+}= \hat{\mathbf n}_\perp\cdot\mathbf r_{0\perp,e}$ | `d + xy0_pop[owner] @ n_vec[:2]` | match (sign checked) |
| **finite footprint, active decoherence** | reducer phases with offset-free geometry | `coherent_retardation` keeps sampled positions | **differs** |
| refusal | refuse every non-windowed coherent case | raises unconditionally when `coherent_emission` and `windows is None`; steps only feed the message | match |

**Divergence, finite footprint.** Since `917ec678a` (2026-10-08, #370),
`cxr/_setup.py::_coherent_offsets` applies `_offset_free_geometry` on the
analytic finite-footprint branch as well. The reducer therefore phases finite
footprints with $\mathbf r - (x_{0,e},y_{0,e},0)$ too.
`diagnostics.coherent_retardation` still applies the add-back only
`if not finite_footprint`, and its docstring says "A finite footprint ... keeps
the sampled positions". The first divergent term is the missing
$+\hat{\mathbf n}_\perp\cdot\mathbf r_{0\perp,e}$ on finite footprints. This
breaks the claim's "in the reducer's geometry": the reported all-electron span
includes the beam footprint projected on $\hat{\mathbf n}$, which the reducer's
flat term no longer carries. The ledger statement "Finite-footprint cases,
including every measurement above, are unchanged" predates `917ec678a` by three
days and is stale for the current reducer. The error is conservative: the step
comes out too fine.

A second, smaller mismatch is on the `physical_electrons` route without
analytic offsets. There, `_setup.py` swaps in the realized `d_all`, which
includes `t0_ang` and the sampled positions. The estimator removes both, so it
under-reports $D$, the non-conservative direction.

Neither mismatch changes the refusal decision, because the refusal is
unconditional. Only the steps and spans quoted in the error message are
affected, and `coherent_fringe_spacing` has no other production consumer.

### Numeric checks

The independent script, outside the repository, builds 40 contiguous synthetic
30 keV segments for one electron. It evaluates the exact straight-segment
amplitude sum $\lvert\sum_j A_j\rvert^2$ in closed form. It does not use
implementation helpers.

- Increment identity:
  $\max\lvert\Delta d - (1-\beta\hat{\mathbf v}\cdot\hat{\mathbf n})t_L\rvert = 1.2\times10^{-11}$ Å
  (roundoff).
- Spans: $D = 104374$ Å and $D_{\rm end} = \sum\Delta d = 106724$ Å
  ($+2.25\%$). The FFT of the densely sampled spectrum holds less than
  $10^{-9}$ of its power beyond $\tau = 106712$ Å, confirming that the support
  ends at $D_{\rm end}$.
- Trapezoid yield error against an $h_{\rm coh}/64$ reference:
  $7.6\times10^{-10}$ at $h_{\rm coh}/2$, $1.8\times10^{-9}$ at $h_{\rm coh}$,
  $2.4\times10^{-3}$ at $2h_{\rm coh}$ (just past $2\pi\hbar c/D_{\rm end}$),
  $2.4\times10^{-2}$ at $4h_{\rm coh}$, and $5.9\times10^{-2}$ at
  $8h_{\rm coh}$. Linear-interpolation $L_1$ shape error is $13\%$ at
  $h_{\rm coh}$: the Nyquist step is a band limit, not an accuracy target.
- At the sinc step (3.776 eV), the incoherent yield error is
  $1.2\times10^{-5}$ while the coherent yield error is $0.198$, reproducing
  the separation in principle.
- `coherent_fringe_spacing` on the same segments returns step 0.05939418285 eV
  and span 104374.0249 Å, equal to the independent values (relative
  difference 0). With `grouped=True` on one electron it gives the same result.
  A single segment returns `(inf, 0.0, 1)`. `sinc_feature_spacing` at
  `aliased_weight_limit=0` returns 3.7763023 eV, the independent minimum
  first zero.
- Two electrons with a $5\times10^{5}$ Å transverse offset on an infinite
  slab give span 104374 Å, offset removed as intended. With
  `crystal_width_ang`/`crystal_height_ang` set, the span is 570394 Å, the
  sampled geometry, against the reducer's current offset-free 104374 Å. This is
  the divergence above.
- Ledger arithmetic: $\pi\hbar c/2.5298\times10^{7}$ Å $= 2.450\times10^{-4}$ eV,
  giving $1.506\times10^{7}$ points over 3690 eV. For the grouped span,
  $\pi\hbar c/4.1553\times10^{5}$ Å $= 1.492\times10^{-2}$ eV. Both are
  consistent with the tables, but the spans themselves were not reproduced:
  that needs hopg transport, which was not run locally.
- Anchor file `tests/energy-grid/test_coherent_line_grid.py`: 7 passed.

### Verdict

**`discrepancy`, narrowly scoped.** The band-limit physics is rederived and
matches: the cross-term rate as a difference of slopes,
$-\mathbf g\cdot\mathbf r_j$ dropping out, the in-medium slope being
negligible, the exact increment identity, $h_{\rm coh}=\pi\hbar c/D$ up to the
$O(1/N_{\rm seg})$ midpoint-versus-endpoint convention, and the refusal logic.
The divergence is the claim's "in the reducer's geometry" clause. For finite
footprints with active decoherence, `coherent_retardation` omits the
offset removal that the reducer has applied since `917ec678a`. On the
`physical_electrons` route it removes offsets that the reducer keeps. The
ledger's single-segment check should read "the estimator returns $\infty$ and
the per-segment envelope, at $\pi\hbar c/\Delta d$, governs", not "returns the
incoherent width". Sign-off remains a human decision.
