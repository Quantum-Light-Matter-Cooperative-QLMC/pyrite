# `coherent-line-grid-fringe-spacing`

Ledger row:
[`coherent-line-grid-fringe-spacing`](../ledger-core-coherent-physics.md).
Instrument: `montecarlo/spectrum/diagnostics.py::coherent_fringe_spacing`
(estimator), `montecarlo/runner/line_grid.py::_refuse_coherent_resolution`
(policy), `energy_grid/convergence_case.py` (ladder, now route-switchable).
No kernel change: this row records the band limit of the existing coherent
reducer and the evidence for refusing automatic resolution on that route
(issue #117).

## Claim

1. **Band limit.** On the coherent route the line-grid Nyquist step is
   $h_\mathrm{coh} = \pi \hbar c / D$, where $D$ is the span of the
   per-segment retardation scalar $d_j = t_{\mathrm{mid},j} - \hat n \cdot
   \mathbf r_j$, not the per-segment sinc width of
   `sinc_feature_spacing`.
2. **Separation of principle.** Along one segment $d$ increments by exactly
   $(1 - \beta\, \hat v \cdot \hat n)\, t_L$, the quantity setting the
   incoherent first zero. The incoherent step therefore follows the
   *per-segment* increment and the coherent step the *total* span, so the
   incoherent estimator cannot bound the coherent route at any calibration.
3. **Measured.** At the spacing where the incoherent yield is converged to
   $3.1\times10^{-8}$, the coherent yield is still $5.0\times10^{-2}$ from
   convergence.
4. **Policy.** Automatic resolution refuses `coherent_emission` rather than
   silently aliasing. An explicit `E_grid_line` is unaffected.

## Derivation

The coherent reducer
(`lines/_per_hkl.py::_accumulate_reflection_coherent`, and the batched
steps 5/7) builds one complex field per (reflection, orientation) row,

$$
A(E) = \sum_j c_j \,
\mathrm{sinc}\!\left(\frac{a_{\mathrm{width},j} (E - E_{r,j})}{\pi}\right)
e^{i \phi_j(E)},
\qquad
\phi_j(E) = d_j \omega(E) - \mathbf g \cdot \mathbf r_j
          - L_{\mathrm{esc},j}\, \Delta\omega(E),
$$

with $\omega(E) = E / \hbar c$ and $d_j = t_{\mathrm{mid},j} - \hat n \cdot
\mathbf r_j$ in ångström at $c=1$ (`lines/_setup.py`). The spectrum is
$|A(E)|^2$, so it carries cross terms $j,k$ whose phase runs over the output
grid at the **difference of per-segment phase slopes**

$$
s_j = \frac{\mathrm d \phi_j}{\mathrm d E}
    = \frac{d_j}{\hbar c}
      - L_{\mathrm{esc},j} \frac{\mathrm d (\Delta\omega)}{\mathrm d E}.
$$

The $-\mathbf g \cdot \mathbf r_j$ term is energy-independent and drops out.
The in-medium term is smaller by $1 - n_\mathrm{re} \sim 10^{-5}$ for x-rays
and is neglected. The fastest cross term therefore has period
$2\pi / (s_\max - s_\min)$, giving the Nyquist step

$$
h_\mathrm{coh} = \frac{\pi}{s_\max - s_\min}
              \simeq \frac{\pi \hbar c}{D},
\qquad D = \max_j d_j - \min_j d_j .
$$

### Limiting case

For a single segment, $D \to 0$ and $h_\mathrm{coh} \to \infty$: the only
remaining scale is the segment's own $\mathrm{sinc}^2$ envelope, and the
coherent route collapses to the incoherent self-term $|A|^2 t_L^2
\mathrm{sinc}^2$ (the limit already pinned by `test_coherent_emission.py`).
The band limit is then correctly the incoherent one.

### Why the incoherent estimator cannot be rescaled into this one

Along a segment of length $L$ at speed $\beta$, $t$ advances by $t_L = L/\beta$
and $\hat n \cdot \mathbf r$ by $L\, \hat v \cdot \hat n$, so

$$
\Delta d = t_L - L\,\hat v \cdot \hat n
         = (1 - \beta\, \hat v \cdot \hat n)\, t_L ,
$$

which is exactly the combination whose reciprocal sets the incoherent first
zero $2\pi\hbar c / [(1 - \beta \hat v \cdot \hat n) t_L]$ in
`sinc_feature_spacing`. The incoherent width is set by one increment; the
coherent fringe by the sum of all of them. Their ratio grows with the number
of coherently summed segments, so no fixed safety factor on the incoherent
estimate bounds the coherent route.

## Decoherence selects which span applies

`decoherence_active` (`lines/_setup.py`) is true whenever the case carries
nonzero bunch offsets or a finite footprint — including the standard
`beam_fwhm_mm = 1.0` cases. The spectrum is then the blend
$(1-F)\sum_e |S_e|^2 + F |\sum_e S_e|^2$ on the offset-free geometric phase:

- the $\sum_e |S_e|^2$ floor is band-limited by the widest **single-electron**
  span (`coherent_fringe_spacing(..., grouped=True)`);
- the $|\sum_e S_e|^2$ term is band-limited by the **all-electron** span.

The all-electron span is set by the transverse beam extent projected on
$\hat n$, so it is ~2.5 mm in every case measured below, essentially
independent of material and beam energy. Those inter-electron fringes are one
realization of the sampled offsets — speckle, which `lines/_setup.py` already
notes "does not shrink with electron count". Whether they are an observable
at all is a modelling question outside this row; see the follow-up issue
recorded in #117.

## Measurements

Transport: `build_ladder_case`, seed 7, tilt 5°, 1 mm, `bunch_length_fs = 100`
(standard ladder cases carry none, and the coherent route refuses without it
once decoherence is active).

Band limits at $N_e = 200$, against the incoherent estimate at
$\varepsilon = 10^{-3}$:

| case | `sinc_feature_spacing` [eV] | $D$ all-electron [Å] | $h_\mathrm{coh}$ all-e [eV] | $h_\mathrm{coh}$ per-e [eV] | ratio all-e / per-e |
|---|---|---|---|---|---|
| hopg 30 keV | 0.974 | $2.53\times10^{7}$ | $2.45\times10^{-4}$ | $1.49\times10^{-2}$ | 3975× / 65× |
| hopg 100 keV | 0.481 | $2.65\times10^{7}$ | $2.34\times10^{-4}$ | $2.81\times10^{-3}$ | 2055× / 171× |
| wse2 30 keV | 7.85 | $2.50\times10^{7}$ | $2.48\times10^{-4}$ | $3.41\times10^{-2}$ | 31640× / 230× |

Refinement ladder, hopg 30 keV, $N_e = 40$, band 10–3700 eV, both routes on
**identical trajectories** (one transport, injected into both ladders;
`sinc_feature_spacing` estimate 1.504 eV):

| $h$ [eV] | points | incoherent yield $\Delta_\mathrm{rel}$ | coherent yield $\Delta_\mathrm{rel}$ | coherent centroid $\Delta$ [eV] |
|---|---|---|---|---|
| 4.0 | 923 | $2.49\times10^{-1}$ | $1.18\times10^{-1}$ | 0.36 |
| 2.0 | 1846 | $8.16\times10^{-2}$ | $5.32\times10^{-2}$ | 0.67 |
| 1.0 | 3691 | $3.08\times10^{-8}$ | $5.01\times10^{-2}$ | 1.38 |
| 0.5 | 7381 | $7.29\times10^{-9}$ | $4.72\times10^{-3}$ | 0.22 |

The incoherent route locks in at the sinc estimate, reproducing
`line-grid-sinc-convergence`. The coherent route does not: it remains above
the $10^{-3}$ intrinsic-source tolerance at three times finer spacing, and its
centroid drift does not shrink monotonically, which is the signature of newly
resolved fringes rather than converging quadrature.

## Policy consequence

Resolving the all-electron fringes over a keV band costs $\sim1.5\times10^{7}$
points against 2453 at the incoherent estimate; the grouped floor alone costs
$\sim2.5\times10^{5}$. Neither is an affordable default, and silently spending
either would misrepresent an unconverged spectrum as a converged one.
`_refuse_coherent_resolution` therefore raises for `coherent_emission` on the
automatic path, reporting both derived steps and their spans, and directs the
caller to an explicit `E_grid_line`. Explicit grids, stored catalog rows, and
the whole incoherent route are unaffected.

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

Heavier ladders (larger $N_e$, 100/300 keV, tilt 85°) belong on `pyrite
remote`; the numbers above are the CPU-affordable subset and are sufficient to
separate the two routes by six orders of magnitude at the policy spacing.
