# Validation: segment-escape-average

## Claim

- **ID:** `segment-escape-average`
- **Symbols:** `montecarlo/spectrum/segment_escape.py::{segment_escape_paths,mean_transmission,segment_escape_pieces,piece_mean_transmission}`; `montecarlo/spectrum/brem_jit_kernel.py::_transmission_scalar`
- **Consumers:** `montecarlo/spectrum/characteristic.py::mc_characteristic_spectrum`, `montecarlo/spectrum/brem.py::mc_brem_spectrum` (CPU and fused CUDA reductions); since issue #181 the incoherent PXR/CBS line route, `montecarlo/spectrum/lines/_batched.py::_batched_incoherent_block` and `montecarlo/spectrum/lines/_per_hkl.py::_accumulate_reflection` (CPU, and CUDA through the same array code)
- **Source:** Beer--Lambert law (`self-absorption`) integrated over a uniform line source; no new physical law
- **Intended quantity:** the escaping fraction of photons emitted uniformly along one straight transport segment, for incoherent emitters (characteristic lines, bremsstrahlung)

## Derivation

A track-length estimator scores an incoherent emitter as $n\,\sigma(T)\,L$ photons per segment of length $L$, spread uniformly along
$\mathbf r(s)=\mathbf r_0+s\hat{\mathbf v}$, $0\le s\le L$. The cross section is held at the segment's representative energy (`substep-radiation-invariance`); only escape varies along the segment. The escaping yield is therefore

```{math}
:label: eq-segment-escape-yield

Y = n\,\sigma(T)\int_0^L e^{-\tau(s)}\,ds = n\,\sigma(T)\,L\,\langle e^{-\tau}\rangle,
\qquad
\tau(s)=\sum_i \mu_i(E)\,\ell_i(\mathbf r(s)),
```

with $\ell_i$ the photon's path in layer $i$ along the observation direction $\hat{\mathbf n}$ (`self-absorption`). The previous code used $e^{-\tau(L/2)}$. Because $e^{-\tau}$ is convex, Jensen's inequality makes that an undercount whenever $\tau$ varies along the segment. The size of the undercount depends on segment length, so it moves with anything that changes free paths (issue #176).

**Linear pieces.** Suppose $\tau$ is affine on $s\in[s_a,s_b]$, with endpoint values $\tau_a,\tau_b$ and $\Delta=|\tau_b-\tau_a|$. Then

```{math}
:label: eq-segment-escape-mean

\frac{1}{s_b-s_a}\int_{s_a}^{s_b}e^{-\tau(s)}\,ds
= e^{-\min(\tau_a,\tau_b)}\,\frac{1-e^{-\Delta}}{\Delta}
= e^{-\bar\tau}\,\frac{\sinh(\Delta/2)}{\Delta/2},
```

where $\bar\tau$ is the midpoint value. The first form is the one implemented: both of its factors are at most 1, so it cannot overflow however opaque the layer. The second is the issue's closed form. Expanding the first gives the second: $e^{-\tau_{\min}}(1-e^{-\Delta})/\Delta = e^{-\tau_{\min}-\Delta/2}\,(e^{\Delta/2}-e^{-\Delta/2})/\Delta$.

**Where $\tau$ is affine.** Along a straight segment:

1. *Laterally infinite planar stack.* Inside the emitting layer, that layer's path $(z-z_\text{top})/|n_z|$ or $(z_\text{bot}-z)/|n_z|$ is affine in $s$. Every other traversed layer contributes a constant $\Delta z_i/|n_z|$. So $\tau$ is affine between the points where the segment crosses a layer boundary. The code cuts there.
2. *Finite rectangular footprint.* The exit distance $D(\mathbf r)=\min_a (f_a-r_a)/n_a$, taken over the faces the ray approaches, is the minimum of affine functions. It is affine between the points where two face distances are equal. Each layer path is the ray length between clipped planes, which is affine in $(z, D)$ except where the exit point $z+n_zD$ crosses a layer boundary. The code cuts at source-layer crossings, face-distance equalities, and exit-point layer crossings. After these cuts every $\ell_i$, and therefore $\tau$, is affine on each piece.
3. *Blazed groove.* `escape_distance_ang` is affine in position inside one sawtooth cell and one valley band. It jumps at relief planes (cell boundaries in the $d=x\cos\theta-z\sin\theta$ coordinate) and at valley-depth band edges. The code cuts at both families of planes. It evaluates each piece's endpoints with the cell and band of that piece's open interval (one-sided values), not with `floor()` on the shared boundary.

**Piece endpoints.** A piece endpoint can lie on a crystal face. Every track starts on the entrance plane $z=0$. When $\hat{\mathbf n}$ leaves through that face, the exit distance there is 0, but `first_prism_exit` skips a face at distance 0 and returns the next one, which is millimetres away. Evaluating $\ell_i$ at the endpoint would then put the first segment of every electron behind the whole footprint. The paths are affine on each piece, so the code samples them at $1/4$ and $3/4$ of the piece and extrapolates, $\ell(0)=\tfrac32\ell(\tfrac14)-\tfrac12\ell(\tfrac34)$ and $\ell(1)=\tfrac32\ell(\tfrac34)-\tfrac12\ell(\tfrac14)$. This is exact for an affine function and gives the one-sided limit from inside the piece. The laterally infinite single slab evaluates its closed-form depth path at the endpoints directly, because that path is continuous at the faces.

Summing {eq}`eq-segment-escape-mean` over the pieces, each weighted by its length fraction, reproduces the integral in {eq}`eq-segment-escape-yield` exactly. The estimator becomes $\sum_p n\,\sigma\,(f_pL)\,\langle e^{-\tau}\rangle_p$ with $\sum_p f_p=1$.

## Limits and checks

- **Units:** $\mu$ [Å$^{-1}$] × $\ell$ [Å] gives a dimensionless $\tau$. $\langle e^{-\tau}\rangle$ is dimensionless, and $f_pL$ keeps the estimator's length units.
- **$\mu\to0$ or $\Delta\to0$:** the ratio tends to 1 and the mean tends to $e^{-\bar\tau}$, which is the midpoint rule. The correction is second order: $\langle e^{-\tau}\rangle=e^{-\bar\tau}(1+\Delta^2/24+O(\Delta^4))$.
- **$\Delta\to\infty$:** the mean tends to $e^{-\tau_{\min}}/\Delta$, which is the exact integral dominated by the near-surface end.
- **Bounds:** $e^{-\bar\tau}\le\langle e^{-\tau}\rangle\le e^{-\tau_{\min}}$ (Jensen above; monotonicity below).
- **Symmetry:** the mean is invariant under swapping $\tau_a$ and $\tau_b$.
- **Additivity:** splitting a segment into collinear pieces leaves $\int e^{-\tau}\,ds$ unchanged. This is the split-invariance acceptance test.
- **Numerics:** below $\Delta=10^{-4}$ (CPU, float64) the series $1-\Delta/2+\Delta^2/6$ replaces $-\operatorname{expm1}(-\Delta)/\Delta$. Its truncation error $\Delta^3/24<5\times10^{-14}$ is below the cancellation it avoids. The float32 CUDA twin switches at $\Delta=10^{-3}$, with error $4\times10^{-11}$, far below float32 resolution.

## Assumptions and scope

- Emission is uniform along the segment. The cross section and line energy are evaluated once per segment, as before.
- Photons travel in a straight line with no refraction and no re-entry. The groove's no-re-entry proof is in `blazed-groove-geometry`.
- Attenuation is constant inside each layer.
- **Out of scope:**
  - The coherent PXR/CBS route and the flight-grouped incoherent reduction (a coherent sum per physical flight). A coherent emitter needs absorption inside the formation integral, a complex exponent per linear piece, not a plain average of the intensity; that treatment is `coherent-formation-absorption` (issue #181, second slice). Its Parseval integral is this mean.
  - Hard radiative events (`brem_events.py`) emit at a point, the segment endpoint, so they need no average.

## Incoherent PXR/CBS line route (issue #181)

One segment's line intensity is the squared formation integral. With absorption damping the amplitude by $e^{-\tau(s)/2}$ along the segment and $q$ the detuning phase rate,

```{math}
:label: eq-segment-escape-line

\frac{d^2N}{dE\,d\Omega}\propto\Bigl|\int_0^{t_L}e^{iqt-\tau(t)/2}\,dt\Bigr|^2,
\qquad
\int_{-\infty}^{\infty}\Bigl|\int_0^{t_L}e^{iqt-\tau(t)/2}\,dt\Bigr|^2 dq
=2\pi\int_0^{t_L}e^{-\tau(t)}\,dt=2\pi\,t_L\,\langle e^{-\tau}\rangle
```

by Parseval. The route's lineshape $t_L^2\,\mathrm{sinc}^2(\cdot)\,T$ integrates to $2\pi t_L T$ over $q$, so taking $T=\langle e^{-\tau}\rangle$ from {eq}`eq-segment-escape-mean` gives the exact integrated line yield; $T=e^{-\bar\tau}$ was the Jensen undercount. The escape paths are $g$-independent and are cut once per call (`segment_escape_pieces`, padded to one row per segment); $\mu(E_\text{res})$ varies per (segment, reflection) and is applied per piece (`piece_mean_transmission`). Layered stacks use each layer's $\mu$, the groove the absorber $\mu$, as the midpoint code did.

**Line-shape decision.** The route keeps the undamped $\mathrm{sinc}^2$ shape at $E_\text{res}$. The exact single-segment shape, $|(e^{zt_L}-1)/z|^2$ with $z=iq-\kappa$, is a damped, $\kappa$-broadened sinc; relative to the kept shape its pointwise error is $O(\Delta)$ within one segment while its integral is unchanged. The ensemble line width is set by the resonance spread across segments, mosaic and detector response, far above a single segment's $1/t_L$, so carrying the damped shape through the incoherent lineshape kernels (node, bin-mean, fused CUDA) is not worth its cost.

**Consequence.** The dataset and case-content identities hash the constant `line_escape_model` marker, orphaning midpoint-era line spectra once (`segment-mean-incoherent-v1`, then `segment-mean-v2-coherent-formation` with the coherent slice). Since that slice one segment's coherent self-term integrates to exactly this yield too, so single-segment coherent and incoherent spectra agree in integral and differ only in shape and in the $O(\delta)$ line-centre offset described under `coherent-formation-absorption`.

## Evidence

- `tests/montecarlo/test_segment_escape.py`:
  - quadrature agreement to $10^{-9}$ across the series/direct switch and at $\tau$ up to 800;
  - endpoint symmetry;
  - split invariance at $10^{-6}$ for characteristic slab, layered-stack and bremsstrahlung yields;
  - exact finite-box side exit through a thin W layer, independent of splitting;
  - a track starting on the entrance face of a finite box (single slab and layered), escaping back through it, against quadrature at $10^{-10}$;
  - groove split invariance at $10^{-10}$ and agreement with dense midpoint quadrature at $2\times10^{-4}$;
  - an absorbed-to-transparent ratio equal to the segment mean, which exceeds twice the midpoint factor in the issue's regime.
- `tests/montecarlo/test_spectrum_cuda_cheap_hoists.py` pins the float32 CUDA reduction against the NumPy `mean_transmission` reference for distinct endpoint paths. This is CUDA-gated.
- Model markers fork on the change: characteristic `l-shell-ck-lorentzian-segment-escape-v6`, bremsstrahlung `...-v3-unit-base-segment-escape`.
- `tests/montecarlo/test_line_segment_escape.py` (issue #181): the incoherent line spectrum's absorbed-to-transparent ratio is split invariant to $10^{-9}$ (measured $\sim10^{-15}$) on the batched slab (ratio 0.56), the batched finite box whose exit face switches along the segment (0.72) and the per-hkl Cu-film-on-crystal stack crossing the interface (0.23); absorption rescales the single line without reshaping it; the padded piece layout reproduces the flat `owner` sum.

## Fresh-context re-derivation

A separate context that did not write the implementation re-derived the claim (2026-09-25). It started from the ledger row and Beer--Lambert, derived the results below, and only then read `segment_escape.py`, the `_transmission_scalar` twin and the two consumers.

**Segment mean.** Put $\tau(t)=\tau_a+(\tau_b-\tau_a)t$ for $t\in[0,1]$. Then

$$
\int_0^1 e^{-\tau(t)}\,dt=\frac{e^{-\tau_a}-e^{-\tau_b}}{\tau_b-\tau_a}.
$$

If $\tau_b\ge\tau_a$, factoring out $e^{-\tau_a}$ gives $e^{-\tau_a}(1-e^{-\Delta})/\Delta$. If $\tau_b<\tau_a$, factoring out $e^{-\tau_b}$ gives $e^{-\tau_b}(1-e^{-\Delta})/\Delta$. Both cases equal $e^{-\tau_{\min}}\,g(\Delta)$, with $g(\Delta)=(1-e^{-\Delta})/\Delta$ and $\Delta=\lvert\tau_b-\tau_a\rvert$. This matches {eq}`eq-segment-escape-mean`. The Taylor series is $g(\Delta)=1-\Delta/2+\Delta^2/6-\Delta^3/24+O(\Delta^4)$. Truncating after $\Delta^2/6$ leaves a relative error of $\Delta^3/24$: $4.17\times10^{-14}$ at the CPU switch $\Delta=10^{-4}$ and $4.17\times10^{-11}$ at the CUDA switch $\Delta=10^{-3}$. Writing the result about the midpoint gives $e^{-\bar\tau}\sinh(\Delta/2)/(\Delta/2)=e^{-\bar\tau}(1+\Delta^2/24+\Delta^4/1920+\dots)$.

**Cut set.** In the sample-frame box $[-W/2,W/2]\times[-H/2,H/2]\times[0,T]$, the exit distance is $D(\mathbf r)=\min_a(f_a-r_a)/n_a$, taken over the faces with $n_a\ne0$ that the ray approaches. This includes the $z$ face. Along $\mathbf r(s)$ each term is affine, so $D$ can change slope only where two terms are equal. The layer-$i$ path is

$$
\ell_i=\max\!\left(0,\;\min(D,t^{i}_{\max})-\max(0,t^{i}_{\min})\right),
\qquad t^{i}=\frac{z_{i}-z}{n_z}.
$$

It has kinks only where

- $t^i_{\min}$ or $t^i_{\max}$ is 0, meaning the source crosses an interior boundary;
- $D$ equals $t^i_{\min}$ or $t^i_{\max}$, meaning the exit point $z+n_zD$ crosses an interior boundary;
- $D$ itself kinks.

The stack's outer boundaries are $z=0$ and $z=T$. On a piece where the $z$ face attains $D$, the exit point stays on that face, so the outer boundaries add no breakpoint. The implementation cuts at all three families. For the exit-point crossings it cuts at the crossing of every candidate face's exit point, not only the minimizing face's. Because the face-equality cuts fix the minimizing face on each piece, this is a superset and is still exact. For the groove, `escape_distance_ang` is $L=(k+1)c-d+\max(m,0)\,c$ with $k=\lfloor d/c\rfloor$ and $m=\lfloor z_1/h\rfloor$. Here $z_1=z-\sin\theta_t\,((k+1)c-d)$ is affine for fixed $k$. So $L$ is affine for fixed $(k,m)$, and it jumps only on the relief planes $d=kc$ and the band planes $z_1=mh$. `_groove_segment_paths` cuts at both families. It takes $k$ and $m$ from the midpoint of each open piece, and it extends the plane range far enough to cover $\lvert\Delta d\rvert$ and $\lvert\Delta z_1\rvert$. **No missing breakpoint was found** for a contiguous stack spanning $[0,T]$.

**Endpoint extrapolation.** For affine $\ell$ on $[0,1]$, $\tfrac32\ell(\tfrac14)-\tfrac12\ell(\tfrac34)=\ell(0)$ and $\tfrac32\ell(\tfrac34)-\tfrac12\ell(\tfrac14)=\ell(1)$ exactly. The samples lie strictly inside the open piece, so the extrapolated values are the one-sided limits, whatever `first_prism_exit` does on a face. The clamp to $\ge0$ only absorbs rounding.

**Float32 twin.** `_transmission_scalar` accumulates $\tau_0$ and $\tau_1$ over the layers with the same $\mu$ per layer. It then applies the same $\min$ and $\lvert\cdot\rvert$ forms, the series below $10^{-3}$, and `expm1` above it. Its layout `(n_seg, 2, n_layers)` matches `xp.stack((path_start, path_end), axis=1)` in `mc_brem_spectrum`, and `n_abs_layers = shape[2]`. The consumers reindex `seg_L` (times `fraction`), `seg_E` and the BremsLib `v_hat` by `owner`.

**Independent numeric checks** (scratch scripts, CPU backend):

| check | result |
| --- | --- |
| `mean_transmission` against an mpmath (40-digit) closed form, $\tau_a\in\{0,\dots,700\}$, $\Delta\in[10^{-9},10^3]$, including $\Delta=10^{-4}\pm10^{-9}$ | max relative error $4.2\times10^{-14}$, the series truncation just below the switch |
| $(\langle e^{-\tau}\rangle/e^{-\bar\tau}-1)/(\Delta^2/24)$ at $\Delta=10^{-3},10^{-2},10^{-1}$ | $1.000000008$, $1.0000012$, $1.000125$ |
| $\langle e^{-\tau}\rangle\,\Delta\,e^{\tau_{\min}}$ at $\Delta=50,200,600$ | $1.0$ |
| bounds, symmetry and split additivity over $10^4$ random pairs | bounds hold; symmetry exact; additivity $3\times10^{-15}$ |
| float32 emulation of `_transmission_scalar` against the exact value | max relative error $1.4\times10^{-7}$ (float32 rounding); a naive `1-exp` form would give $1.1\times10^{-5}$ at $\Delta=1.1\times10^{-3}$ |
| 60 random box and layer cases (1 to 3 layers, random $\hat{\mathbf n}$, every third case starting on $z=0$) against a separately written box/layer ray oracle integrated with a $2\times10^6$-point midpoint rule | worst relative difference $5.9\times10^{-10}$ |
| entrance-face start exiting through $z=0$; grazing side exit; segment ending on the exit face | $8\times10^{-13}$, $4\times10^{-11}$, $3\times10^{-13}$ |
| laterally infinite layered and single slab, 40 cases | $3.7\times10^{-9}$ |
| groove (12 cases) against a dense midpoint rule over `escape_distance_ang`, with $N=10^5,10^6,10^7$ | $1.1\times10^{-4}$, $5.8\times10^{-6}$, $4.6\times10^{-7}$, converging as $1/N$ as the jumps require |

`pyrite-dev test tests/montecarlo/test_segment_escape.py tests/montecarlo/test_spectrum_cuda_cheap_hoists.py` gives 15 passed and 1 skipped (CUDA).

**Observations (not discrepancies).**

- `_cut_at_zero` returns `start` itself when `start == end`. A constant quantity whose value in Å falls in $(0,1)$ therefore adds a spurious cut. One example is the $z$-face exit-point term $f_z-z_b$ when a layer boundary lies within 1 Å of the exit face. An extra collinear cut leaves the result exact by split additivity, so this costs only a piece.
- The cut set relies on the documented precondition that the layers are contiguous and span $[0,T]$ with $T=$ `thickness_ang` (`_stack_tau` docstring). `segment_escape_paths` does not enforce it. A stack ending above $T$ would need an extra exit-point cut at its last $z_{\rm bot}$.
- The groove oracle `escape_distance_ang` is the separately ledgered `blazed-groove-geometry` claim. This check covers only the averaging over it.

**Verdict:** `rederived`. The derivation matches the implementation symbolically, and it matches numerically to the brute-force resolution. The CUDA hardware parity run, the #176 split ladder and the line-route quantification are still pending. They do not affect this verdict.

## Status

`rederived` (fresh-context re-derivation above, 2026-09-25). Human sign-off pending. Pending:

- CUDA hardware execution of the parity tests;
- fresh-context verification of the issue #181 incoherent-line extension (the Parseval step and the padded piece layout);
- re-measurement of the #176 hopg C K split ladder;
