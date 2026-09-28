# coherent-formation-absorption

## Claim

- **ID:** `coherent-formation-absorption`
- **Symbols:** `montecarlo/spectrum/lines/_formation.py::{formation_coefficients,formation_factor,formation_argument,formation_profile,formation_window_half_width,expand_escape_pieces}`; CUDA twins `montecarlo/spectrum/coherent_jit_kernel.py::{_formation_re,_formation_im,run_coherent_reduction_kernel}`, `montecarlo/spectrum/coherent_stream_jit_kernel.py::{_coherent_prologue_kernel,run_coherent_field_accumulation_kernel}`, `montecarlo/spectrum/coherent_grouped_jit_kernel.py::run_coherent_grouped_intensity_kernel`
- **Consumers:** the phased-field line reductions of `mc_spectrum`: the coherent route (`coherent=True`; batched `lines/_batched.py::_batched_coherent_block` and its CUDA stream, per-reflection `lines/_per_hkl.py`) and the flight-grouped incoherent reduction (a coherent sum per physical flight, `lines/_per_hkl.py`). Setup: `lines/_setup.py::_prepare_spectrum` splits their rows into escape pieces.
- **Source:** Feranchuk--Spence 2000 Eqs. (8), (10), (12)--(14) finite-time field (as derived under `coherent-emission`), with the amplitude damped by Beer--Lambert $e^{-\tau/2}$ (`self-absorption`) and phased by the in-medium escape leg $-\delta\omega L_\text{esc}$ (`xray-in-medium-propagation-phase`) *inside* the integral instead of at the segment midpoint
- **Intended quantity:** one straight segment's coherent line field under absorption and refraction, exact for any segment length

## Derivation

A segment of duration $T=t_L$ moves at constant velocity $\mathbf v$ (c = 1). With $t'$ the time from its midpoint, the coherent route already phases a point emitter by

```{math}
:label: eq-formation-phase

\Phi(t')=\omega\bigl(t-\hat{\mathbf n}\cdot\mathbf r\bigr)-\mathbf g\cdot\mathbf r-\delta(E)\,\omega\,L_\text{esc}(\mathbf r),
\qquad \mathbf r=\mathbf r_c+\mathbf v t',
```

and the escaping amplitude carries $e^{-\tau/2}$, $\tau=\mu(E_\text{res})L_\text{esc}$. The midpoint model froze both at $t'=0$ and summed only the vacuum part of $\Phi$ into a real sinc. On one linear escape piece (`segment-escape-average`) $L_\text{esc}$ is affine in $t'$, so $\Phi$ and $\tau$ are too:

```{math}
\Phi=\Phi_c+\frac{2v}{T}t',\qquad
v=\frac T2\bigl[(1-\mathbf v\cdot\hat{\mathbf n})\omega-\mathbf v\cdot\mathbf g\bigr]-\delta\omega\frac{\Delta L}{2}
 =a_\text{vac}(E-E_\text{vac})-\delta(E)\,\omega(E)\,\frac{\Delta L}{2},
\qquad
\tau=\tau_c+\frac{4q}{T}t',\quad q=\frac{\tau_\text{end}-\tau_\text{start}}4,
```

with $a_\text{vac}=(1-\mathbf v\cdot\hat{\mathbf n})T/(2\hbar c)$, $E_\text{vac}=\hbar c\,\mathbf v\cdot\mathbf g/(1-\mathbf v\cdot\hat{\mathbf n})$, $\Delta L=L_\text{end}-L_\text{start}$, and start/end ordered along travel. With $s=2t'/T$ the integrand is one exponential:

```{math}
:label: eq-formation-factor

\int_{-T/2}^{T/2}e^{i\Phi-\tau/2}\,dt'
=T\,e^{i\Phi_c}\,F,\qquad
F=e^{-\tau_c/2}\,\frac{\sinh w}{w}
=\frac{b\,e^{iv}-a\,e^{-iv}}{2w},\qquad
w=iv-q,
```

$a=e^{-\tau_\text{start}/2}$, $b=e^{-\tau_\text{end}/2}$. The midpoint phase $\Phi_c$ stays where the route has always applied it; $F$ replaces the real $\operatorname{sinc}(x)$ and the amplitude's midpoint $e^{-\tau_c/2}$.

**Split invariance.** The slope inside a piece is the time derivative of the same $\Phi$ that separates pieces, so the pieces of a straight flight sum to its field exactly, absorption and escape-leg phase included. The previous model was exact only for the vacuum phase and moved at first order in $\delta\omega\,\Delta L_\text{esc}$ and in $\mu\,\Delta L_\text{esc}$ (`coherent-segment-midpoint-time`).

**Parseval.** $\int|F|^2dv=\tfrac14\cdot2\pi\int_{-1}^{1}e^{-\tau(s)}ds=\pi\langle e^{-\tau}\rangle$, against $\int\operatorname{sinc}^2=\pi$. A piece's integrated coherent self-term is therefore exactly the incoherent route's segment-mean escape yield (`segment-escape-average`). Absorption only reshapes the line (damped, $q$-broadened); $|F|^2=(a^2+b^2-2ab\cos2v)/[4(v^2+q^2)]$ is even in $v$, so it never moves the centre.

### Why the vacuum centre and width

The coherent line centre is the root of $v=0$, not the bulk in-medium root $\omega=\mathbf v\cdot\mathbf g/(1-\operatorname{Re}n\,\mathbf v\cdot\hat{\mathbf n})$ of `xray-in-medium-resonance`. Writing $v$ on that bulk denominator instead would charge a refractive slope $\delta\omega(\mathbf v\cdot\hat{\mathbf n})$ inside the piece on top of the escape-path slope $\delta\omega\,\Delta L/T$ -- the medium counted twice (at normal exit, $\hat{\mathbf n}=-\hat{\mathbf z}$, the two slopes are identical) -- and would break split invariance.

For a flat exit face the escape-path root is also the physically refracted one. Along a $+z$ flight leaving through the entrance face, $L_\text{esc}=z/(-n_z)$, so $v=0$ at

```{math}
\omega\Bigl(1-\beta n_z-\frac{\beta\,\delta}{-n_z}\Bigr)=\beta g_z .
```

Snell refraction at the face conserves the tangential wavevector, so the in-medium normal component is $k_z=\omega\sqrt{n^2-n_\perp^2}\approx\omega\bigl(-n_z-\delta/(-n_z)\bigr)$: the same $\delta/(-n_z)$. The bulk root instead uses the unrefracted $k=\operatorname{Re}n\,\omega\hat{\mathbf n}$, i.e. $\delta(-n_z)$. The two agree only at normal exit. At hopg 002, 100 keV, $\theta_\text{obs}=119^\circ$ the coherent line sits at 1600.5865 eV, the bulk root at 1600.3788 eV, the vacuum root at 1600.3150 eV. The amplitudes ($\chi_g$, $U_g$, $\mu$) stay evaluated at the bulk $E_\text{res}$ -- a narrow-line freeze, as before -- and the incoherent route keeps the bulk root; see Open items.

## Limits and checks

- **Units:** $v$, $q$, $w$ dimensionless ($a_\text{vac}$ [eV$^{-1}$] × eV; $\delta\omega$ [Å$^{-1}$] × Å). $F$ dimensionless; $TF$ keeps the field's time units.
- **Transparent:** $\mu\to0$ gives $a=b=1$, $q=0$, $F=\sin v/v$; with $\Delta L=0$ as well this is the legacy sinc on the vacuum centre.
- **Uniform damping:** $q=0$ gives $F=e^{-\tau/2}\sin v/v$, the midpoint model exactly.
- **Opaque end:** $\tau_\text{end}\to\infty$ gives $F\to-a\,e^{-iv}/(2w)$, finite; $a+b\le2$ and $|w|\ge|v|$ bound $|F|\le1/|v|$, so a cutoff on $|v|$ stays a conservative tail cut.
- **Normal exit:** $\Delta L=-\hat{\mathbf n}\cdot\Delta\mathbf r$ makes the escape-path and bulk roots coincide.

## Numerics

- **Overflow-free constants.** `formation_coefficients` returns $(a+b,\;b-a,\;q)$, each bounded by 2 for $\tau\ge0$. Below $|q|<0.5$ they come from $2e^{-\tau_c/2}\cosh q$ and $-2e^{-\tau_c/2}\sinh q$, so $b-a$ does not cancel.
- **Closed form.** $\operatorname{Re}F=(v(a+b)\sin v-q(b-a)\cos v)/(2|w|^2)$ has no cancellation for $|v|<\pi$ and beyond it loses only $\epsilon$ times the $1/|v|$ envelope; $\operatorname{Im}F=-(v(b-a)\cos v+q(a+b)\sin v)/(2|w|^2)$ loses only absolute rounding ($\lesssim\epsilon$).
- **Series.** Below $|w|^2<10^{-6}$: $F=\tfrac{a+b}2\bigl(1-\tfrac{q^2}3-\tfrac{v^2}6\bigr)-i\tfrac{a+b}2\tfrac{qv}3$, with truncation $\approx q^4/8$ ($\cosh q$ is kept exactly in $a+b$ while the bracket is cut), about $10^{-13}$ at the switch.
- **Rows.** `expand_escape_pieces` splits each coherent/grouped row at its escape pieces: piece length, midpoint and start age (parent $t$ + offset$/\beta$). A flight's pieces keep its grouping key. Only single slabs reach it -- both routes refuse layers.
- **Window.** The per-hkl `sinc_cutoff` window bounds $|v|$; its energy half-width $(\text{cutoff}+|\Delta L/2|\max\delta\omega)/a_\text{vac}$ covers the refractive shift.
- **CUDA.** Formation mode is an all-or-none launch-uniform branch; omitted, every kernel evaluates the legacy sinc unchanged. All seven kernels transpile and compile to PTX (NVRTC, compute_80) on a host without a GPU; device execution passed on the RTX 5080 (2026-09-26).

## Evidence

- `tests/montecarlo/test_coherent_formation_absorption.py`: $F$ against 400k-point quadrature for seven $(v,\tau_\text{start},\tau_\text{end})$ cases (series branch, uniform damping, into and out of the crystal, opaque) to $10^{-9}$; $\mu\to0$ is the sinc to $10^{-15}$; Parseval $\int|F|^2dv=\pi\langle e^{-\tau}\rangle$ to $10^{-6}$ with the analytic tail; $|F|\le1/|v|$; the coherent `mc_spectrum` of one straight flight is invariant under 1/4/16-way splitting on the batched slab, the batched finite box whose exit face switches and the per-reflection route (strongly absorbed: coherent/transparent yield < 0.8), to $10^{-9}$ of peak; one absorbed segment's integrated coherent yield equals the incoherent route's to $2\times10^{-3}$ (mostly the $O(\delta)$ Jacobian difference $dv/dE=a_\text{vac}+(\Delta L/2)\delta\omega/E$ against the bulk width, predicted $1-2.1\times10^{-4}$; the rest grid truncation).
- `tests/montecarlo/test_coherent_emission.py::test_straight_flight_is_invariant_to_two_half_segments`: the two-half residual is now $1.8\times10^{-15}$ of peak (was $7.8\times10^{-6}$ at the 40 Å flight); `::test_single_segment_coherent_equals_incoherent_self_term`: integrated yields agree to $10^{-6}$.
- `tests/montecarlo/test_xray_dispersion.py::test_single_segment_coherent_line_sits_on_the_escape_path_root`: the absorbed single-segment coherent peak lands on the closed-form escape-path root to within two 2.5e-5 eV grid steps, more than 1000 steps from the bulk root.
- `tests/montecarlo/test_substep_invariance.py::test_incoherent_cxr_converges_under_substep_refinement` (flight-grouped): finest-rung grid L1 $6.5\times10^{-6}$ and peak error $5.6\times10^{-8}$ (were $2.1\times10^{-4}$ and $2.7\times10^{-5}$).
- `tests/montecarlo/test_coherent_formation_absorption_cuda.py` (CUDA-gated): the reduction kernel (1/2/3 energies per block), the stream field and grouped kernels against the float64 NumPy reference, with and without the cutoff; the prologue's formation outputs against its own legacy outputs (coefficients lose exactly $e^{-\mu L_\text{mid}/2}$, one $\mu$ across both ends, vacuum centre and width).
- Identity: `LINE_ESCAPE_MODEL = "segment-mean-v2-coherent-formation"` forks line spectra; profile digests re-minted.

## Assumptions and scope

- Constant velocity, energy and amplitude over a segment (the finite-time line model); $\mu$ and $\delta$ constant across the narrow line, evaluated at $E_\text{res}$ and on the output grid respectively, as before.
- Escape is straight, single-slab, no re-entry; $L_\text{esc}$ affine on each piece (the `segment-escape-average` cut set). Layered stacks stay refused on the coherent route.
- Bulk response only -- no interface/Fresnel amplitude; the refraction enters as the escape-leg phase, as in `xray-in-medium-propagation-phase`.

## Open items

- The incoherent route and every amplitude still use the bulk root $1-\operatorname{Re}n\,\mathbf v\cdot\hat{\mathbf n}$, which for a flat exit face is the unrefracted wavevector. Coherent and incoherent line centres therefore differ by $O(\delta)$ (0.21 eV of 1600 eV above). Moving `xray-in-medium-resonance` and the incoherent route to the escape-path root is tracked as issue #187, not changed here.
- `expand_escape_pieces` advances piece ages with a float64 $\beta$ while the device $t_L$ uses REAL $\beta$: a rounding-level clock mismatch on float32 backends.

## Fresh-context verification (2026-09-26)

A separate context that did not write the implementation verified the claim. It read only the ledger row and this record's claim, derived the result below, and only then read `lines/_formation.py`, `lines/_setup.py`, `lines/_batched.py`, `lines/_per_hkl.py` and the three CUDA modules.

**Independent derivation.** A point emitter at $\mathbf r(t')=\mathbf r_c+\mathbf v t'$ reaches the far field along $\hat{\mathbf n}$ with the optical path $R-\hat{\mathbf n}\cdot\mathbf r+(n-1)L_\text{esc}(\mathbf r)$. With $n=1-\delta$ its Fourier phase is therefore $\omega(t-\hat{\mathbf n}\cdot\mathbf r)-\mathbf g\cdot\mathbf r-\delta\omega L_\text{esc}$. The refractive term has the sign of the vacuum retardation, because both come from one optical path. The amplitude is damped by $e^{-\mu L_\text{esc}/2}$. On a piece where $L_\text{esc}$ is affine, $L_\text{esc}=L_c+(\Delta L/T)t'$ with $\Delta L=L(t'=T/2)-L(t'=-T/2)$, i.e. end minus start along travel. Put $s=2t'/T$. Then

$$
\frac{d\Phi}{ds}=\frac T2\bigl[\omega(1-\mathbf v\cdot\hat{\mathbf n})-\mathbf v\cdot\mathbf g\bigr]-\delta\omega\frac{\Delta L}{2}\equiv v,
\qquad
\frac{\tau}{2}=\frac{\tau_c}{2}+\frac{\Delta\tau}{4}s\equiv\frac{\tau_c}{2}+qs,
$$

and

$$
\int_{-T/2}^{T/2}e^{i\Phi-\tau/2}dt'
=\frac T2e^{i\Phi_c-\tau_c/2}\int_{-1}^{1}e^{(iv-q)s}ds
=T\,e^{i\Phi_c}\,e^{-\tau_c/2}\frac{\sinh w}{w},\qquad w=iv-q .
$$

Since $\tau_c/2\pm q=\tau_\text{end/start}/2$, $e^{-\tau_c/2}e^{\pm w}$ gives $b\,e^{iv}$ and $a\,e^{-iv}$ with the stated $a$, $b$. Writing $\omega=E/\hbar c$ gives $v=a_\text{vac}(E-E_\text{vac})-\delta\omega\Delta L/2$, with the stated $a_\text{vac}$ and $E_\text{vac}$. All of this matches the claim term for term, including the start/end ordering and the minus sign of the refractive term. Parseval with $F=\tfrac12\int_{-1}^1e^{-\tau(s)/2}e^{ivs}ds$ gives $\int|F|^2dv=\tfrac14\cdot2\pi\int_{-1}^1e^{-\tau}ds=\pi\langle e^{-\tau}\rangle$. The limits $\mu\to0$, $q=0$, the opaque end and $|F|\le(a+b)/(2|w|)\le1/|v|$ (for $\tau\ge0$) follow directly. The series $e^{-\tau_c/2}[1+(q^2-v^2)/6-iqv/3]$ equals the implemented $\tfrac{a+b}2(1-q^2/3-v^2/6)-i\tfrac{a+b}2\tfrac{qv}3$ to second order. The closed-form real and imaginary parts follow from $N\bar w/|w|^2$ with $N=(b-a)\cos v+i(a+b)\sin v$.

**Vacuum centre (item b).** For a planar exit face with outward normal $\hat{\mathbf e}$, $L_\text{esc}=(f-\hat{\mathbf e}\cdot\mathbf r)/(\hat{\mathbf e}\cdot\hat{\mathbf n})$. The spatial phase is then $-\mathbf k_\text{eff}\cdot\mathbf r$ with $\mathbf k_\text{eff}=\omega\hat{\mathbf n}-\delta\omega\,\hat{\mathbf e}/(\hat{\mathbf e}\cdot\hat{\mathbf n})$. This conserves the tangential component and gives the normal component $\omega(\hat{\mathbf e}\cdot\hat{\mathbf n})-\delta\omega/(\hat{\mathbf e}\cdot\hat{\mathbf n})$, which is the first-order expansion of Snell's $\omega\sqrt{n^2-n_\parallel^2}$. So the escape-path root is the refracted-wave root for every planar face, including the finite box's side faces. The bulk root instead uses $\mathbf k=\operatorname{Re}n\,\omega\hat{\mathbf n}$. That wave does not satisfy tangential continuity, so it would leave the crystal in a different direction. Adding a bulk-denominator slope on top of $-\delta\omega\Delta L/2$ would count $\delta$ twice. Numerically, at hopg 002, 100 keV, $119^\circ$, flight along $+z$, the roots are: exact Snell $1600.5859$ eV, escape-path $1600.5858$ eV, bulk $1600.3781$ eV, vacuum $1600.3143$ eV. (These use the rounded captured $\mathbf v\cdot\mathbf g$, which is why they sit $7\times10^{-4}$ eV below the record's figures.) **Adjudication:** the vacuum centre with the escape-path slope is correct and is the more physical of the two roots. Freezing the amplitudes ($\chi_g$, $U_g$, $\mu$) at the bulk $E_\text{res}$ is a consistent narrow-line approximation: the shift is $1.3\times10^{-4}$ relative, far below the tabulations' energy structure away from edges. The incoherent route keeping the bulk root is not an error in *this* claim, but it is now the less accurate route. See residual risks.

**Implementation diff.** No factor, sign, unit or convention divergence.

- `_formation.py:64-81`: $(a+b,b-a,q)$ with the $|q|<0.5$ cosh/sinh branch.
- `_formation.py:84-106`: the closed form and series above.
- `_formation.py:109-118`: $v$ with $\delta\omega$ taken from `delta_omega_grid` ($=\delta(E)\,\omega(E)$, `_setup.py:460-466`).
- `lines/_per_hkl.py::_formation_lines` and `lines/_batched.py:568-576`: centre and width use the vacuum `denom_all` $=1-\mathbf v\cdot\hat{\mathbf n}$ (`_setup.py:429`), $\tau=\mu(E_\text{res})L$ with the bulk-root $\mu$, and the midpoint escape distance $(L_\text{start}+L_\text{end})/2$ for the inter-piece phase. That midpoint distance is exact because $L$ is affine (item f).
- `expand_escape_pieces`: each piece gets start age $t_\text{parent}+\text{offset}\cdot L/\beta(E_\text{repr})$ and midpoint $\mathbf r_\text{mid}+(\text{offset}+f/2-1/2)L\,\hat{\mathbf v}$. The route adds $t_L/2$ (`_setup.py:447`), so the piece-midpoint age and position are exact. Pieces are expanded after `_clip_segments_to_cutoff`, and the grouping keys are gathered with the pieces.
- Window: $|v|\le c$ implies $|E-E_\text{vac}|\le(c+|\Delta L/2|\max\delta\omega)/a_\text{vac}$, as coded (item g).

**CUDA (item h), algebraic only.**

- After normalizing the energy-slot suffixes, every formation branch is textually identical, with matching $E_k$, $\delta\omega_k$ and accumulator indices: 6 in `coherent_jit_kernel.py`, 3 in `coherent_stream_jit_kernel.py`, 2 in `coherent_grouped_jit_kernel.py`.
- `_formation_re`/`_formation_im` are the NumPy expressions, and the complex products are correct.
- The prologue saves `dnm_vac` before the in-medium fixed point overwrites `dnm`. It emits the vacuum $E_r$ and $a$, sets the transmission to 1 in the coefficients, and forms $\tau$ from $\mu$ at the converged bulk $E_\text{res}$ with the same $|q|<0.5$ branch.
- It rejects non-finite or zero `apb` and NaN `bma`, like the CPU `_formation_good`. The only difference is that $\pm\infty$ `bma` is not rejected, which is unreachable for $L,\mu\ge0$.
- Every call site passes the formation arguments, so the legacy `exp(-L_esc mu)` branch (`coherent_stream_jit_kernel.py:301`) is unreachable from `mc_spectrum`.

**Independent numerical oracles** (scratch scripts, CPU backend). The oracle integrates $\int e^{i\Phi(t)-\mu L_\text{esc}(t)/2}dt$ by brute force over the whole flight, with $4\times10^5$ trapezoid points. It uses its own escape geometry (slab $z/(-n_z)$; box $\min$ over faces), $\delta$ read directly from `refractive_index`, and $\mathbf v\cdot\mathbf g$, $\mu(E_\text{res})$ captured from the run. It is compared with `mc_spectrum(coherent=True)` as $S=K|f|^2$.

| check | result |
| --- | --- |
| slab, exit through the entrance face, batched and per-hkl routes, 1/3/16 collinear pieces | spread of $K$ over the line $1.0\times10^{-9}$; $\max\lvert S-K\lvert f\rvert^2\rvert/\max S=9.5\times10^{-10}$; peaks coincide (1600.590 eV) |
| finite box, exit face switches along the flight, both routes, 1/3/16 pieces | $2.1\times10^{-9}$ / $1.7\times10^{-9}$ |
| absolute scale: $K_\text{absorbed}/K_\text{transparent}$ (escape ends zeroed) | $1+7\times10^{-12}$ |
| flight-grouped incoherent reduction, one flight in 5 substeps | $K$ spread $1.0\times10^{-9}$, $K_\text{grouped}/K_\text{coherent}=1$ |
| per-hkl `sinc_cutoff=5`, 3 pieces, against the oracle's own per-piece quadrature masked at $\lvert v_p\rvert>5$ | $9.5\times10^{-10}$; identical support (128 bins) |
| `formation_factor` against a 40-digit $e^{-\tau_c/2}\sinh w/w$, 3090 cases including both branch switches and $\tau$ up to 800 | worst error $1.2\times10^{-13}$ |
| float32 emulation of the CUDA prologue and `_formation_re/_im` | worst error $2.7\times10^{-5}$ of $\min(1,1/\lvert v\rvert)$, at $\lvert v\rvert\sim500$–$1000$ (float32 rounding of $v$, as for the legacy sinc) |

The listed anchors pass: 44 passed, 1 skipped (CUDA).

**Observations (not discrepancies).**

- The series truncation is not the $O(|w|^4/120)$ that `_formation.py:54-57` and the Numerics section state. The implemented $\tfrac{a+b}{2}$ carries $\cosh q$ exactly while the bracket is cut, so the error is $\approx q^4/8$: measured $1.2\times10^{-13}$ at $q\approx10^{-3}$, not "below $10^{-14}$". This is harmless, but the stated bound is wrong.
- "$\operatorname{Re}F$ has no cancellation" holds for $|v|<\pi$ only. Beyond that $v\sin v$ changes sign, but the error stays at $\epsilon$ times the $1/|v|$ envelope.
- Parseval is exact in $v$, not in $E$. The coherent Jacobian is $dv/dE=a_\text{vac}+(\Delta L/2)\,\delta\omega/E$, while the incoherent route uses the bulk width. The single-segment coherent-to-incoherent integrated ratio is therefore $1-O(\delta)$: predicted $1-2.1\times10^{-4}$ and measured $0.99977$ on a 200 eV grid. That is most of what the Evidence section attributes to "grid truncation".
- The groove (piecewise escape with jumps) is covered by the cut set and the code path, but not by this oracle.

**Verdict:** `rederived`. CUDA hardware execution remains pending; the kernels match the reference algebraically only.

**Residual risks.**

1. The coherent and incoherent line centres differ by $O(\delta)$: 0.21 eV of 1600 eV here. The coherent one is the Snell-consistent one, so any coherent-versus-incoherent comparison (for example `emission="both"`) carries this offset. `xray-in-medium-resonance` should consider adopting the escape-path root.
2. The first-order $\delta/(-n_z)$ fails as $n_z^2\to2\delta$ (grazing exit, $\approx1.1^\circ$ here), which is already out of scope.
3. The float64 $\beta$ in `expand_escape_pieces` against the REAL $t_L$ is a rounding-level clock mismatch on float32.

## Status

`rederived` (fresh-context verification above, 2026-09-26). CUDA hardware run of the gated parity tests passed (2026-09-26); the before/after split ladder is recorded under `segment-escape-average`. Human sign-off pending.
