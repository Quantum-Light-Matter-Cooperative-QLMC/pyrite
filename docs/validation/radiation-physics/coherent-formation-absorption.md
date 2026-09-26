# Validation: coherent-formation-absorption

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
- **Closed form.** $\operatorname{Re}F=(v(a+b)\sin v-q(b-a)\cos v)/(2|w|^2)$ has no cancellation; $\operatorname{Im}F=-(v(b-a)\cos v+q(a+b)\sin v)/(2|w|^2)$ loses only absolute rounding ($\lesssim\epsilon$).
- **Series.** Below $|w|^2<10^{-6}$: $F=\tfrac{a+b}2\bigl(1-\tfrac{q^2}3-\tfrac{v^2}6\bigr)-i\tfrac{a+b}2\tfrac{qv}3$, dropping $O(|w|^4/120)$.
- **Rows.** `expand_escape_pieces` splits each coherent/grouped row at its escape pieces: piece length, midpoint and start age (parent $t$ + offset$/\beta$). A flight's pieces keep its grouping key. Only single slabs reach it -- both routes refuse layers.
- **Window.** The per-hkl `sinc_cutoff` window bounds $|v|$; its energy half-width $(\text{cutoff}+|\Delta L/2|\max\delta\omega)/a_\text{vac}$ covers the refractive shift.
- **CUDA.** Formation mode is an all-or-none launch-uniform branch; omitted, every kernel evaluates the legacy sinc unchanged. All seven kernels transpile and compile to PTX (NVRTC, compute_80) on a host without a GPU; device execution is pending.

## Evidence

- `tests/montecarlo/test_coherent_formation_absorption.py`: $F$ against 400k-point quadrature for seven $(v,\tau_\text{start},\tau_\text{end})$ cases (series branch, uniform damping, into and out of the crystal, opaque) to $10^{-9}$; $\mu\to0$ is the sinc to $10^{-15}$; Parseval $\int|F|^2dv=\pi\langle e^{-\tau}\rangle$ to $10^{-6}$ with the analytic tail; $|F|\le1/|v|$; the coherent `mc_spectrum` of one straight flight is invariant under 1/4/16-way splitting on the batched slab, the batched finite box whose exit face switches and the per-reflection route (strongly absorbed: coherent/transparent yield < 0.8), to $10^{-9}$ of peak; one absorbed segment's integrated coherent yield equals the incoherent route's to $2\times10^{-3}$ (grid truncation).
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

- The incoherent route and every amplitude still use the bulk root $1-\operatorname{Re}n\,\mathbf v\cdot\hat{\mathbf n}$, which for a flat exit face is the unrefracted wavevector. Coherent and incoherent line centres therefore differ by $O(\delta)$ (0.21 eV of 1600 eV above). Whether `xray-in-medium-resonance` should move to the escape-path root is a separate physics question, not changed here.
- `expand_escape_pieces` advances piece ages with a float64 $\beta$ while the device $t_L$ uses REAL $\beta$: a rounding-level clock mismatch on float32 backends.

## Status

`unverified`. Pending: fresh-context re-derivation; CUDA hardware run of the gated parity tests; remote before/after run of `checks/segment_escape_split_ladder.py`. Human sign-off pending.
