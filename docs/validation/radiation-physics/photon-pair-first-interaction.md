# First interaction and pair conversion of coupled hard photons

`Validation: photon-pair-first-interaction` — independent re-derivation for the
[ledger claim](../ledger-transport-background.md#photon-pair-first-interaction).

Verifier: a fresh context that did not write the implementation. Reviewed at
`e4603d98` on branch `issue-275-pair-daughters`.

## Source, scope, and assumptions

Sources:

- [PENELOPE-2024, NEA/MBDAV/R(2024)1](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf),
  §1.4.2 (Eqs. 1.106–1.111, mean free path and the exponential free-path PDF),
  §1.4.3–1.4.4 (Eq. 1.119 point probabilities of the interaction type,
  Eqs. 1.124–1.125 free-flight sampling), and the interface rule of §1.4.4.
  When a particle reaches an interface, it is stopped there and resumed with
  the properties of the new medium.
- EPDL2025 MF=23 partial cross sections through
  `pyrite.materials.photon_cross_sections` (`narrow-beam-total-attenuation`).

The quantity under review is one analog first-interaction step for every
coupled BremsLib hard photon with $k>2m_ec^2$. It covers the interaction
point, interaction layer, channel (pair or other) and target atom, the
clock and launch of the pair electron, and the energy bookkeeping of a
converted photon in `secondary_energy_balance`. The pair kinematics
($\epsilon$, angles) belong to `pair-production-sampling` and are used here
only through $k=E_-+E_++2m_ec^2$. Positions and clocks are in Å with $c=1$;
energies are in eV inside the photon step and in keV in the balance.

Assumptions taken from the ledger row and docstrings: there is no general
photon transport. A non-pair first interaction leaves the photon radiated.
Geometry is planar layers $0=z_0<z_1<\dots<z_K$ with an optional finite
rectangular footprint $\lvert x\rvert\le w/2$, $\lvert y\rvert\le h/2$
centred on the beam axis, and no grooves. Positrons are recorded but not
transported (#276).

## Independent construction before code inspection

### Free path through a piecewise-constant stack

Let the photon start at $\mathbf r_0$ with unit direction $\hat{\mathbf n}$,
so $\mathbf r(s)=\mathbf r_0+s\hat{\mathbf n}$ for $s\ge0$. Eq. 1.106 gives the
interaction probability per unit path as $\mu=N\sigma_T$. In layer $l$ the
narrow-beam coefficient is

$$
\mu_l(k)=\sum_i n_{i,l}\,\sigma_{{\rm tot},i}(k),\qquad
\sigma_{\rm tot}=\sigma_{\rm ph}+\sigma_{\rm coh}+\sigma_{\rm incoh}
+\sigma_{\rm pair,nuc}+\sigma_{\rm pair,el}.
$$

Because the process is Markovian, stopping at each interface and restarting
in the next medium (PENELOPE §1.4.4) is equivalent to the survival law
obtained from Eq. 1.108 with a position-dependent $\mu$:

$$
F(s)=\exp\!\left[-\int_0^s\mu(s')\,ds'\right]
=\exp\!\left[-\sum_l\mu_l\,\bigl\lvert[s_l^-,s_l^+]\cap[0,s]\bigr\rvert\right],
\qquad
p(s)=\mu(s)F(s).
$$

The geometry ends at the first face crossed, $s_{\rm exit}$, the minimum
over forward-facing planar and side faces:

$$
s_{\rm exit}=\min\Bigl\{\tfrac{z_K-z_0}{n_z}\,[n_z>0],\;
\tfrac{-z_0}{n_z}\,[n_z<0],\;
\tfrac{\pm w/2-x_0}{n_x}\,[\pm n_x>0],\;
\tfrac{\pm h/2-y_0}{n_y}\,[\pm n_y>0]\Bigr\}.
$$

A face whose normal component is zero is never nominated, and with no
footprint the side terms are absent. For $n_z\neq0$, layer $l$ holds the
ray on

$$
s_l^-=\max\!\Bigl(\min\bigl(\tfrac{z_{l-1}-z_0}{n_z},\tfrac{z_l-z_0}{n_z}\bigr),0\Bigr),\qquad
s_l^+=\min\!\Bigl(\max\bigl(\tfrac{z_{l-1}-z_0}{n_z},\tfrac{z_l-z_0}{n_z}\bigr),s_{\rm exit}\Bigr),
$$

and on the empty set when $s_l^+\le s_l^-$. For a ray going toward the
entrance face ($n_z<0$), the layers are entered in decreasing $l$; ordering
by $s_l^-$ handles both directions. For $n_z=0$, the ray stays at depth
$z_0$. It occupies exactly one layer on $[0,s_{\rm exit}]$, using the
half-open convention $z_{l-1}\le z_0<z_l$. A closed convention would count
a ray lying on an interface in both layers and double its optical depth.

With $\Lambda=\sum_l\mu_l(s_l^+-s_l^-)$ and the inverse transform of $F$, one
uniform $\xi\in[0,1)$ gives $\tau=-\ln(1-\xi)$, exponentially distributed with
unit mean. The photon:

- escapes iff $\tau\ge\Lambda$ (or $\Lambda=0$), so
  $P_{\rm esc}=e^{-\Lambda}=\exp(-\sum_l\mu_lL_l)$, which is exactly the
  escape transmission used by the hard-event scorer for the same ray;
- otherwise interacts in the first layer $l^\ast$, in ray order, whose
  cumulative depth $T_{l^\ast}=\sum_{l\preceq l^\ast}\mu_l(s_l^+-s_l^-)$
  first reaches $\tau$. That layer must have nonzero optical depth, since a
  layer the ray never enters cannot hold the point. The interaction is at
  $s=s_{l^\ast}^-+(\tau-T_{l^\ast-1})/\mu_{l^\ast}$.

For a single slab entered at normal incidence this gives the truncated
exponential $P(s\le x\mid{\rm interact})=(1-e^{-\mu x})/(1-e^{-\mu d})$.

### Channel and atom

Eq. 1.119 selects the channel by its share of the total. Conditional on an
interaction in layer $l$,

$$
P({\rm pair}\mid l)=\frac{\mu_{{\rm pair},l}}{\mu_l},\qquad
\mu_{{\rm pair},l}=\sum_in_{i,l}(\sigma_{{\rm pair,nuc},i}+\sigma_{{\rm pair,el},i}),\qquad
P(i\mid{\rm pair},l)=\frac{n_{i,l}\sigma_{{\rm pair},i}}{\mu_{{\rm pair},l}}.
$$

Both can be drawn from one uniform $\eta$ by inverting the joint discrete law
$P({\rm pair},i\mid l)=n_{i,l}\sigma_{{\rm pair},i}/\mu_l$. With
$x=\eta\mu_l$, the event is a pair iff $x<\mu_{{\rm pair},l}$, and the atom
is the $i$ with $C_{i-1}\le x<C_i$, where
$C_i=\sum_{j\le i}n_{j,l}\sigma_{{\rm pair},j}$. Conditional on $x<\mu_{\rm
pair}$, $x$ is uniform on $[0,\mu_{\rm pair})$, which gives the conditional
atom law.

### Emission point, clock and launch conventions

Under the existing hard-photon contract (`bremslib-radiative-event-spectrum`),
the photon is emitted at the row endpoint
$\mathbf r_{\rm mid}+\tfrac12L\hat{\mathbf v}$ along its stored lab-frame
direction. A shell secondary from `shell-secondary-transport` is launched from
the same point at clock $t_{\rm end}$. The transport clock is $ct$ in Å:
an electron row advances it by $L/\beta$. A photon at speed $c$ advances it by
$s$, so the pair is created at

$$
\mathbf r_{\rm pair}=\mathbf r_{\rm mid}+\tfrac12L\hat{\mathbf v}+s\hat{\mathbf n},\qquad
t_{\rm pair}=t_{\rm end}+s.
$$

The pair electron is launched at $(\mathbf r_{\rm pair},t_{\rm pair})$ with
kinetic energy $E_-$ and its sampled direction iff $E_->T_s$, the same strict
inequality as shell secondaries. It inherits its parent history and bunch
offset, and its cutoff is $T_s$.

### Energy balance with converted photons

For each history, write the telescoping identity of
`shell-secondary-transport` with every hard photon's energy $k$ in
radiated. A converted photon has $k=E_-+E_++2m_ec^2$ exactly. Replacing its
$k$ in radiated by $2m_ec^2+E_++E_-$ moves energy between terms without
changing the sum. Its $E_-$ is either deposited (if $E_-\le T_s$) or becomes
the launch energy of a new track. That track's own rows then telescope to
$E_-$ (escape + continuous + cutoff residual + its own hard debits), and its
launch energy is not counted as incident. Hence, per history,

$$
\sum E_0=E_{\rm escaped}+E_{\rm deposited}+E_{\rm binding}
+E_{\rm radiated}^{\rm unconverted}+\sum_{\rm conv}\bigl(2m_ec^2+E_+\bigr),
$$

where $E_{\rm deposited}$ now also contains $\sum_{E_-\le T_s}E_-$. An
equivalent check that does not need the new terms is this. If the pair
bookkeeping is dropped (every $k$ left in radiated), the residual of the old
balance must be exactly $-\sum_{\rm launched}E_-$ per history.

### Stream keys and the `None` limit

A photon's draws must be a function of a stable identity, (seed, parent track,
photon ordinal), and of nothing else. Then its outcome cannot depend on batch
size, processing order or core. The ordinal must be counted in the
parent's flight order over all photon rows, including sub-threshold photons
and photons on terminal cutoff rows, so that it is fixed by the parent's
transport alone. The pair electron's transport key must use a salt distinct
from the shell secondary salt. Otherwise a shell secondary with hard ordinal
$j$ and a pair electron from photon ordinal $j$ on the same parent would share
a stream. With `pair_production_model=None` no photon draw is made, no
launch is added, and shell launches, keys and order are unchanged.

### Limiting cases (derived)

| Case | Expected |
|---|---|
| $\mu L\to0$ | $P_{\rm esc}=e^{-\Lambda}\to1$ |
| $n_z=0$, laterally infinite slab | $s_{\rm exit}=\infty$, interacts in its own layer with probability 1, mean depth $1/\mu_l$ |
| $n_z=0$ from the footprint centre toward $+x$ | $P_{\rm esc}=e^{-\mu_l w/2}$ |
| origin on interface $z_l$ | the layer entered by the ray holds $[0,\cdot)$; the other has zero length; with $n_z=0$, the deeper layer (half-open) |
| origin on entrance face, $n_z<0$ | $s_{\rm exit}=0$, $\Lambda=0$, escape |
| $\xi=0$ ($\tau=0$) | interaction at $s=0$ in the first layer of nonzero depth along the ray; escape if $\Lambda=0$ |
| $\sigma_{\rm pair}=0$ | no pair, so rows and tracks are identical to the mode off |

All terms are dimensionally consistent. $\mu$ is in Å$^{-1}$ ($n$ in
Å$^{-3}$ times $\sigma$ in Å$^2$), $\mu s$ is dimensionless, and the clock
increment $s$ is in Å. The energy terms are all kinetic or rest energies in keV.

## Implementation comparison

`photon_first_interactions` (`pair_production.py:293`) evaluates $s_{\rm exit}$
with `first_prism_exit` on $[-w/2,w/2]\times[-h/2,h/2]\times[0,z_K]$.
`_layer_intervals` (`:262`) builds exactly the $s_l^\pm$ above, including the
$n_z=0$ branch with the half-open $z_{l-1}\le z_0<z_l$ test (`:274–275`).
Layers are ordered by $s_l^-$ and cumulated. `reached` (`:343`) requires
$T\ge\tau$ **and** nonzero layer depth, so a never-entered layer cannot hold
the point. The distance is $s_{l^\ast}^-+(\tau-T_{l^\ast-1})/\mu_{l^\ast}$
with $\tau=$ `-log1p(-u0)`. `_channel_coefficients` (`:281`) sums the five
EPDL channels for $\mu_l$ and the nuclear plus electron pair channels for
$\mu_{\rm pair}$. The channel and atom use one uniform as the joint inverse
CDF above, with $x=$ `u1 * mu_hit` and `searchsorted(..., side="right")`
(`:360`). The origin depth is clamped to $[0,\mathrm{nextafter}(z_K,0)]$
(`:320`). This changes nothing for interior endpoints and keeps a ray at
$z=z_K$ inside the half-open deepest layer.

`convert_hard_photons` (`:401`) selects every row with `hard_radiative_k_eV > 0`,
which includes terminal `CUTOFF` rows. It assigns ordinals by
(track, `flight_id`, `substep_id`) **before** filtering $k>2m_ec^2$, and keys
each photon with `pair_photon_stream_key(seed, parent_track_offset + parent,
ordinal)` (`:439`) under `_PAIR_PHOTON_SALT`. The two first-interaction
uniforms and then the pair kinematics are drawn from that photon's own
Philox stream. The origin is `r_mid + 0.5 * L * v_hat` and the event clock is
`t_end + distance`, as derived.

`transport_secondary_cascade` keeps the shell harvest and its keys unchanged
(the key computation moved earlier in the loop, `secondaries.py:417`, with
identical arguments). It appends launched pair electrons after the shell
launches (`_merge_pair_launches`, `:498`) with a separate
`_PAIR_ELECTRON_STREAM_SALT`, `launch_kind = 2`, the event point, clock,
direction and $E_-$, and launches iff `electron_keV > threshold_keV`.
`secondary_energy_balance` zeroes the converted rows' photon in `radiated`
(`:672`) and adds `pair_rest_mass`, `positron` and `pair_subthreshold`
(the latter inside `deposited`). This is term for term the identity above.

### Numerical comparison (verifier's own scripts)

| Check | Result |
|---|---|
| Deterministic replay: my own $s_{\rm exit}$, intervals, inverse-transform and joint channel/atom inversion vs `photon_first_interactions` on 4000 random rays (W / SiO₂ / Si stack, 3 + 2 + 5 mm, $k\in[1.1,8]$ MeV, finite 20 × 12 mm footprint and laterally infinite) | 0 mismatches in escape/layer/pair/element; max relative distance difference $1.3\times10^{-15}$ |
| Special rays with $\xi=0$: on interface $z_1$ with $n_z=\pm1$ and $n_z=0$; at $z=0$ with $n_z<0$; on $z_2$, oblique; at $z_K$ with $n_z<0$ | $s=0$ in layers 1, 0, 1 (deeper, half-open); escape; layer 2; layer 2, as derived |
| Oblique ray toward the entrance face from $z=8$ mm through all three layers, 5 MeV, $4\times10^5$ photons | $P_{\rm esc}$: expected 0.729933, MC 0.729945 ($0.02\sigma$); per-layer occupancy 0.02374 / 0.01459 / 0.23174 expected vs 0.02340 / 0.01449 / 0.23217 |
| Pair share in SiO₂ layer; Si share of SiO₂ pairs | 0.1286 vs 0.1225 ($1.4\sigma$, $n=5794$); 0.6012 vs 0.6056 ($n=710$) |
| Lateral ray ($n_z=0$) from the centre of a 2 mm wide W footprint | $e^{-\mu w/2}=0.923989$ vs 0.924225 ($0.6\sigma$); only layer 0 is hit |
| Lateral ray exactly on the W/SiO₂ interface, infinite slab | always interacts, always in SiO₂ (deeper layer); $\langle s\rangle\mu=1.023$ for $10^3$ photons |
| Normal incidence on 2 mm W, KS against the truncated exponential | $p=0.64$; $P_{\rm esc}$ 0.85376 vs 0.85326 |
| `_channel_coefficients` total vs `_mu_total_inv_ang` (escape scorer's $\mu$), Si and W/Se, 1.03 MeV–100 MeV | agree to $2.2\times10^{-16}$ |
| Cascades (3 MeV, 10 µm Si, test fixture; scale 1e9/3e6/1e6; Ne 6–200; $T_s$ 100 and 600 keV) | per-history residual $\le9.1\times10^{-13}$ keV; residual with pair terms dropped equals $-\sum_{\rm launched}E_-$ to $9\times10^{-13}$ keV; sub-threshold pairs (1 and 4 events) close as well |
| Event geometry in cascades | $t_{\rm pair}-t_{\rm end}=\lvert\mathbf r_{\rm pair}-\mathbf r_{\rm end}\rvert$ to $1.1\times10^{-11}$ Å; $\mathbf r_{\rm pair}-\mathbf r_{\rm end}\parallel\hat{\mathbf n}$; launched track's $\mathbf r,\hat{\mathbf v},t,E$ equal the event's exactly, and its first row starts at $E_-$ |
| Clock convention | consecutive-row $\Delta t/L\in[1.0108,1.8236]$ equals the $1/\beta$ range, so the clock is $ct$ in Å |
| Per-photon replay from a fresh Philox stream keyed by my own recomputed (track, ordinal) | 0 mismatches in distance, channel and $E_-$ (16 conversions) |
| `pair_production_model=None` vs the pre-change `secondaries.py` (`e4603d98^`), 200 primaries, $T_s=20$ keV, 2 seeds | every row array is bit-identical; every `secondary_tracks` column is identical except the new additive `launch_kind` |
| Pair mode on vs off, same seed | generation-0 rows and generation-1 shell launches (energy, point, direction, parent, ordinal) identical |
| Repository tests `test_pair_production.py`, `test_shell_secondary_transport.py` | 47 passed |

### Findings (no divergent factor, sign, unit or convention)

1. **Stream-key independence is narrower than the physics page states.**
   `hard-bremsstrahlung-events.md:60` says a photon's outcome is
   "independent of batching, generation order and transport core". The key
   uses the global track id `parent_track_offset + parent`
   (`pair_production.py:439`), so it holds for generation-0 parents. For
   parents in generation 1 and later, the id depends on `Ne` and on how many
   shell and pair launches the earlier generations made in all histories. The
   same applies to pair-electron keys and, once any pair electron is launched,
   to shell-secondary keys from generation 2 on. It is still independent of
   batch size and processing order within a run. This is the caveat already
   recorded for `shell-secondary-transport`, now extended to photons. It is a
   documentation-precision finding, not a physics discrepancy.
2. **Missing `Validation: photon-pair-first-interaction` markers on
   ledger-anchored symbols.** These are `pair_production.py:248`
   (`pair_photon_stream_key`, no marker), `secondaries.py:363`
   (`transport_secondary_cascade` lists only `shell-secondary-transport`)
   and `secondaries.py:498` (`_merge_pair_launches`, no marker). This is
   bookkeeping only.
3. Minor robustness, no effect in the supported range: a NaN $\mu$ (photon
   energy outside EPDL's 1 eV–100 GeV) makes every comparison at `:343`
   false, and the photon is counted as `escaped` instead of raising. The
   escape scorer raises in that case. It cannot occur for beams below
   100 GeV.

**Verdict: rederived.** The first-interaction law, interface handling
(including rays toward the entrance face, rays on an interface and rays
parallel to the layers), side-face capping, channel and atom selection,
emission point and clock, launch rule, energy identity and `None` limit
all match the independent construction. Only a human may set
`signed-off`.

## Outstanding

- Fix the precision of the RNG-independence sentence (finding 1) and add the
  missing markers (finding 2). Neither changes physics.
- Conversions in the cascades above all came from generation-0 parents. The
  generation-$\ge1$ path (photons from launched daughters) is exercised by the
  same code but was not populated in the verifier's runs.
- No `other`-channel first interactions occurred in the boosted-cross-section
  cascades. The non-pair branch is covered by the stand-alone statistical
  checks above.

## Resolution of findings

Addressed after this validation on branch `issue-275-pair-daughters`: the
physics page and `pair_photon_stream_key` now state that keys from
generation-1 and later parents depend on `Ne` and on earlier launches through
global track numbering (independent of batch size and order), as for
`shell-secondary-transport`; `Validation: photon-pair-first-interaction`
markers were added to `pair_photon_stream_key`,
`transport_secondary_cascade` and `_merge_pair_launches`; a NaN EPDL
coefficient (photon outside 1 eV–100 GeV) now raises instead of counting as
an escape (`test_photon_outside_epdl_domain_is_refused`).

## Anchoring (2026-10-02)

The ledger row moved from `rederived` to `anchored` after the verdict above.
`tests/montecarlo/test_pair_production.py::test_first_interaction_depth_follows_layered_optical_depth`
pins the escape probability, the piecewise optical-depth CDF and the per-layer
pair share for an oblique ray through three layers. The pins sit beside the
single-slab, backward-crossing and finite-footprint tests already listed.
`tests/montecarlo/test_pair_production_cuda.py` passed on an RTX 5080. The
pair cascade on the CUDA core is deterministic, and its generation-0 events
replay exactly through the host step. It also closes the energy balance per
history, and its pair counts agree with the per-electron core. Mode-off
outputs are bitwise equal to `main` at `0810f2ce`, apart from the additive
`launch_kind` column.
