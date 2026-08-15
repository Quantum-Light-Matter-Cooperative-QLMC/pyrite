# Multilayer stack transport — independent re-derivation

- **id**: `multilayer-stack`
- **anchor**: `montecarlo/transport.py::simulate_trajectories` (`layers=`)
- **source**: [Multilayer film-on-substrate materials](../../physics/materials/multilayer-materials.md)
  §(3) option A — CASINO-style boundary-aware multilayer transport
- **verifier**: fresh context (did not write the implementation)

Scope note: this row covers the **transport** side of a layered target. The
cross-stack Beer–Lambert escape factor is a separate claim
(`self-absorption`, `materials/attenuation.py::_stack_tau`) and is not
re-derived here; the per-flight energy rule belongs to
`transport-midpoint-stopping` and `energy-step-convergence`.

## 1. Intended quantity

A stack is an ordered list of layers, entrance first, with contiguous
boundaries $0 = z_0 < z_1 < \dots < z_N$; layer $i$ occupies
$[z_{i-1}, z_i]$ and carries its own composition. The claim is that
`simulate_trajectories(layers=…)` performs *exact* per-layer transport: at each
step the free path, stopping power, and scattering element are taken from the
layer containing the electron, flights truncate at internal boundaries without
a collision there, each emitted segment records its layer, and a one-layer
stack reproduces the single-material path bit-for-bit.

## 2. Independent derivation

### 2.1 What "exact" requires

Within one homogeneous layer the distance to the next elastic collision is
exponential with rate $\Sigma_i = \sum_e n_e \sigma_e(E)$, so its survival
function is $P(s > \ell) = e^{-\Sigma_i \ell}$. Across a boundary at
$z_1$ the correct process is the *inhomogeneous* one: the collision density
is $\Sigma(z(s))$ along the ray, so

$$
P(s > \ell)
= \exp\!\left[-\int_0^{\ell}\Sigma\big(z(s')\big)\,ds'\right].
$$

The exponential is memoryless, so this is sampled exactly by the standard
boundary-truncation algorithm:

1. sample $s = -\lambda_i \ln R$ with $\lambda_i = 1/\Sigma_i$ for the
   **current** layer $i$;
2. if $s$ exceeds the distance $s_b$ to the layer boundary along
   $\hat{\mathbf d}$, move only $s_b$, apply **no** collision, and repeat
   from step 1 in the neighbouring layer;
3. otherwise the collision occurs at $s$.

The memoryless property is what makes discarding the unused remainder of the
truncated draw unbiased: conditioned on surviving to the boundary, the residual
free path in the new medium is again exponential with the new medium's rate.
Correspondingly, all **physical** observables (backscatter, transmission,
stopping fraction, path length per electron, depth distribution) must be
invariant under an *artificial* subdivision of a homogeneous slab into
identical sub-layers, while the **segmentation** observables (segment count,
mean segment length) must not be — subdivision inserts extra truncations.

### 2.2 Boundary distance

With the electron at $(p_x,p_y,p_z)$ inside layer $[z_{\rm top}, z_{\rm bot}]$
moving along $\hat{\mathbf d}$,

$$
s_b =
\begin{cases}
(p_z - z_{\rm top})/(-d_z), & d_z < 0,\\[2pt]
(z_{\rm bot} - p_z)/d_z, & d_z > 0,\\[2pt]
\infty, & d_z = 0 .
\end{cases}
$$

A crossing is an *exit* only when the crossed face is the outer surface of the
stack ($z_{\rm top} \le 0$ or $z_{\rm bot} \ge z_N$); otherwise the
electron continues.

### 2.3 Layer lookup

Given internal boundaries $\{z_1,\dots,z_{N-1}\}$, the layer containing depth
$z$ is the number of internal boundaries $\le z$, i.e. a right-side binary
search. The convention must be paired with an unambiguous placement of an
electron sitting exactly *on* a boundary, otherwise the lookup can return the
layer it just left and the step repeats.

## 3. Comparison with the implementation

| quantity | derived above | implementation | verdict |
| --- | --- | --- | --- |
| layer lookup | right-side search over internal boundaries | `L = np.searchsorted(internal_bounds, pos[e, 2], side="right")`, short-circuited to `L = 0` for one layer | identical |
| rates/stopping source | current layer only | `J_arr`/`Z_arr`/`k_arr`/`coeff_arr`/`sr_*`/`mott_*` all indexed by `L` before the step | identical |
| free path | $-\lambda_L\ln R$, resampled after each truncation | `lam_ang = 1e8 / total_rate`; `step_j = -lam_ang * np.log(rng.random())` | identical |
| boundary distance | §2.2 | `s_boundary = (pz - z_top_L) / (-dz)` and `(z_bot_L - pz) / dz` | identical |
| no collision at a boundary | required | `if crossed_internal: … continue` before the element/angle draws | identical |
| exit vs internal crossing | outer faces only | `exit_top_j = cross_up_j and z_top_L <= 0.0`, `exit_bot_j = cross_dn_j and z_bot_L >= z_total` | identical |
| on-boundary placement | must be unambiguous | `pos[e, 2] += (±1) * EPS` with `EPS = 1e-6` Å | see below |
| segment layer tag | layer of the flight | `seg_lay[nseg] = L` | identical |
| energy loss over a truncated flight | current layer's stopping over the truncated length | `dEds` from layer `L`, evaluated after truncation | identical |

The `EPS = 1e-6` Å nudge across a crossed internal boundary is the
implementation's disambiguation of the on-boundary case. It is
$10^{-10}$ of a typical 20 keV elastic mean free path in carbon
($\lambda \approx 288$ Å) and $10^{-9}$ of the thinnest physically
meaningful layer, so it cannot perturb an observable; it is a
tie-break device, not a physical displacement. Its sign is taken from
$d_z$, so it always advances into the layer the electron is entering.

## 4. Filters

- **Units.** Layer bounds, positions, and step lengths are all Å; rates are
  cm$^{-1}$ converted by the same $10^{8}$ factor as the single-layer path.
  Pass.
- **Limits.** $N = 1$ must collapse to the single-material path; a stack of
  identical layers must be physically indistinguishable from the undivided
  slab; a heavy substrate under a light film must raise backscatter. All three
  verified numerically in §5. Pass.
- **Signs/conventions.** Entrance-first ordering with increasing $z$; the
  crossed face determines up/down; only outer faces terminate the history.
  Pass.

## 5. Numeric evidence

**One-layer reduction.** `composition=[("C", 0.1128)]` versus
`layers=[(0, 3e4, [("C", 0.1128)])]`, 20 keV, $N_e = 4000$, seed 3:
`L_ang`, `E_keV`, `t_ang`, `elec_id`, `layer`, `r_mid`, `v_hat`, and the
backscatter/transmission counters are **bit-for-bit equal**.

**Artificial subdivision invariance.** Carbon slab, 3 µm, 20 keV,
$E_{\rm cut} = 2$ keV, `mott`, 16 seeds $\times$ $2\times10^{4}$
electrons, undivided versus three identical 1 µm sub-layers:

| observable | rule `frozen` | rule `midpoint` | expectation |
| --- | --- | --- | --- |
| transmitted fraction | $-0.73\%$, $z = -2.38$ | $-0.06\%$, $z = -0.17$ | invariant |
| path length per electron | $+0.03\%$, $z = +1.20$ | $-0.02\%$, $z = -0.52$ | invariant |
| path-weighted $\langle z\rangle$ | $-0.08\%$, $z = -1.71$ | $-0.06\%$, $z = -1.05$ | invariant |
| backscatter fraction (4 seeds) | $z = +0.41$ | — | invariant |
| stopped-at-cutoff fraction (4 seeds) | $z = +0.51$ | — | invariant |
| segments per electron (4 seeds) | $+0.77\%$, $z = -8.33$ | — | **not** invariant |
| mean segment length (4 seeds) | $-0.76\%$, $z = +14.69$ | — | **not** invariant |

The segmentation observables move by equal and opposite fractions at constant
path length per electron, which is exactly the extra-truncation signature and
not a physics change. The only physical residual — a $-0.73\%$ transmission
shift — **disappears under the midpoint energy rule**, identifying it as the
known left-endpoint energy-step discretization error (`transport-midpoint-stopping`,
`energy-step-convergence`) rather than a boundary-handling bias. Under the
midpoint rule every physical observable is invariant within $1.1\sigma$.

**Layer tagging.** 100 nm C on Si, 5000 electrons: 34 710 segments tagged
layer 0 with midpoint depths in $[0.025, 999.993]$ Å and 2 320 924 tagged
layer 1 with depths in $[1000.004, 29999.985]$ Å — every segment midpoint
lies inside its labelled layer, with no straddling.

**Substrate backscatter.** 20 keV, 4 seeds $\times$ $2\times10^{4}$
electrons, 1 µm total:

| target | $\eta$ |
| --- | --- |
| C bulk (one layer) | 0.0330 |
| C 100 nm on C (two layers) | 0.0335 |
| C 100 nm on Si | 0.1090 |

The two-layer carbon control reproduces bulk carbon, and the silicon substrate
raises $\eta$ more than threefold — the physical substrate-backscatter effect
that motivated option A, bracketed below bulk Si ($\eta \approx 0.160$) by
the carbon overlayer.

## 6. Verdict

- **Claim**: `multilayer-stack` — `montecarlo/transport.py::simulate_trajectories`
  (`layers=`) — film-on-substrate transport.
- **Filters**: units **pass**; limits **pass**; signs/conventions **pass**.
- **Re-derivation**: **matches** — the boundary-truncation algorithm is the
  exact sampler for the inhomogeneous collision density by memorylessness, and
  the implementation realizes it term-for-term. No divergent factor, sign,
  exponent, unit, or convention found.
- **Verdict**: `rederived`.
- **Write-up**: `docs/validation/materials/multilayer-stack.md`.
- **Suggested ledger change**: status `unverified → rederived`; record the
  one-layer bit-for-bit reduction, the subdivision-invariance matrix, and the
  attribution of the $-0.73\%$ `frozen`-rule residual to the energy-step rule
  rather than to boundary handling. A human applies `signed-off`.
