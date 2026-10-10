# Independent validation: finite transverse crystal

- **Claim id:** `finite-transverse-crystal`
- **Code:** `montecarlo/geometry.py::first_prism_exit`; `montecarlo/transport/api.py::simulate_trajectories`; `montecarlo/spectrum/lines/_spectrum.py::mc_spectrum`; and `montecarlo/spectrum/brem.py::mc_brem_spectrum`
- **Source:** rectangular-prism ray intersection and Beer–Lambert attenuation
- **Quantity:** For the sample-frame prism $[-W/2,W/2] \times [-H/2,H/2] \times [z_{\min},z_{\max}]$, the geometry helper returns the first forward ray-boundary intersection for an origin inside the prism (0 on the exit face; a negative signed distance past it, which escape callers clip to zero). Electron transport uses its own strictly positive helper, so a free flight never takes a zero step. Transport stops there, and each segment's radiation receives its Beer–Lambert escape factor along the fixed photon ray to the same boundary. Public widths and heights are in mm; geometry uses Å.
- **Assumptions:** escape origins lie inside the prism or on its boundary (origins past an exit face have no in-crystal path); directions are in the sample frame; a finite footprint has two positive dimensions; and the observation direction is fixed in the far field. With both transverse dimensions omitted, the model is the laterally infinite $z$-only slab.

## Independent re-derivation (before implementation inspection)

Let $\mathbf r=(x,y,z)$ and $\mathbf d=(d_x,d_y,d_z)$. For coordinate $q$, with lower and upper faces $q_-$ and $q_+$, the forward candidate is

$$
t_q = \begin{cases}
 (q_+-r_q)/d_q,&d_q>0,\\
 (q_--r_q)/d_q,&d_q<0,\\
 +\infty,&d_q=0.
\end{cases}
$$

Thus, for a finite prism,

$$
t_{\mathrm{exit}}=\min(t_x,t_y,t_z),\qquad
\mathbf r_{\mathrm{exit}}=\mathbf r+t_{\mathrm{exit}}\mathbf d.
$$

All finite candidates are positive for a strictly interior origin. Equal candidates describe an edge or corner. A fixed face order gives a deterministic face identity without changing the distance; the convention is the face with the lowest constant. If both transverse dimensions are omitted, only the two $z$ faces remain. Thus a lateral ray ($d_z=0$) exits through a transverse face when the footprint is finite.

The public conversion is

$$
W_{\mathrm{\mathring A}}=10^7 W_{\mathrm{mm}},\qquad
H_{\mathrm{\mathring A}}=10^7 H_{\mathrm{mm}},
$$

because $1\ \mathrm{mm}=10^7\ \mathrm{\mathring A}$. Convert once before the ray calculation.

For a photon emitted at $\mathbf r_0$ in direction $\mathbf n$, use the same expression with $\mathbf d=\mathbf n$ to obtain the capped path $T=t_{\mathrm{exit}}$. With piecewise-constant attenuation coefficient $\mu_j\,[\mathrm{\mathring A}^{-1}]$ in layer $j$, Beer–Lambert attenuation is

$$
A=\exp(-\tau),\qquad
\tau=\int_0^T \mu(z_0+s n_z)\,ds
     =\sum_j \mu_j L_j,
$$

where $L_j$ is the portion of $[0,T]$ that lies in layer $j$. For $n_z\ne0$, the layer-boundary parameters are $s_i=(Z_i-z_0)/n_z$; intersect each resulting interval with $[0,T]$ and sum their lengths. This includes every crossed layer and clips the last interval at the prism exit. For $n_z=0$, $z(s)=z_0$ and $\tau=\mu_{\mathrm{current}}T$: lateral escape stays in the emitting layer, without division by $n_z$.

Electron free flights follow the same minimum-distance rule. A Gaussian-beam entry outside the footprint has no material path or segments, but remains an incident trial. A spectrum is therefore normalized by `Ne`, not by the number of entered or radiating electrons. All misses yield typed empty segments and a finite zero spectrum. With both dimensions `None`, transverse coordinates do not affect transport or attenuation, so the preceding $z$-only result is bit-for-bit recoverable.

## Implementation comparison

`first_prism_exit` implements the six candidates above, nominating per axis only the exit-side face the direction component points toward, replaces zero-direction candidates with infinity, and takes their minimum. For an origin inside the prism every exit-side candidate is $\ge0$ and every other one $\le0$, so this is the forward minimum; an origin on its exit face returns 0. An origin outside the prism, past an exit face, returns that negative signed distance, which escape callers clip to zero. The compiled CPU loop and the array (CUDA) path share this rule for every origin (#298). Its face order is `X_MIN`, `X_MAX`, `Y_MIN`, `Y_MAX`, `Z_MIN`, `Z_MAX`; `argmin` therefore realizes the specified tie convention. The all-`None` branch constructs only the two $z$ candidates.

`simulate_trajectories` validates the public pair, converts each dimension as `mm * 1e7` once, and uses the common exit helper for each finite free flight. Missed entries are excluded from `alive` but the returned `Ne` is the supplied incident count. The line spectrum returns its accumulated result divided by `Ne`; the bremsstrahlung spectrum does the same (and its `1/(4 pi)` factor is the stated isotropic solid-angle convention, not a changed normalization).

Both spectrum paths obtain finite `L_esc` from the same helper. For a single material, $\tau=L_{\mathrm{esc}}\mu(E)$. Layered paths use the interval expression above: `_layer_path_length` handles `n_z == 0` in the current layer and otherwise clips each layer interval at `L_esc`. A lateral exit therefore cannot add a deeper layer; a $z$-facing ray includes every layer before its cap.

### Independent numerical checks

All distances below are Å ray parameters; supplied directions are unit vectors unless noted.

* For `W=10`, `H=20`, `z in [0,10]`, the independent face distances for origins at `(0,0,5)` are `5` to `+x`, `5` to `-x`, and `10` to `+y`; a `(1,1,0)` corner ray returns `5, X_MAX`, matching the lower face constant at the tie. The z-only probe `(0,0,5)+(0,0,-2)t` returns `t=2.5, Z_MIN`.
* Public widths `1.25e-6 mm` and `2.5e-6 mm` produced stored widths `12.500000000000002 Ang` and `25.000000000000004 Ang`, respectively (ordinary binary rounding around the expected 12.5 and 25 Ang).
* A lateral path at `z=2` with cap 3 has layer lengths `(3,0)` for layers `[0,4]` and `[4,10]`; a `n_z=0.5` ray from the same depth capped at 10 has `(4,6)`, exactly the independently integrated z crossing.
* A deliberately all-missed 32-electron finite-beam run returned `Ne=32`, `n_missed=32`, typed `(0,3)` segment positions, and a finite bremsstrahlung spectrum `[0,0]`; no entered-trajectory renormalization is possible.

### Anchor evidence

The following focused selections passed:

* `tests/montecarlo/test_finite_transverse_geometry.py`, `tests/montecarlo/test_spectrum_escape_helpers.py`, `tests/montecarlo/test_multilayer.py`, and `tests/montecarlo/test_chunk_invariance.py` with the finite/prism/layer selection: 22 passed.
* `tests/montecarlo/test_montecarlo.py` finite-footprint/all-missed/omitted-footprint selection: 8 passed.
* `tests/scan/test_sweep.py -k footprint`: 8 passed; and `tests/scan/test_run.py -k finite_footprint`: 3 passed.

- **Filters**: units **pass**; limits **pass** (all-`None` legacy recovery, misses, lateral residence, and capped z-layer crossings); signs/conventions **pass** (forward minimum over exit-side faces, 0 on the exit face (#298), parallel-face exclusion, deterministic lowest-constant ties, and `exp(-tau)`).
- **Re-derivation**: **matches** — no divergent factor, sign, exponent, unit, or tie convention found.
- **Verdict**: **rederived**.
- **Write-up**: `docs/validation/geometry/finite-transverse-crystal.md`.
- **Suggested ledger change**: change status from `unverified` to `rederived` and append this independent write-up plus the passing-anchor evidence; a human may subsequently assess the `anchored`/`signed-off` transitions.
