# Independent validation: finite transverse crystal

- **Claim**: `finite-transverse-crystal` —
  `montecarlo/geometry.py::first_prism_exit`; `montecarlo/transport.py::simulate_trajectories`; `montecarlo/spectrum/lines.py::mc_spectrum`; `montecarlo/spectrum/brem.py::mc_brem_spectrum` — rectangular-prism ray intersection + Beer--Lambert.
- **Source and intended quantity**: The validated design defines the sample-frame
  prism `[-W/2,W/2] x [-H/2,H/2] x [z_min,z_max]`. The geometry helper returns
  the first strictly forward ray-boundary intersection. Transport stops at that
  boundary; radiation from a segment is multiplied by its Beer--Lambert escape
  factor along the fixed photon ray to that same first boundary. `W` and `H`
  enter public configuration in mm and the physics boundary uses Angstrom.
- **Assumptions**: origin is inside the prism; the ray direction is expressed in
  the sample frame; a finite footprint has positive paired dimensions; and the
  far-field observation direction is fixed. Omitting both transverse dimensions
  is the laterally infinite z-only slab.

## Independent re-derivation (before implementation inspection)

Let `r=(x,y,z)` and `d=(d_x,d_y,d_z)`. For each coordinate `q` with lower
and upper faces `q_-`, `q_+`, the only forward face candidate is

\[
t_q = \begin{cases}
 (q_+-r_q)/d_q,&d_q>0,\\
 (q_--r_q)/d_q,&d_q<0,\\
 +\infty,&d_q=0.
\end{cases}
\]

Thus, for a finite prism,

\[
t_{\\rm exit}=\\min(t_x,t_y,t_z),\qquad r_{\\rm exit}=r+t_{\\rm exit}d.
\]

All finite candidates are positive for a strictly interior origin. Equal
candidates are a geometrical corner/edge; they have the same distance, so a
fixed face-order tie rule gives deterministic identity without changing the
physics. The stated convention is the lowest face constant. If `W,H` are
both omitted, remove `t_x,t_y` and retain the original z-face result. In
particular, a lateral ray (`d_z=0`) has no z-face candidate and exits through
a transverse face if the footprint is finite.

The public conversion is

\[
W_{\\mathring{\\rm A}}=10^7 W_{\\rm mm},\qquad
H_{\\mathring{\\rm A}}=10^7 H_{\\rm mm},
\]

because `1 mm = 10^7 Angstrom`; it must occur once before the ray calculation.

For a photon emitted at `r_0` in direction `n`, use the same expression with
`d=n` to obtain a capped path `T=t_exit`. With piecewise-constant linear
attenuation coefficient `mu_j [Angstrom^-1]` in z layer `j`, the independent
Beer--Lambert result is

\[
A=\\exp(-\\tau),\qquad
\\tau=\\int_0^T\\mu(z_0+s n_z)\\,ds
     =\\sum_j\\mu_j L_j,
\]

where `L_j` is the length of `[0,T]` whose z coordinate lies in layer `j`.
For `n_z != 0`, layer-boundary parameters are
`s_i=(Z_i-z_0)/n_z`; intersect consecutive resulting parameter intervals with
`[0,T]` and sum their lengths. This both includes every layer crossed before
a top/bottom exit and clips the final interval at the prism exit. For
`n_z=0`, `z(s)=z_0`, so `tau=mu_current*T`: a lateral escape remains entirely
in its emitting z layer and no division by `n_z` is permitted.

Electron free-flight candidates obey the identical minimum-distance rule. A
finite Gaussian beam entry outside the footprint has no material path and
therefore no segments, but it is still an incident trial. Consequently a
per-incident spectrum is proportional to `sum(segment contributions)/Ne`,
not divided by the number of entered or radiating trajectories. With all
entries missed, the typed empty segment representation must yield a finite
zero spectrum. With both dimensions `None`, no transverse coordinate may
affect transport or attenuation, so the pre-footprint z-only result is
bit-for-bit recoverable.

## Filters and implementation comparison

### Comparison

`first_prism_exit` implements the six candidate quotients above, replaces
zero-direction candidates with infinity, discards non-positive candidates,
and takes their minimum. Its ordered face constants are `X_MIN, X_MAX, Y_MIN,
Y_MAX, Z_MIN, Z_MAX`; therefore `argmin` realizes the stated lowest-constant
corner tie convention. The all-`None` branch constructs only the two z
candidates.

`simulate_trajectories` validates the public pair, converts each dimension as
`mm * 1e7` once, and uses the common exit helper for each finite free flight.
Missed entries are excluded from `alive` but the returned `Ne` is the supplied
incident count. The line spectrum returns its accumulated result divided by
`Ne`; the bremsstrahlung spectrum does the same (and its `1/(4 pi)` factor is
the stated isotropic solid-angle convention, not a changed normalization).

Both spectrum paths obtain their finite `L_esc` from the same helper. The
single-material optical depth is `L_esc * mu(E)`. The layered paths call the
interval expression derived above: `_layer_path_length` has an explicit
`n_z == 0` current-layer branch and otherwise clips every layer interval at
`L_esc`. Thus a lateral face cannot add a deeper layer, whereas a z-facing ray
adds every layer reached before its cap.

### Independent numerical checks

All distances below are Angstrom ray parameters (the supplied directions are
unit unless noted):

* For `W=10`, `H=20`, `z in [0,10]`, the independent face distances for
  origins at `(0,0,5)` are `5` to `+x`, `5` to `-x`, and `10` to `+y`; a
  `(1,1,0)` corner ray returns `5, X_MAX`, matching the lower face constant
  at the tie. The z-only probe `(0,0,5)+(0,0,-2)t` returns `t=2.5, Z_MIN`.
* Public widths `1.25e-6 mm` and `2.5e-6 mm` produced stored widths
  `12.500000000000002 Ang` and `25.000000000000004 Ang`, respectively
  (ordinary binary rounding around the expected 12.5 and 25 Ang).
* A lateral path at `z=2` with cap 3 has layer lengths `(3,0)` for layers
  `[0,4]` and `[4,10]`; a `n_z=0.5` ray from the same depth capped at 10 has
  `(4,6)`, exactly the independently integrated z crossing.
* A deliberately all-missed 32-electron finite-beam run returned `Ne=32`,
  `n_missed=32`, typed `(0,3)` segment positions, and a finite bremsstrahlung
  spectrum `[0,0]`; no entered-trajectory renormalization is possible.

### Anchor evidence

Using the worktree virtual environment, the following all passed:

* `tests/montecarlo/test_finite_transverse_geometry.py`,
  `tests/montecarlo/test_spectrum_escape_helpers.py`, `tests/montecarlo/test_multilayer.py`, and
  `tests/montecarlo/test_chunk_invariance.py` with the finite/prism/layer selection:
  22 passed.
* `tests/montecarlo/test_montecarlo.py` finite-footprint/all-missed/omitted-footprint
  selection: 8 passed.
* `tests/scan/test_sweep.py -k footprint`: 8 passed; and
  `tests/scan/test_run.py -k finite_footprint`: 3 passed.

- **Filters**: units **pass**; limits **pass** (all-`None` legacy recovery,
  misses, lateral residence, and capped z-layer crossings); signs/conventions
  **pass** (strictly forward minimum, parallel-face exclusion, deterministic
  lowest-constant ties, and `exp(-tau)`).
- **Re-derivation**: **matches** — no divergent factor, sign, exponent, unit,
  or tie convention found.
- **Verdict**: **rederived**.
- **Write-up**: `docs/validation/finite-transverse-crystal.md`.
- **Suggested ledger change**: change status from `unverified` to `rederived`
  and append this independent write-up plus the passing-anchor evidence; a
  human may subsequently assess the `anchored`/`signed-off` transitions.
