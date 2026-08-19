# Validation: `xray-in-medium-propagation-phase`

## Claim (from the ledger row, before reading the implementation body)

`montecarlo/spectrum/lines.py::mc_spectrum` (`xray_dispersion="refractive"`,
`coherent=True`): coherent segment-to-segment propagation phase on the
in-medium wavevector. Segment `j` accumulates

```
-delta(E) omega(E) L_esc,j
```

on top of the vacuum retardation phase `omega d_j`, `d_j = t_j - n_hat.r_j`,
where `L_esc,j` is the in-crystal escape path from the emission point to the
exit face.

Cited source/equation: observation-time phase `omega(t_j + n_med L_esc,j +
L_vac,j)` (Jackson §14.65-type kernel `exp{i omega t_obs}`), closed with the
first-order far-field path split `L_esc + L_vac = R - n_hat.r_j`; the medium
leg contributes `omega(Re n - 1) L_esc = -delta omega L_esc`; and `exp(i n
omega L) = exp(i omega L) exp(-i delta omega L) exp(-beta omega L)` is claimed
to show this phase is the real partner of the Beer-Lambert amplitude
`exp(-beta omega L) = sqrt(exp(-mu L))` already applied over the *same* path.
Builds on `xray-refractive-index` (`n = sqrt(1+chi_0) ~= 1-delta-i beta`,
Maxwell dispersion `k^2=(1+chi_0)omega^2`) and `xray-in-medium-resonance`.

Docstring assumptions recorded before touching the implementation body:
real part only (`Im n` is the absorption already carried by the Beer-Lambert
`mu(E)` escape factor — folding it in again would double-count it); bulk
response only (no interface/Fresnel term); single-slab absorbers only
(refused for LAYERED absorbers, whose per-layer delta along the escape path
is not modelled); stated cross-check that normal exit (`n_hat` along the face
normal) should reduce this to the naive `k(E) n_hat.r_j` form up to a
segment-independent global phase, and should differ from that naive form by
more than a global phase off the face normal.

## Independent derivation

**Setup.** An electron trajectory inside a crystal slab radiates coherently.
Segment `j` emits at spacetime point `(t_j, r_j)`, with `r_j` measured from an
origin on/near the exit face and the outward exit direction `n_hat`
(`n_hat` fixed, single reflection/orientation, so `n_hat` is also the
direction to the far-field detector for that line). The photon must first
traverse the *remaining* crystal material along `n_hat` to the exit face — a
path of length `L_esc,j`, in a medium of complex index `n(omega) = 1 -
delta(omega) - i beta(omega)` — and then travel through vacuum to a detector
at `R n_hat`, `R -> infinity`.

**Phase from a fixed-frequency plane wave through the two legs.** For a
single Fourier component at angular frequency `omega`, the wave equation in a
homogeneous medium gives dispersion relation `k = n(omega) omega` (natural
units `c=1`, so `omega` and `k` are inverse lengths, matching `t_ang` stored
in Ang). Propagating a plane wave a distance `L` along its direction of
travel multiplies the field by `exp(i k L) = exp(i n omega L)`. The relevant
timing quantity that enters the *phase* of a monochromatic field is the
**phase**, not group, velocity, so the emission-to-observation phase is built
from `Re(n)` for the oscillatory part and `Im(n)` for the amplitude decay —
they do not mix at this order.

Writing the total oscillatory phase accumulated from emission to the far
field as `omega` times an effective time-of-flight (the standard
retarded-time / frequency-domain radiation construction, i.e. the source
term `exp[i omega t_obs]` with `t_obs` built from the optical path):

```
Phi_j = omega * ( t_j + Re(n) L_esc,j / c + L_vac,j / c )        (c=1)
      = omega * t_j + omega * (1 - delta) * L_esc,j + omega * L_vac,j
      = omega * t_j + omega * (L_esc,j + L_vac,j) - delta * omega * L_esc,j
```

i.e. exactly the vacuum-equivalent geometric phase for the *total* path
`L_esc,j + L_vac,j` (as if the whole flight were vacuum, `n->1`), plus a
correction `-delta * omega * L_esc,j` that acts on the escape leg alone.
**This already answers "on which path length the index acts": only the
in-medium leg `L_esc,j`, not the vacuum leg `L_vac,j`, and not the full
`L_esc,j + L_vac,j`.**

**Far-field expansion.** To first order in `1/R` (Fraunhofer / far-field
limit, `R = |R n_hat|` fixed, `r_j` near the origin):

```
|R n_hat - r_j| = R - n_hat . r_j + O(1/R)
```

and this total geometric length is exactly `L_esc,j + L_vac,j` (the straight
path from `r_j` to the far-field point, split at the exit face). So

```
L_esc,j + L_vac,j = R - n_hat . r_j
```

which is the path split quoted in the ledger row and matches the "far-field
retardation `omega n_hat.r_j`" already used by the vacuum kernel elsewhere in
`mc_spectrum`'s docstring (`E_j(omega) ... exp{i[omega t_abs,j - (omega
n_hat+g).r_j]}`).

**Assembling and dropping the global phase.** Substituting:

```
Phi_j = omega * t_j + omega * (R - n_hat.r_j) - delta * omega * L_esc,j
      = omega * R + [ omega * (t_j - n_hat.r_j) - delta * omega * L_esc,j ]
      = omega * R + [ omega * d_j - delta(E) * omega(E) * L_esc,j ],   d_j = t_j - n_hat.r_j
```

`omega * R` is common to every segment of a given reflection/orientation
(same fixed detector distance `R` along the same `n_hat`), so it is a
segment-independent global phase that drops out of every observable
(interference terms only depend on phase *differences* between segments, and
the squared modulus of the coherently-summed field is insensitive to an
overall constant phase). Dropping it:

```
Phi_j (mod global phase) = omega(E) d_j - delta(E) omega(E) L_esc,j,   d_j = t_j - n_hat . r_j
```

This is **exactly** the ledgered claim: the vacuum retardation phase `omega
d_j` plus `-delta(E) omega(E) L_esc,j` acting only on the escape path.

**Cross-check against the "naive" alternative the docstring rules out.**
Consider the alternative form `omega t_j - k(E)(n_hat . r_j)` with `k(E) =
n(E) omega(E)` applied to the *whole* projected distance `n_hat.r_j` (i.e.
treating the complete flight as though it happens in the medium). Expanding:

```
omega t_j - k(E)(n_hat.r_j) = omega t_j - (1-delta) omega (n_hat.r_j)
                             = [omega t_j - omega n_hat.r_j] + delta*omega*(n_hat.r_j)
                             = omega d_j + delta(E) omega(E) (n_hat.r_j)
```

Compare to the derived correct term `-delta(E) omega(E) L_esc,j`. These
coincide **iff** `L_esc,j = -(n_hat.r_j)`, i.e. iff `n_hat` is along the exit
face normal (straight perpendicular exit, where the escape distance from
depth-along-normal `z` to the face is exactly `-z = -(n_hat.r_j)` when
`r_j`'s normal component is measured along `n_hat`). For any other exit
direction the geometric escape distance to a flat face is
`L_esc,j = -(n_hat.r_j)/(n_hat . n_face)` (ray–plane intersection with the
crystal's physical face normal `n_face`, generally `!= n_hat`), which differs
from `-(n_hat.r_j)` by a segment-dependent (not merely global) factor whenever
`n_hat != n_face`. So the two forms differ by more than a global phase off
the face normal — precisely the limiting-case/counterexample the docstring
states, independently reproduced here from the geometry rather than assumed.

**Consistency with the Beer-Lambert amplitude factor already on the coherent
path.** The escape leg's *full* complex contribution to the field is
`exp(i n omega L_esc,j)` with `n = 1 - delta - i beta`:

```
exp(i n omega L_esc,j) = exp(i omega L_esc,j) * exp(-i delta omega L_esc,j) * exp(-beta omega L_esc,j)
```

- `exp(i omega L_esc,j)`: absorbed into the vacuum-equivalent geometric phase
  `omega(L_esc,j + L_vac,j) = omega(R - n_hat.r_j)` derived above (the `n->1`
  reference).
- `exp(-i delta omega L_esc,j)`: the new term under verification.
- `exp(-beta omega L_esc,j)`: field-amplitude attenuation. Using
  `mu(E) = 2 beta(E) omega(E)` (the same Henke/Chantler relation underlying
  the `absorption-length` claim, `mu = 2k beta` with `k=omega` in these
  units), `exp(-beta omega L_esc,j) = exp(-mu L_esc,j/2) =
  sqrt(exp(-mu L_esc,j)) = sqrt(T_abs)`, i.e. exactly the Beer-Lambert field
  amplitude already used to build `amp = sqrt(alpha*omega/(4 pi^2 hbar c) *
  T_abs)` on the coherent path, over the identical escape length `L_esc,j`.

So `Im(n)` and `Re(n)` are the imaginary and real parts of *one* complex
exponential taken over *one* path (`L_esc,j`); the ledgered claim uses only
`Re(n)` here because `Im(n)` is already supplied elsewhere (the Beer-Lambert
`T_abs`/`amp` factor) — applying it again in the phase term would double the
same physics rather than adding an independent effect. This reproduces, from
the same complex-`n` construction, the "real partner of the Beer-Lambert
amplitude" statement in the ledger row without assuming it.

## Filters

- **Units**: `delta` dimensionless, `omega(E)` and `L_esc,j` both inverse-
  length/length in the same `c=1`, Ang convention as the rest of `mc_spectrum`
  (`d_j`, `t_ang`), so `delta*omega*L_esc` is dimensionless (radians). Pass.
- **Limiting cases**: `xray_dispersion="vacuum"` forces `delta=0` (`n=1`)
  identically, so the added term is identically zero and the phase reduces to
  the pre-existing vacuum `omega d_j`, bit-for-bit — reproduced above by
  direct substitution `delta=0`. Pass. Normal-exit reduction to the naive
  `k(E) n_hat.r_j` form up to a global phase, and divergence off the face
  normal, both reproduced independently above from the escape-distance
  geometry. Pass.
- **Sign/convention**: off an absorption edge `delta>0` (`Re n<1`, phase
  velocity `>c`, standard X-ray regime), so the medium leg's phase lags the
  vacuum-equivalent phase by `delta*omega*L_esc>0`, i.e. the term subtracts
  from the accumulated phase relative to a purely-vacuum flight of the same
  total geometric length — matches `-delta*omega*L_esc,j` with `delta>0`.
  Convention `n=1-delta-i beta` with time factor `exp(+i omega t)` is the same
  one used by `xray-refractive-index`/`grazing-optical-constants`, so the sign
  of the `Im(n)` amplitude term (`exp(-beta omega L)`, decaying, passive
  medium) is also self-consistent. Pass.

## Comparison with the implementation (read after the derivation above)

`montecarlo/spectrum/lines.py::mc_spectrum`, `coherent=True` branch (around
the `delta_omega_grid` construction and its consumers):

```python
delta_omega_grid = (1.0 - Re(n(E_grid))) * omega_grid   # = delta(E) * omega(E)
...
arg = d[...] * omega_grid[...] - g_phase[...]
if delta_omega_grid is not None:
    arg = arg - L_esc[...] * delta_omega_grid[...]
ph = xp.exp(1j * arg)
```

matches `Phi_j = omega(E) d_j - g.r_j - delta(E) omega(E) L_esc,j` term for
term (the `-g.r_j` reciprocal-harmonic phase is a separate, already-ledgered
piece of the same complex field construction, not part of this claim). The
in-code comment block directly preceding `delta_omega_grid` (lines ~941-964
of `lines.py`) reproduces the same derivation chain independently arrived at
above (observation-time phase, far-field path split, `exp(i n omega L)`
factorization, non-double-counting with the Beer-Lambert amplitude, and the
naive-form counterexample), which is corroborating evidence, not a
substitute for the independent derivation performed before reading it.

`L_esc` fed into the `delta_omega_grid` multiplication (`L_esc[sel]` /
`L_esc[sl]`) is the *same* `L_esc` used a few lines earlier to build
`tau = L_esc * mu_i` (or the groove/layered/finite-footprint variants) and
`T_abs = xp.exp(-tau)`, which in turn builds
`amp = xp.sqrt(ALPHA_FS * om / (4*pi^2*HBARC_EV_ANG) * T_abs)` — i.e. the
implementation applies `Re(n)` and `Im(n)` (via `mu = 2*beta*omega`) over the
identical escape path, exactly as required by the `exp(i n omega L_esc)`
factorization above. No double-counting: the incoherent path never adds this
phase term at all, and the coherent path adds the phase term exactly once,
alongside (not on top of) the existing amplitude factor.

The GPU raw-kernel routes reproduce the same term:
`coherent_jit_kernel.py::run_coherent_reduction_kernel` and
`coherent_stream_jit_kernel.py::run_coherent_field_accumulation_kernel` both
take an optional `(L_esc, delta_omega)` pair and apply
`phase = phase_slope*E - g_phase - L_esc[seg]*delta_omega[E]` (see
`coherent_stream_jit_kernel.py` around lines 347-357, 464-475), i.e. the same
per-segment-scalar times per-energy-table product derived above, gated
identically to `None` under the vacuum model so that model stays bit-for-bit.

Numerically: `tests/montecarlo/test_xray_dispersion.py` (hopg 002, 100 keV,
theta_obs=119 deg) reports the two-segment relative phase
`-delta(E) omega(E)(z1-z2)/(-n_hat_z) = +1.588643 rad`, matching the derived
closed form to 5.7e-13 rad (float64 rounding) — reran locally:

```
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test \
    tests/montecarlo/test_xray_dispersion.py -k "phase or escape or coherent"
# 6 passed, 10 deselected
```

## CUDA streaming-route note (row's own implementation notes)

The ledger row's notes column states: *"...the streaming route still falls
back to the exact array path because its prologue kernel derives the
resonance from the vacuum denominator on device and has no in-medium root."*

This is now **stale**. Commit `4d40255` ("solve the in-medium resonance in
the CUDA stream prologue", same day as this verification) ports the fixed
point into `coherent_stream_jit_kernel.py::run_coherent_prologue_kernel`,
which now accepts `v_dot_n`/`n_re_tab` and iterates the same
`omega = v.g / (1 - Re n(omega) v.n_hat)` root used by the CPU/NumPy path
(`_in_medium_kinematics`). In `mc_spectrum`, the gate

```python
_use_jit_coherent_stream = (
    coherent and _USE_JIT_COHERENT_STREAM
    and getattr(xp, "__name__", "") == "cupy"
    and np.dtype(REAL) == np.dtype(np.float32)
    and sinc_cutoff is None
)
```

no longer excludes `xray_dispersion="refractive"`; when active it passes
`n_re_tab=n_re_tab_g` (`None` under vacuum) into the prologue kernel and
`L_esc=...`, `delta_omega=delta_omega_grid` into
`run_coherent_field_accumulation_kernel` unconditionally alongside the
non-streaming reduction path. `tests/montecarlo/test_xray_dispersion_cuda.py`
carries CUDA-gated tests exercising exactly this
(`test_prologue_solves_the_in_medium_resonance`,
`test_stream_field_kernel_carries_the_in_medium_phase`). So as of this
verification, **both** GPU coherent routes (`coherent_jit_kernel.py`'s
reduction kernel and `coherent_stream_jit_kernel.py`'s streaming kernel)
carry the propagation phase under `xray_dispersion="refractive"`; only the
prose in this ledger row's notes column needs updating (the physics term
itself is unaffected by which kernel launches it).

## Verdict

Independent re-derivation (Jackson-style observation-time phase, far-field
`R - n_hat.r_j` expansion, `exp(i n omega L)` factorization into phase and
Beer-Lambert amplitude, and the naive-form counterexample) matches the
ledgered claim and the implementation term-for-term:

```
Phi_j (mod global phase) = omega(E) d_j - g.r_j - delta(E) omega(E) L_esc,j
```

with the index acting **only on the in-crystal escape leg** `L_esc,j`
(confirmed the open question posed by the task: not on `L_vac,j`, not on the
full `R - n_hat.r_j`), and self-consistent with the Beer-Lambert amplitude
factor already carried over the same `L_esc,j` (same `n`, real vs. imaginary
part, no double-counting). No discrepancy found in the physics or the code.

One accuracy finding, not a physics discrepancy: the row's notes column's
statement about the CUDA streaming route falling back for `refractive` is
now outdated as of commit `4d40255`; recommend updating that sentence when
the row is edited (see Suggested ledger change below). This does not affect
the `rederived` determination, which concerns the phase formula itself.


## Addendum 2026-08-19: the vacuum-dispersion switch was removed

`xray_dispersion` no longer exists. The in-medium relation is unconditional, so
every statement above about the `"vacuum"` model as a *selectable* code path is
historical. What the removal changed, and what it did not:

- **Unchanged:** the phase formula itself, its derivation, and its agreement
  with the implementation. The `rederived` determination stands.
- **Retired evidence:** the limiting-case check "`xray_dispersion="vacuum"`
  leaves the phase expressions untouched and is bit-for-bit" is no longer
  expressible in production code. The kernel-level version survives — the JIT
  reduction and stream-field kernels keep their optional `(L_esc, delta_omega)`
  pair, and the CUDA-gated tests still pin that omitting it, or feeding
  `Re n = 1`, reproduces the vacuum arithmetic exactly.
- **Replacement evidence:** the host anchors no longer difference two runs. They
  compare the measured two-segment interference term against a CLOSED FORM built
  from the same source equations (`test_interference_phase_matches_the_in_medium_closed_form`),
  which pins the absolute phase over four depth separations rather than an
  increment between two code paths.
- **New consequence:** because the in-medium leg varies WITHIN a segment in a
  way the sinc finite-time factor does not carry, coherent subdivision
  invariance became first-order rather than exact. See the addendum on
  `coherent-segment-midpoint-time` and the re-measured convergence figures on
  `substep-radiation-invariance`.
