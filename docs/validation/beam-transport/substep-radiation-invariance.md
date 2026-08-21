# `substep-radiation-invariance`

Ledger row: [`substep-radiation-invariance`](../ledger-transport-background.md#substep-radiation-invariance).
Code: `montecarlo/spectrum/lines.py::mc_spectrum` (flight grouping, `E_repr_keV`),
`montecarlo/spectrum/brem.py::mc_brem_spectrum` (`E_repr_keV`),
`montecarlo/transport/api.py::simulate_trajectories` (`E_repr_keV`),
`montecarlo/spectrum/diagnostics.py::subdivide_flights`.
Measurement: `checks/substep_invariance.py`.

## Claim

A numerical substep is integration detail, not an emitter. Splitting one
physical flight into `N` substeps therefore refines the quadrature of that
flight's emission integral and must not change what the flight radiates beyond
that refinement. Two rules make this so.

**1. The default (incoherent) CXR reduction groups by physical flight.** Rows
sharing an `(electron_id, flight_id)` key are summed as complex field and the
sum is squared; only whole flights add incoherently:

$$
I(E)=\sum_{\rm flights}
\left\lvert\sum_{k\in\mathrm{flight}}A_kQ_k\right\rvert^2.
$$

replacing the previous per-row `sum_rows |A_j Q_j|^2`.

**2. Every row is evaluated at the propagator's representative energy.** Under
`energy_model="midpoint"` transport emits

$$
E_{\rm repr}=\frac{E_{\rm start}+E_{\rm end}}{2}.
$$

the endpoint-average energy used for the transport clock. The explicit RK2
stopping update instead evaluates at `(E_start + E_pred)/2`; see
`transport-midpoint-stopping`. Both the line kernel and the
bremsstrahlung kernel read that field when it is present. Frozen rows carry no
`E_repr_keV` and keep the historical start-energy evaluation bit-for-bit.

## Why grouping, and when its finite-time factor is exact

The line kernel's per-row form factor is `Q = t_L sinc(P t_L / pi)` with `t_L`
the row's duration and `P` its resonance detuning. Under `sum |A Q|^2` a flight
split into `N` equal substeps contributes `N` terms each carrying `(t_L/N)^2`
instead of one carrying `t_L^2`, so the line peak falls roughly as `1/N`.
Tightening `max_dE_frac` would then silently dismantle the line — the failure
this row exists to close.

Grouping is not a patch but the definition of the reduction. The physical object
is the amplitude radiated by *one flight*; substeps are quadrature nodes of that
amplitude's own time integral, and the intensity is the modulus squared of the
completed integral.

At frozen energy and constant clock rate the grouped finite-time factor
recovers the unsplit row exactly in the vacuum/zero-dispersion limit. Substep
`k` of `N` has duration `t_L/N` and a phase offset
`P t_L (k - (N-1)/2) / N` about the flight's mid-time, so

$$
\sum_kQ_k
=\frac{t_L}{N}\operatorname{sinc}\!\left(\frac{Pt_L}{N\pi}\right)
\sum_k\exp\!\left[iPt_L\frac{k-(N-1)/2}{N}\right]
=\frac{2\sin(Pt_L/2)}{P}
=t_L\operatorname{sinc}\!\left(\frac{Pt_L}{\pi}\right).
$$

the Dirichlet-kernel identity, with the substep `sinc` cancelling the kernel's
denominator term for term. Under production in-medium propagation, the escape
phase varies with row position outside this finite-time factor, leaving the
first-order residual documented below.

## Why the representative energy

A row's bremsstrahlung contribution is a one-point quadrature of

$$
\int_{\rm flight} n\,\frac{d\sigma}{dk}(E(s))\,ds.
$$

Evaluated at `E_start` this is a left-endpoint rule, first order in the flight's
length; evaluated at `E_repr` it is a midpoint rule, second order. An unsplit
flight is thus already second-order accurate, which is what makes the yield
insensitive to how finely the flight was integrated. The same substitution makes
the line kernel's `t_L = L/beta(E_repr)` equal the transported flight duration
`t_end - t_start` exactly, and `t_ang + t_L/2` its exact midpoint age.

## Limiting cases

- **Lossless flight** (`dE/ds -> 0`). Every substep shares one `E` and one
  `beta`, so the Dirichlet identity makes the grouped finite-time factor exact
  in the vacuum/zero-dispersion limit. Under production in-medium propagation,
  the position-dependent escape phase is outside that factor and leaves a
  first-order refinement residual even for a lossless flight. Bremsstrahlung
  still telescopes exactly,
  `n dsigma/dk(E) * sum_k L_k = n dsigma/dk(E) L`.
- **`N = 1`.** Grouping is a no-op: with one row per flight every group is a
  singleton, the grouped sum is the incoherent sum, and the code keeps the proven
  path (`grouped` is `False` unless some `(electron_id, flight_id)` key repeats).
- **Frozen mode.** No `E_repr_keV` field and one row per flight, so both rules
  are inert and production output is bit-for-bit historical.

## What refinement is expected to do

Converge — not stand still. Holding a whole flight at one energy and one clock
evaluates its resonance condition and form factor at a single point, which
**overstates the line peak** because the resonance drifts across the flight as
`beta` falls. The measured overstatement is 2.6–5.7% on the matrix below. That
error is the same one `radiation-error-estimators` already measures per flight
and warns on; refinement is its cure.

The acceptance criterion is therefore convergence of the grouped reduction, and
the contrast against the ungrouped one on identical rows. The ungrouped column
does not converge at all.

## Measurement

`checks/substep_invariance.py`. One transport run per case
(`energy_model="midpoint"`, lockstep core, `E_cut = 1 keV`, seed 7), then
`subdivide_flights` refines the *same* physical flights along their own rays at
`f =` unrefined / 5e-3 / 1e-3 / 2e-4 with at most 256 substeps per flight.
Fixing the flights is what makes this a quadrature measurement: re-running
transport at a tighter `f` also moves the sampled collision points and
decorrelates the trajectories entirely (`energy-controlled-propagation`).

Columns. `brem` and `cxr` are the production reductions; `brem_lep` withholds
`E_repr_keV` (the historical left-endpoint evaluation) and `cxr_ungr` withholds
`flight_id` (every substep an independent emitter). L1 columns are the grid L1
against the `f = 2e-4` rung *of the same column*, so a converging reduction
drives its own column to zero. `peak` is the grouped line peak over the finest
rung's; `ungr/gr` is the ungrouped peak over the grouped peak **at the same
rung**, so the two share every row.

Grids: bremsstrahlung 200 eV – 20 keV; CXR a 1 eV grid spanning the hopg
`(0,0,2)` forward-beam resonance `E_res = hbar_c (beta v.g)/(1 - beta v.n)` at
±400 eV, with the peak read from the ±40 eV band around `E_res` (the wide
window's own maximum sits on the soft edge of the spectrum, not on the line).
CXR columns are carbon-only because the crystal must be the transported
material; the tungsten cases contribute bremsstrahlung alone.

Ne = 200 per case. `rows` is the row count at that rung, i.e. the refinement
factor actually achieved.

### C 25 keV thin (2×10³ Å)

| `f` | rows | `brem` | `brem_lep` | `cxr` | `cxr_ungr` | `peak` | `ungr/gr` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 1411 | 2.58e-03 | 4.33e-03 | 1.91e-02 | 9.67e-01 | 1.0305 | 1.0000 |
| 5e-3 | 1615 | 9.16e-04 | 2.01e-03 | 1.46e-03 | 9.54e-01 | 1.0021 | 0.7904 |
| 1e-3 | 4230 | 6.48e-05 | 3.22e-04 | 3.74e-05 | 7.82e-01 | 1.0000 | 0.4033 |
| 2e-4 | 18056 | — | — | — | — | 1.0000 | 0.1282 |

### C 25 keV thick (2×10⁴ Å)

| `f` | rows | `brem` | `brem_lep` | `cxr` | `cxr_ungr` | `peak` | `ungr/gr` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 18466 | 4.64e-04 | 2.66e-03 | 2.57e-02 | 5.79e-01 | 1.0264 | 1.0000 |
| 5e-3 | 27610 | 1.40e-04 | 1.21e-03 | 7.86e-04 | 5.63e-01 | 1.0012 | 0.8434 |
| 1e-3 | 91356 | 1.12e-05 | 2.48e-04 | 1.85e-05 | 4.54e-01 | 1.0000 | 0.5234 |
| 2e-4 | 410354 | — | — | — | — | 1.0000 | 0.2422 |

### C 100 keV thick (2×10⁵ Å)

| `f` | rows | `brem` | `brem_lep` | `cxr` | `cxr_ungr` | `peak` | `ungr/gr` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 46379 | 5.11e-04 | 1.30e-03 | 3.42e-02 | 1.55e-01 | 1.0567 | 1.0000 |
| 5e-3 | 49495 | 4.95e-04 | 1.21e-03 | 2.27e-02 | 1.45e-01 | 1.0813 | 0.9985 |
| 1e-3 | 91959 | 1.27e-04 | 3.61e-04 | 3.70e-04 | 8.84e-02 | 1.0002 | 0.7222 |
| 2e-4 | 347033 | — | — | — | — | 1.0000 | 0.4230 |

`f = 5e-3` barely refines this case (46379 → 49495 rows, a 7% row increase), so
its `peak` is not below the unrefined rung's; the first rung that actually
refines, `f = 1e-3` at 2× the rows, lands on the reference to 2e-4. Per-flight
fractional loss is smallest at 100 keV, which is exactly why a fractional-loss
cap is a weak control variable here — the point slice E's `energy-step-convergence`
row already records.

### W 25 keV thick (5×10³ Å) and W 100 keV thick (5×10⁴ Å)

Bremsstrahlung only; tungsten is not the CXR crystal.

| case | `f` | rows | `brem` | `brem_lep` |
| --- | --- | --- | --- | --- |
| W 25 keV | none | 119189 | 1.46e-04 | 6.37e-04 |
| | 5e-3 | 142539 | 1.06e-04 | 6.16e-04 |
| | 1e-3 | 338621 | 1.85e-05 | 1.85e-04 |
| | 2e-4 | 1406345 | — | — |
| W 100 keV | none | 422503 | 1.85e-04 | 2.85e-04 |
| | 5e-3 | 442157 | 1.81e-04 | 3.07e-04 |
| | 1e-3 | 613240 | 1.52e-04 | 2.57e-04 |
| | 2e-4 | 1786195 | — | — |

## Result

**CXR converges; the ungrouped reduction does not.** The grouped line spectrum
falls to a 3.7e-5 grid L1 and a 2e-5 peak error by `f = 1e-3` in both 25 keV
cases, from 1.9e-2 / 2.6e-2 unrefined, i.e. two to three orders of magnitude of
convergence over a 3–5× row increase. On identical rows the ungrouped reduction
loses 87% / 76% / 58% of the line peak at the finest rung (`ungr/gr` 0.128 /
0.242 / 0.423) and its own L1 against its own finest rung is still 0.78 / 0.45 /
0.088 at `f = 1e-3` — it is not converging to anything, it is being divided by
the substep count. `ungr/gr = 1.0000` on the unrefined rung in all three cases
confirms the two reductions coincide exactly where there is nothing to group,
which is the production default.

**Bremsstrahlung converges, and the representative energy is worth 2–10×.**
`brem` falls monotonically down every ladder, reaching 1.1e-5–1.5e-4 at
`f = 1e-3`. Withholding `E_repr_keV` (`brem_lep`) leaves an error 1.7× to 22×
larger at the same rung, largest where the flights are longest relative to their
energy (C 25 keV thick: 2.66e-3 vs 4.64e-4 unrefined, 2.48e-4 vs 1.12e-5 at
`f = 1e-3`). The ratio grows as the ladder tightens, which is the expected
signature of a first-order rule being outrun by a second-order one.

**The unrefined line peak is high by 2.6–5.7%.** That is the one-point
evaluation error of a whole flight, not a defect of the grouping; refinement
removes it. It is the same quantity `radiation-error-estimators` reports per
flight, and it sets the scale at which `max_dE_frac` must be chosen if the line
peak matters.

## Assumptions and limits

- **Host-only.** The grouped reduction is a segmented complex accumulation
  (`np.add.reduceat` over flight-boundary-snapped blocks) on the per-`hkl`
  accumulation path. A non-NumPy backend with substepped rows raises rather than
  silently falling back to the row-incoherent sum. Checklist step H completed
  without porting the grouping to the batched or device paths: the device port
  needs a segmented complex reduction with no CuPy `reduceat` (cumsum-and-
  difference was rejected on precision grounds), and the batched path falls
  back to the proven per-`hkl` loop on grouped rows. Both are performance ports
  of an already-correct path, and both fail closed, so neither is a correctness
  gap. Grooved and per-electron transport under substepping is supported; it is
  the radiation-side grouping that stays host-only.
- `components=True` raises on substepped rows: the per-component decomposition
  is defined per row and has no grouped form yet.
- `xray_dispersion="refractive"` with `layers` raises on substepped rows, for
  the same reason the coherent path does — the intra-flight coherent sum needs a
  per-layer dispersive propagation phase that is not modelled.
- The globally coherent reduction (`coherent=True`) is unaffected: it already
  sums one complex field over all rows, so substeps are already coherent within
  it. Its own ill-conditioning under any per-row change is a separate defect
  recorded under `energy-step-convergence`.
- Ungrooved single-layer slab, lockstep core, `energy_model="midpoint"`.
- The residual after refinement is bounded by the kernel's own per-row
  escape-factor quadrature, which is unchanged here and is not a substep effect.
- Energy-loss straggling is modeled optionally. This row's radiation-grouping
  derivation remains conditional on a fixed set of realized transport rows;
  distributional substep invariance of the Urban loss itself is validated by
  `energy-loss-straggling`.

## Independent verification

Fresh-context rederivation (2026-08-18, verifier context separate from the
implementation). Filters: units `pass` -- $t_L$ carries Å (c = 1) and $P$
carries $1/\text{Å}$, so $Q = t_L\operatorname{sinc}(Pt_L/\pi)$ has units of
Å and $\lvert Q\rvert^2$ of $\AA^2$, matching the kernel's own $t_L^2\operatorname{sinc}^2$
prefactor; limits `failure` for the current general claim ($N=1$ is a grouping
no-op, but a lossless dispersive flight retains the escape-phase refinement
residual); signs/conventions
`pass` once `mc_spectrum`'s own convention is used for $P$ (below) -- reading
the ledger's shorthand "$Q=t_L\operatorname{sinc}(Pt_L/\pi)$" with the bare
detuning $D=(1-\beta\hat v\cdot\hat n)(\omega-\omega_{\rm res})$ in place of
$P$ does not reproduce the code numerically; it does once $P=D/2$, exactly as
`mc_spectrum`'s docstring defines it.

**Rule (a), re-derived before reading `lines.py`.** The physical amplitude
radiated by one flight is $\int_0^{t_L} e^{iDt}\,dt = (e^{iDt_L}-1)/(iD)$, with
magnitude $2\sin(Dt_L/2)/D$. Writing $P=D/2$ this is $\sin(Pt_L)/P$, and since
$t_L\operatorname{sinc}(Pt_L/\pi) = t_L\cdot\sin(\pi\cdot Pt_L/\pi)/(\pi\cdot
Pt_L/\pi) = \sin(Pt_L)/P$ under NumPy's normalized $\operatorname{sinc}$, this
is exactly $Q$. Splitting $[0,t_L]$ at frozen $D$ into any partition (equal or
not) and summing each sub-interval's own closed-form integral reproduces the
whole integral exactly, by linearity of a definite integral over a partition
-- a general fact, not special to equal substeps. Specializing to $N$ equal
substeps with midpoint-referenced phase offsets $D t_L(k-(N-1)/2)/N$
reproduces the symmetric Dirichlet-kernel sum

$$
\sum_{k=0}^{N-1} e^{i\frac{Dt_L}{N}\left(k-\frac{N-1}{2}\right)}
=\frac{\sin(Dt_L/2)}{\sin(Dt_L/(2N))},
$$

which multiplied by each substep's own weight $(t_L/N)\sin(Dt_L/(2N))/(Dt_L/(2N))$
gives $2\sin(Dt_L/2)/D=Q$ for every $N$ -- the same value as the general
argument, and term-for-term the identity the ledger's Checks row states. Both
derivations were done before reading `lines.py`.

**Rule (b), re-derived before reading the transport cores.** Evaluating each row's
one-point emission integral at $E_{\rm repr}=(E_{\rm start}+E_{\rm end})/2$
turns a left-endpoint quadrature into a midpoint one (second order in the
flight's length, matching `transport-midpoint-stopping`'s own RK2 accuracy
argument). The stated identity $t_L=L/\beta(E_{\rm repr})=t_{\rm end}-t_{\rm
start}$ is exact only if the SAME $\beta(E_{\rm repr})$ used by the clock is
also what the radiation kernel reads for $t_L$ -- not merely close.

Read the implementation after both derivations. `lines.py` builds each row's
complex field from its own resonance $E_{r,j}$, escape phase, and a per-row
midpoint time `seg_t_mid = seg_t + 0.5*t_L_all` rather than a shared
left-endpoint phase. The finite-time sinc factor integrates the row's
detuning phase exactly, but the separate in-medium escape phase is sampled
at that midpoint and varies along a physical flight. Consequently the
flight-grouped production sum is a first-order convergent quadrature, not an
exact partition identity; exact subdivision survives only when the escape
phase is constant, such as vacuum or zero dispersion. The line-kernel
docstring defines $P=(1-\beta\hat v\cdot\hat n)(\omega-\omega_{\rm res})/2$ exactly and
`a_width = dnm*t_L/(2*HBARC_EV_ANG)` makes the `xp.sinc`
argument $a_{\rm width}(E-E_r)/\pi = Pt_L/\pi$ term for term, confirming the
convention above rather than the bare-detuning reading. `_transport_core_ungrooved`
in `montecarlo/transport/cores.py` computes
`beta_j = beta_from_keV_scalar(0.5*(E_j + E_end_j))`,
`t_end_j = clock[e] + step_j/beta_j`, and `seg_len[nseg] =
step_j` -- the SAME `step_j` and the SAME midpoint `beta_j` feed both the
transported clock and the row's `L_ang`, and `E_repr_keV = 0.5*(E_seg +
seg_E_end)` is read back by `lines.py` as `seg_E`
for `beta_all = beta_from_keV(seg_E)`. So `t_L = L_ang/beta(E_repr)` and
`t_end - t_start` share every input bit-for-bit; rule (b) is confirmed, not
merely close.

**Grouping key and the slice-H regression.** `lines.py:1019-1030` groups by
`(elec_id, flight_id)` VALUE via `np.lexsort` + a sorted-order boundary scan,
not by adjacency in emission order, so a step-major lockstep trace (one
flight's rows separated by every other electron's rows for that step) still
groups correctly. `tests/montecarlo/test_substep_invariance.py::test_transport_substeps_reach_the_grouped_reduction`
pins exactly this case for both `lockstep` and `per-electron` cores, asserting
for `lockstep` that every adjacent row pair's `(elec_id, flight_id)` differs
(`test_substep_invariance.py:257-262`) -- the row order an adjacency-keyed
grouping would silently miss -- and that the grouped reduction still equals
an independent per-flight coherent sum. The regression the slice cites cannot
recur without this test failing first.

**Fail-closed guards.** Read and confirmed as hard raises with no fallback
branch: non-NumPy backend (`lines.py:1034-1039`, `"host-only"`,
`components=True`) (`lines.py:1048-1053`) — both pinned by
`test_substepped_rows_fail_closed_on_unported_options[cuda|components]`; and
`xray_dispersion="refractive"` with `layers` under a grouped/substepped call
(`lines.py:1040-1047`, `NotImplementedError`). This last guard has no
regression test of its own -- only its `coherent=True` analogue is pinned
(`tests/montecarlo/test_xray_dispersion.py::test_refractive_coherent_through_a_layer_stack_is_refused`).
Anchor gap, not a physics finding; flagged for the ledger's `Checks`/`Anchor`
row rather than fixed here.

**Framing.** The write-up's own "Result" and "What refinement is expected to
do" sections state the claim honestly: the unrefined line peak's 2.6-5.7%
overstatement (Ne=200 matrix) is a one-point quadrature error that
refinement removes, not evidence the unrefined answer is already correct.
Confirmed, not refuted.

**Numeric corroboration (independent run, today).**
`uv run pyrite-dev test tests/montecarlo/test_substep_invariance.py` -- 11
passed. `uv run python checks/substep_invariance.py --quick` (Ne=40, smaller
statistics than the ledger's Ne=200 table) reproduces the same qualitative
signature: `cxr`/`brem` L1-vs-finest-rung fall to 0 by `f=2e-4` in every case,
`cxr_ungr` stays near 1.0 throughout (0.80-1.03), `ungr/gr` falls
monotonically with refinement (1.0000 -> 0.11-0.30 across the three CXR
cases), and the unrefined `peak` is high (1.14-1.17 at Ne=40 vs the ledger's
2.6-5.7% at Ne=200 -- noisier at lower Ne, not a discrepancy).

No stale wording found beyond the ledger note's already-flagged "checklist
step H" phrasing (out of scope here).

Verdict after the 2026-08-20 audit: `discrepancy`. The grouping identity remains
correct for the row finite-time factor and in the vacuum/zero-dispersion limit,
and measured production output converges. The former general exactness claim
omitted the within-flight in-medium escape-phase gradient. A lossless
dispersive-flight anchor and fresh-context review are required before returning
this row to `rederived`; `signed-off` remains a human decision.

## Addendum 2026-08-19: the vacuum-dispersion switch was removed

`xray_dispersion` no longer exists, so the guard described above as
`xray_dispersion="refractive"` with `layers` is now simply *layers* under a
grouped/substepped call: it fires unconditionally rather than only under an
opt-in model. Its `coherent=True` analogue is pinned by
`tests/montecarlo/test_xray_dispersion.py::test_coherent_through_a_layer_stack_is_refused`;
the substepped-grouped guard remains the anchor gap flagged above.

The convergence figures in this record were measured under the retired vacuum
kinematics. The in-medium escape-path phase varies WITHIN a segment in a way the
sinc finite-time factor does not carry, so refinement is now first-order rather
than near-exact. Re-measured on the committed hopg 25 keV ladder
(`none`/5e-3/1e-3/3e-4): grid L1 4.7e-2 / 1.0e-2 / 2.0e-3 / 2.1e-4 and peak
error 13% / 9.8e-3 / 5.4e-4 / 2.7e-5. Convergence itself is unaffected --
monotone down the ladder -- but reaching a given accuracy now needs a tighter
`max_dE_frac` than this record's numbers imply.
