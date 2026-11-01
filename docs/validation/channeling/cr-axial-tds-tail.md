# `cr-axial-tds-tail`: continuum TDS output beyond the outgoing order box

**Status:** task-owner derivation (2026-11-01), awaiting fresh-context verification.
Code: `src/pyrite/montecarlo/channeling/axial_tds_continuum.py::{TailRecord,string_jump}`
and the `"tail"` branch of `axial_fiber_unraveling._jump` / `_trajectory`
(`FiberChannels.tail_q_max_inv_ang`). Builds on `cr-axial-continuum-tds-jumps`
(the exact joint `(Q, u)` draw) and `cr-axial-fiber-unraveling` (the absorbing
fibre model and its outgoing box).

## Why

A continuum string jump draws the output momentum $Q$ (relative to the carrier)
and the struck-string displacement $u$ exactly. When $\kappa+Q$ folds to an
order outside the outgoing box $[-B,B]^2$, the output used to be tallied as
`unresolved`, discarding a sample that the model already resolves. For
Si [001], 5 MeV, at $B=18$ this is about 0.05 of the incident weight at
$3\times10^4$ Å, and the remainder falls as $B^{-2}$. This claim gives that
weight its law, states when the sampled momentum may be emitted as a terminal
record, and keeps `unresolved` for the part that cannot be.

## Derivation

**Exact tail measure.** For a window state $\chi$ (unit norm, orders
$\mathbf g_p$ about the carrier $\kappa$) and one string, the joint rate density
of $(Q,u)$ is (`cr-axial-continuum-tds-jumps`)

$$
f(Q,u)=\frac{A\lambda}{(2\pi)^2}\Big|\sum_p\chi_p\,c(\mathbf t_p)\,w_{\mathbf t_p}(u)\Big|^2N(u),
\qquad \mathbf t_p=Q-\mathbf g_p,
$$

with $c(\mathbf t)=\varphi(|\mathbf t|)e^{-i\mathbf t\cdot\mathbf r}/A$ and
$w_{\mathbf t}(u)=e^{-i\mathbf t\cdot u}-e^{-a^2t^2/2}$. Averaging over $u$ with
$\mathbb E_u[w_{\mathbf t}w^*_{\mathbf t'}]=e^{-a^2|\mathbf t-\mathbf t'|^2/2}-e^{-a^2(t^2+t'^2)/2}$:

$$
\bar f(Q)=\frac{\lambda}{A(2\pi)^2}\sum_{pp'}\chi_p\chi^*_{p'}\varphi_p\varphi_{p'}
e^{i(\mathbf g_p-\mathbf g_{p'})\cdot\mathbf r}
\Big[e^{-a^2|\mathbf g_p-\mathbf g_{p'}|^2/2}-e^{-a^2(t_p^2+t_{p'}^2)/2}\Big].
\tag{1}
$$

The tail rate is $\Gamma_{\rm tail}(B)=\int_{\kappa+Q\notin\mathcal B}\bar f\,d^2Q
=\langle\chi|\Gamma|\chi\rangle-\int_{\mathcal B}\bar f$, where the absolute box is
$\{K:|\mathrm{fold}(K)|_\infty\le B\}$, i.e. $K_i\in[-(B+\tfrac12)g_i,(B+\tfrac12)g_i)$.

**The tail record is a refinement of the existing terminal outcome.** Given
$(\Delta,u)$, the output $J_{\Delta,u}\chi$ is a superposition of orders $n$;
the joint draw picks $n$ with probability $|\psi_n|^2/\lVert J\chi\rVert^2$, so
$Q=\Delta+\mathbf g_n$ is an exact momentum-basis measurement outcome of the
output. For $\kappa+Q$ outside the box, the absorbing model already terminates
the history (out-of-box effect $\sum_{o\notin\mathcal B}|o\rangle\langle o|$
applied to $J\chi$). Refining that effect into the rank-one momentum
projectors $|K\rangle\langle K|$, $K=\kappa+Q\notin\mathcal B$, and recording
$(K,u,\text{string})$ leaves every probability of the model unchanged:

$$
P(\text{retained})+P(\text{outgoing})+\underbrace{\textstyle\int_{K\notin\mathcal B,|K|\le K_c}}_{\text{tail}}+\underbrace{\textstyle\int_{K\notin\mathcal B,|K|>K_c}+(\text{lattice, kick remainders})}_{\text{unresolved}}=1 .
$$

The record is a label on an outcome the model already draws; it consumes no
random numbers, so with any cap the event depths and jump counts are bitwise
those of the uncapped run (`tail_q_max_inv_ang = None`).

**Large-$Q$ (incoherent single-atom) law.** For $|Q|\gg|\mathbf g_p|$ over the
window, $\varphi(|\mathbf t_p|)=\varphi(Q)[1+O(g_p/Q)]$ and
$e^{-a^2t_p^2/2}=e^{-a^2Q^2/2}[1+O(a^2Qg_p)]$. Inserting in (1), the sum over
$p,p'$ factorizes through the window wavefunction
$\tilde\psi(\mathbf x)=\sum_p\chi_pe^{i\mathbf g_p\cdot\mathbf x}$ (cell average of
$|\tilde\psi|^2$ is 1):

$$
\bar f(Q)\;\to\;\frac{\lambda\,|\varphi(Q)|^2}{A(2\pi)^2}\Big[\langle|\tilde\psi(\mathbf r+u)|^2\rangle_u-e^{-a^2Q^2}|\tilde\psi(\mathbf r)|^2\Big].
\tag{2}
$$

$\langle|\tilde\psi(\mathbf r+u)|^2\rangle_u/A$ is the probability density at the
thermally smeared string (the flux-peaking factor). So in the tail:

- **rate** = density at the struck nucleus × single-atom phase-shift cross
  section $\lambda|\varphi(Q)|^2/(2\pi)^2$ per unit $d^2Q$; once $a^2Q^2\gg1$ the
  thermal subtraction vanishes and the law is the incoherent single-atom
  (Rutherford-like, screened) law;
- **direction**: the azimuth of $Q$ is uniform up to $O(g/Q)$ anisotropy; the
  polar angle to the axis is $\theta=|K|/k$ with $k=p/\hbar$ (paraxial);
- **energy**: elastic (phonon energies are neglected throughout the TDS model);
  the transverse kinetic energy is $(\hbar c|K|)^2/(2E)$, $E$ the total energy;
- **displacement**: $u\mid Q\to N(u)|\tilde\psi(\mathbf r+u)|^2/\langle|\tilde\psi|^2\rangle$
  for $a^2Q^2\gg1$;
- **box scaling**: for $\varphi\propto Q^{-2}$ (unscreened nuclear tail,
  beyond the screening momentum) $\int_{\notin\mathcal B}|\varphi|^2d^2Q\propto
  Q_B^{-2}$ with $Q_B=(B+\tfrac12)g$, so $\Gamma_{\rm tail}\propto B^{-2}$.

The plane-wave input $|\tilde\psi|=1$ makes (2) exact for every $Q$; its
integral is $\lambda D(0)/(A(2\pi)^2)$, the continuum plane-wave rate.

**Classification condition.** A sampled out-of-box output is emitted as a
`"tail"` record when:

1. $\kappa+Q$ lies outside the box (exact, by the fold shift); and
2. $|K|\le K_c$ = `tail_q_max_inv_ang`, a caller-declared paraxial cap. The
   phase-shift (projected-potential) jump $\varphi=-V/\hbar v$ neglects the
   longitudinal transfer $q_z\simeq Q^2/2k$ over the interaction length
   $\sim1/Q$, a relative error $\sim\theta/2$; the record is not claimed
   beyond $K_c$, and that weight stays `unresolved`. The research study uses
   $K_c=0.1\,k$ ($\theta\le0.1$ rad).

Why it may be emitted rather than carried coherently: within the absorbing
no-return model every output outside $P(\kappa')$ terminates, and the
represented box $-P$ part is already terminal. A tail order is coupled to the
retained window only by potential harmonics at transfer
$\ge(B-M+\tfrac12)\min g$, further than any box order, so it satisfies the
no-return premise at least as well as the represented outgoing class. The
tail is therefore a classified **model** outcome; whether either class is a
physical outgoing state is the outgoing-instrument definition (#883), not
this claim. A tail is never relabelled as physical escape.

**What stays unresolved.** Lattice-column outputs beyond the box (no momentum
sample is drawn for the Kraus remainder), kick outputs beyond the box (zero
when $M+H+1\le B$), string outputs beyond $K_c$, and every string tail when
`tail_q_max_inv_ang` is None (the default; the earlier behaviour).

**Limits.**

- $B\to\infty$: $\Gamma_{\rm tail}\to0$ as $B^{-2}$; retained weight is $B$
  independent and outgoing + tail is $B$ independent for string-only models.
- $K_c\to0$: every tail is unresolved, the pre-change model.
- $a^2\to0$: $w\to0$, no rate and no tail.
- Plane-wave input: (2) is exact.

## Evidence

All in `tests/montecarlo/test_channeling_axial_tds_continuum.py`:

- **Large-$Q$ law.** Random 9-order state, off-centre string, non-square
  cell, $\varphi=1/(1+0.05q^2)$ ($|\varphi|^2\sim q^{-4}$). The exact tail
  rate ($\langle\chi|\Gamma|\chi\rangle$ minus a tensor Gauss–Legendre box
  integral of (1)) against the box complement of (2): relative difference
  2.7%, 1.2%, 0.61%, 0.36% at $B=6,12,18,24$, decreasing as
  $O((g/Q_B)^2)$. $(B+\tfrac12)^2\Gamma_{\rm tail}/\Gamma$ = 0.721, 0.757,
  0.771 at $B=12,18,24$ ($B^{-2}$ up to the screening correction).
- **Labels only.** Runs with cap None, 25 Å$^{-1}$ and $10^4$ Å$^{-1}$
  (400 incidents) have identical depths and jump counts; exactly the
  formerly unresolved events become tails, each with fold shift beyond $B$,
  and the 25 Å$^{-1}$ cap splits them by $|K|$.
- **Closure and $B$ convergence** (free-particle fibre, 1,500 incidents,
  $B=12,18,24$): retained + outgoing + tail + unresolved = 1 per incident,
  unresolved = 0 (string-only, cap $10^4$); retained and outgoing + tail
  agree with $B=12$ within 4.5σ; the tail falls monotonically with
  $\mathrm{tail}(12)/\mathrm{tail}(24)\in(2,8)$ (law: 3.84; measured
  0.049, 0.023, 0.012 in a 1,000-incident pilot).
- **Refusals:** zero, negative, infinite or NaN caps.

**Si [001], 5 MeV, $B=18$** (4,000 incidents, seed 0, $K_c=0.1k=278.08$ Å$^{-1}$,
$k=2780.8$ Å$^{-1}$; [check record](../check-records/axial-zone/fiber-tail-001-local-4k.json)).
Retained and outgoing are identical to the unclassified run
(`fiber-continuum-001-local-4k.json`), as the labels-only property requires.
At $3\times10^4$ Å:

| Model | Tail | Unresolved | 95% upper (Clopper–Pearson) | Law: $P(\lvert K\rvert>K_c\mid\notin\mathcal B)\times$(tail + unresolved) |
| --- | ---: | ---: | ---: | ---: |
| TDS only | 0.0487 | 0.0025 (10 events) | 0.0042 | 0.0030 |
| TDS + electronic | 0.0452 | 0.0045 (18 events) | 0.0067 | 0.0029 (string part; 9 of 18 events are kick remainders) |

$Q_B=18.5g=60.5$ Å$^{-1}$ is the nearest out-of-box $|K|$. The TDS-only
residual is the string tail beyond the paraxial cap, as the $Q^{-2}$ law
predicts. With electronic kicks the residual is about twice the law
(Poisson $p\approx0.007$ for 18 given 9.4). The kick nodes reach
$q=573$ Å$^{-1}$, and 250 of 680 nodes (0.19% of the kick column weight) can
place output orders beyond $B$; those kick remainders stay unresolved by
design and are not separated in the event record. Tail $|K|$ quantiles
(TDS only): median 91, 90% 173, max 272 Å$^{-1}$.

## Limitations

- Absorbing no-return fibre model; tail records are model outcomes, not a
  physical escape, launch or angular distribution (#883).
- Elastic, paraxial; the cap is declared by the caller.
- Only continuum strings produce tails; lattice-column and kick remainders
  stay unresolved.
- The carrier on a tail event is the pre-jump carrier; the record's
  `tail_momentum_inv_ang` is absolute and unfolded.

## Fresh-context verification (2026-11-01)

**Verdict: `anchored`** (re-derivation matches; anchor tests green, no
discrepancy). The derivation below was written from the ledger row and the
cited joint $(Q,u)$ law of `cr-axial-continuum-tds-jumps` before the
implementation bodies and the task-owner derivation above were read. Commit
`2e0fff505` (code at `35545eb20`).

- **Claim**: `cr-axial-tds-tail`, `montecarlo/channeling/axial_tds_continuum.py::{TailRecord,string_jump}` and `montecarlo/channeling/axial_fiber_unraveling.py::{FiberChannels,_jump,_trajectory}`. Source: the joint $(Q,u)$ law of `cr-axial-continuum-tds-jumps`; Castin–Dalibard–Mølmer Eqs. (4), (8).
- **Filters**: units pass (density per $d^2Q$ in Å: $\lambda$ [Å$^{-1}$] $\times$ $\lvert\varphi\rvert^2$ [Å$^4$] / $A$ [Å$^2$]; integrates to a rate in Å$^{-1}$); limits pass (plane wave exact, $a^2\to0$ gives zero, $B\to\infty$ gives $B^{-2}$, $K_c\to0$ gives the pre-change labels); signs and conventions pass ($K=\kappa+Q$ absolute and unfolded, half-open fold zone, string phase $e^{-i\mathbf t\cdot\mathbf r}$).
- **Re-derivation**: `matches`.

### Independent derivation

Let the window state be $\chi$ on orders $\mathbf k_p=\kappa+\mathbf g_p$, and
let $\psi(\mathbf x)=\sum_p\chi_pe^{i\mathbf k_p\cdot\mathbf x}$, so that the
cell average of $\lvert\psi\rvert^2$ is $\sum\lvert\chi_p\rvert^2=1$. The string
jump with displacement $u$ maps $\chi$ to the output amplitude at absolute
wavevector $K$:

$$
a(K,u)=\sqrt\lambda\sum_p\chi_p\,\frac{\varphi(\lvert\mathbf t_p\rvert)}{A}e^{-i\mathbf t_p\cdot\mathbf r}
\big(e^{-i\mathbf t_p\cdot u}-e^{-a^2t_p^2/2}\big),\qquad \mathbf t_p=K-\mathbf k_p,
$$

with rate density $A\,\lvert a\rvert^2N(u)/(2\pi)^2$ per $d^2K\,d^2u$. The
Gaussian average gives
$\mathbb E_u[w_{\mathbf t}w^*_{\mathbf t'}]=e^{-a^2\lvert\mathbf t-\mathbf t'\rvert^2/2}-e^{-a^2(t^2+t'^2)/2}$
(the two cross terms each give $e^{-a^2(t^2+t'^2)/2}$ and the constant term
gives it back once), with $\mathbf t_p-\mathbf t_{p'}=\mathbf g_{p'}-\mathbf g_p$.
This is the owner's Eq. (1).

**Large-$Q$ limit.** Write $Q=K-\kappa$. For $Q\gg\lvert\mathbf g_p\rvert$
replace $\varphi(\lvert\mathbf t_p\rvert)\to\varphi(Q)$ and
$e^{-a^2t_p^2/2}\to e^{-a^2Q^2/2}$. Since
$e^{-i\mathbf t_p\cdot(\mathbf r+u)}=e^{-iK\cdot(\mathbf r+u)}e^{i\mathbf k_p\cdot(\mathbf r+u)}$,

$$
\sum_p\chi_pe^{-i\mathbf t_p\cdot\mathbf r}w_{\mathbf t_p}(u)
\to e^{-iK\cdot\mathbf r}\big[e^{-iK\cdot u}\psi(\mathbf r+u)-e^{-a^2Q^2/2}\psi(\mathbf r)\big].
$$

Squaring and averaging over $u$, the cross term is
$-2e^{-a^2Q^2/2}\,\mathrm{Re}\,\psi^*(\mathbf r)\,\mathbb E_u[e^{-iK\cdot u}\psi(\mathbf r+u)]$.
The expectation equals $\sum_p\chi_pe^{i\mathbf k_p\cdot\mathbf r}e^{-a^2t_p^2/2}\to e^{-a^2Q^2/2}\psi(\mathbf r)$,
so

$$
\bar f(Q)\to\frac{\lambda\lvert\varphi(Q)\rvert^2}{A(2\pi)^2}
\Big[\langle\lvert\psi(\mathbf r+u)\rvert^2\rangle_u-e^{-a^2Q^2}\lvert\psi(\mathbf r)\rvert^2\Big],
$$

which is the ledger law. For a single order (plane wave), $\lvert\psi\rvert=1$
and the sum has one term, so with $Q$ measured from that order the law is exact
at every $Q$: $\lambda\lvert\varphi\rvert^2(1-e^{-a^2Q^2})/(A(2\pi)^2)$. Its
integral is $\lambda D(0)/(A(2\pi)^2)$. The leading correction is
$O(g_p\varphi'/\varphi)$, which is linear in $\mathbf g_p\cdot\hat Q$. It is
odd in the azimuth for the diagonal terms, so the integrated complement error
is $O((g/Q_B)^2)$ apart from the off-diagonal terms. This is consistent with the
owner's 2.7%, 1.2%, 0.61%, 0.36% sequence.

**Box scaling.** The box complement in $K$ is the outside of the rectangle
$\lvert K_i\rvert\ge(B+\tfrac12)g_i$ (half-open fold). For
$\lvert\varphi\rvert^2=CQ^{-4}$ and $a^2Q^2\gg1$, in polar coordinates,

$$
\int_{\notin\mathcal B}\lvert\varphi\rvert^2d^2Q=\frac{C}{2}\int_0^{2\pi}\frac{d\theta}{\rho(\theta)^2},
$$

where $\rho(\theta)$ is the rectangle's boundary radius. The integral is
homogeneous of degree $-2$ in the box size, which gives $\Gamma_{\rm tail}\propto(B+\tfrac12)^{-2}$.
For a square of half-side $Q_B$ it is $\pi C\,c_\square/Q_B^2$ with
$c_\square=\tfrac12+\tfrac1\pi=0.818$. The complement of the disk $\lvert K\rvert\le K_c$
is $\pi C/K_c^2$.

**Classification.** A window order sits at $\lvert n_i\rvert\le M$ and an
out-of-box output has some $\lvert K_i\rvert\ge(B+\tfrac12)g_i$, so every
transfer that couples them has a component of at least $(B-M+\tfrac12)g_i$.
That is further than any transfer into the represented box. Recording an
outcome the model already draws (the momentum-basis refinement of the
terminal out-of-box effect) changes no probability. If it consumes no random
numbers, every history is bitwise preserved. Closure follows:
retained + outgoing + tail + unresolved = 1, because tail is a subset of the
former unresolved set.

### Code comparison

`string_jump` forms `moved = kappa + q_rel`, the absolute $K$, from the exact
joint draw. It returns `TailRecord(moved, u)` when
`max|fold(moved)| > box`, before any further RNG call. `_jump` turns it into
`_TAIL` only if `hypot(K) <= tail_q_max_inv_ang` and otherwise into the
earlier `None` (unresolved). `_trajectory` emits the record with the
pre-jump `kappa`; `group - 1` is the string index, since lattice is group 0.
`_check_channels` refuses zero, negative and non-finite caps. Lattice and
kick branches are unchanged. All of this matches the derivation.

### Own numerical checks

Scripts are in the session scratchpad (`val_tail_*.py`) and were not committed.

| Check | Setup | Result |
| --- | --- | --- |
| Own Eq. (1) total vs `StringChannel.rate` | Cell $g=(0.8,1.25)$ Å$^{-1}$, string at $(0.37,-0.71)$ Å, $a^2=0.15$ Å$^2$, $\varphi=(1+0.2q)^{-2}$, random 9-order $\chi$; tensor GL to $160g$ plus analytic law remainder | ratio $1-1.4\times10^{-7}$ |
| Out-of-box probability: own quadrature vs `string_jump` TailRecord fraction | Same, $\kappa=(0.23,-0.41)$, 60k jumps per $B$ | $B=3$: 0.74532 vs $0.74477\pm0.00178$ ($z=-0.31$); $B=6$: 0.47707 vs $0.47567\pm0.00204$ ($z=-0.69$) |
| Large-$Q$ law vs exact complement (small boxes) | Same | $B=3$: 0.973, $B=6$: 0.969 (truncated at $40\times$ box; $g/Q_B\sim0.2$) |
| Tail $\lvert Q\rvert$ marginal, plane wave, $\lvert Q\rvert>25$ | 4,000 samples, KS vs $q\lvert\varphi\rvert^2(1-e^{-a^2q^2})$ | radial $p=0.25$, azimuth uniform $p=0.48$ |
| Same, random $\chi$ (asymptotic law) | 3,000 samples | radial $p=0.42$, azimuth $p=0.39$ |
| Cap relabelling bitwise | Free fibre, 2 lattice columns, 3 kick nodes ($q=0.4,2.5,6$), two strings, $B=3$; caps None / 8 / $10^4$, 600 incidents | depth, lattice and kick counts, carrier and weight identical; non-unresolved kinds and Weyl indices identical; tails (30 / 61, both strings) only from former unresolved, all outside box, all within cap |
| Si [001] square-box law | Production `_potential` (local slope $-1.92$ at 30–60, $-2.00$ at $\ge278$ Å$^{-1}$), $a^2=0.0061$ Å$^2$, $Q_B=60.54$, $K_c=278.08$ Å$^{-1}$ | $P(\lvert K\rvert>K_c\mid\notin\mathcal B)=0.0589$ (owner's $(Q_B/K_c)^2=0.0474$; ratio 1.24) |
| Si [001] TDS + electronic unresolved by origin | `fiber_report(4000, seed=0, electronic=True, B=18, strings, θ≤0.1)` with `_jump` wrapped to log the terminal branch | reproduces the record (tail 0.04525, unresolved 0.0045 = 18). 181 string tails; 9 string outputs beyond $K_c$ (law: $190\times0.0589=11.2$); 9 kick remainders (`_split` choice 2); 0 lattice |

Anchor tests: `pyrite-dev test --slow tests/montecarlo/test_channeling_axial_tds_continuum.py tests/montecarlo/test_channeling_axial_fiber_unraveling.py`,
33 passed.

### Findings

No blocking findings.

Minor findings:

1. **Si law column uses a disk for the square box.** The owner's prediction
   $(Q_B/K_c)^2\times(\text{tail}+\text{unresolved})=0.0024$ omits the
   square-box factor $1/c_\square$ and the slightly sub-$Q^{-2}$ production
   slope near $Q_B$. With production $\varphi$ and the rectangle complement the
   prediction is $0.0030$, or 12.1 events expected against 10 observed for the
   TDS-only run. The conclusion, that the residual is the string tail beyond
   the cap, still holds. The electronic excess is confirmed by instrumentation: 9 of the 18 unresolved events are kick remainders and 9 are string outputs beyond the cap. With the corrected law that is about 11 expected string events. The doc table, the ledger note and the
   check record's `law_prediction_(Q_B/K_c)^2_x_(tail+unresolved)` should carry
   the corrected value.
2. **$k$ quoted inconsistently.** The doc says $k=2782$ and $K_c=278.2$ Å$^{-1}$;
   the record has $k=2780.79$ and $K_c=278.08$ Å$^{-1}$. This is cosmetic.
3. **Anchor coverage of "labels only".** The anchor compares depth and
   `lattice_jumps` on a string-only fibre. The verifier's check extends this to
   lattice and kick channels, kick counts and carriers, and passed. The anchor
   could add `kick_jumps` and a kick channel.

### Owner decisions

- **Caller-set paraxial cap ($\theta\le0.1$ rad).** This is sound. The
  projected-potential (eikonal) jump drops $q_z\simeq Q^2/2k$, a relative
  error of about $\theta/2$, and the Si tail sits mostly well inside the cap
  (median $\lvert K\rvert=91$ Å$^{-1}$, $\theta=0.033$). A hard,
  caller-declared cap with refusals is the honest choice. Weight beyond the
  cap stays visible as unresolved.
- **Tail as a model outcome inheriting no-return.** This is correct. The
  coupling-distance argument makes tail orders at least as no-return as the
  represented outgoing class. Not calling it physical escape (#883) is
  appropriate.
- **Kick and lattice remainders left unresolved.** This is consistent: the
  draw produces no momentum sample for them, so classifying them would need a
  new draw and would break the labels-only property.
  The Si electronic run measures their cost: 9 of 4,000 incidents (0.0023).
- **Pre-jump carrier on tail events.** This is acceptable and documented in
  `FiberEvent`. Consumers must use `tail_momentum_inv_ang` (absolute,
  unfolded), not `carrier_inv_ang`, for a tail.
