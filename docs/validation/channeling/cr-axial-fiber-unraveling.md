# Bloch-fibre unraveling with lattice and kick jumps

## Independent derivation

Claim: `cr-axial-fiber-unraveling`. This fresh verification context wrote
this section from the ledger row, the module and function docstrings, the
cited source and the anchored dense oracle `cr-axial-absorbing-jump-instrument`
before reading any implementation body. Source:
[Castin, Dalibard and Mølmer, arXiv:0805.4002](https://arxiv.org/abs/0805.4002),
Eqs. (4) (waiting law) and (8) (jump selection); the terminal-event algebra
inside one fibre is the already re-derived
[`cr-axial-absorbing-jump-unraveling`](cr-axial-absorbing-jump-unraveling.md).
The claim is finite sampling algebra on a supplied model; it establishes no
physical escape, matching surface or admission.

### Fibres and block diagonality

Take a rectangular transverse lattice with reciprocal vectors
$\mathbf g_{\mathbf p}=(p_xg_x,p_yg_y)$. A Bloch state of carrier
$\boldsymbol\kappa$ is
$\psi=\sum_{\mathbf p}c_{\mathbf p}e^{i(\boldsymbol\kappa+\mathbf g_{\mathbf p})\cdot\mathbf r}$.
The model's operators act as follows:

- A lattice-periodic $H$ maps each fibre to itself; its matrix
  $H(\boldsymbol\kappa)$ acts on orders.
- A lattice-transfer jump $L_j=\sum_{\mathbf t}\ell_{j,\mathbf t}e^{i\mathbf g_{\mathbf t}\cdot\mathbf r}$
  is a periodic multiplication. It keeps $\boldsymbol\kappa$ and acts as the
  convolution $(L_jc)_{\mathbf p}=\sum_{\mathbf t}\ell_{j,\mathbf t}c_{\mathbf p-\mathbf t}$.
- A kick family
  $K_i(\theta)=S_ie^{iq_i\hat{\mathbf e}_\theta\cdot\mathbf r}$, where
  $S_i=\sum_{\mathbf t}s_{i,\mathbf t}e^{i\mathbf g_{\mathbf t}\cdot\mathbf r}$
  is periodic, sends fibre $\boldsymbol\kappa$ to
  $\boldsymbol\kappa+q_i\hat{\mathbf e}_\theta$.

The dissipator is
$\sum_i\int_0^{2\pi}\frac{d\theta}{2\pi}K_i(\theta)\rho K_i(\theta)^\dagger$.
Write $\rho=\int d\boldsymbol\kappa\,\rho(\boldsymbol\kappa)$, supported on
fibre-diagonal blocks. Then each term $K\rho K^\dagger$ sends the block
$(\boldsymbol\kappa,\boldsymbol\kappa)$ to the block
$(\boldsymbol\kappa',\boldsymbol\kappa')$. The coherent part and the
anticommutator act within a block. So no inter-fibre coherence is ever
created from a single-fibre entrance: the fibre label is a classical random
variable. A pure-state unraveling therefore holds a definite folded carrier
plus order coefficients, and the jump family indexed by $\theta$ is sampled
as a continuous channel. This construction is exact.

### Azimuth-independent kick rate

$e^{i\mathbf q\cdot\mathbf r}$ is unitary, and it commutes with the
multiplication $S_i$. Hence

$$
\lVert S_ie^{i\mathbf q\cdot\mathbf r}\psi\rVert^2
=\lVert e^{i\mathbf q\cdot\mathbf r}S_i\psi\rVert^2=\lVert S_i\psi\rVert^2 .
$$

The joint density of (node $i$, azimuth $\theta$) is
$\lVert S_i\psi\rVert^2\,d\theta/2\pi$. It factorizes, so drawing the node
with weight $\lVert S_i\psi\rVert^2$ and then a uniform $\theta$ is exact.
For a discrete commensurate azimuth set the measure is uniform over those
angles. With $S_i=\sqrt{c_i\rho_i}$ the node weight is
$c_i\langle\rho_i\rangle$. This matches the production electronic jump
$\phi\leftarrow\sqrt{\rho_q}\phi$, $\boldsymbol\kappa\to\boldsymbol\kappa+\mathbf q$.

Summed over the azimuth, the kick contributes
$\Gamma_{\rm kick}=\sum_iS_i^\dagger S_i$ to the no-jump generator,
independent of $\theta$. The fibre-$\boldsymbol\kappa$ effective Hamiltonian
is therefore
$P[H(\boldsymbol\kappa)-\tfrac{i}{2}\hbar v(\Gamma_{\rm lat}+\Gamma_{\rm kick})]P$,
in eV, with rates in Å$^{-1}$ and depth $z=vt$.

### Order relabelling after folding

The output plane waves are
$e^{i(\boldsymbol\kappa+\mathbf q+\mathbf g_{\mathbf p})\cdot\mathbf r}$.
Fold with an integer vector $\mathbf n_0$:
$\boldsymbol\kappa'=\boldsymbol\kappa+\mathbf q-\mathbf g_{\mathbf n_0}$.
Then

$$
e^{i(\boldsymbol\kappa+\mathbf q+\mathbf g_{\mathbf p})\cdot\mathbf r}
=e^{i(\boldsymbol\kappa'+\mathbf g_{\mathbf p+\mathbf n_0})\cdot\mathbf r},
\qquad \boxed{\mathbf p\mapsto\mathbf p+\mathbf n_0}.
$$

The window $R$ in the new fibre covers orders $[-M,M]^2$ relative to
$\boldsymbol\kappa'$.

### Linear convolutions, kick rate and kick loss

Store inputs $\mathbf p\in[-M,M]^2$ at index $\mathbf p+M$ and kernel orders
$\mathbf t\in[-D,D]^2$ at $\mathbf t+D$. The full linear convolution then
puts output order $\mathbf p+\mathbf t$ at index $\mathbf p+\mathbf t+M+D$.
An FFT of size $2M+2D+1$ per axis is wrap-free, and the in-window outputs
$[-M,M]^2$ sit at index offset $D$. For a kick of half-width $H$, a relabelled
window order $\mathbf p'$ reads the unrelabelled order
$\mathbf p'-\mathbf n_0$, which is full index $\mathbf p'-\mathbf n_0+M+H$.
Outputs outside $R$ are never wrapped back.

Parseval on the wrap-free convolution gives
$\lVert S\chi\rVert^2=N^{-2}\sum_{\mathbf k}\lvert\hat s_{\mathbf k}\rvert^2\lvert\hat\chi_{\mathbf k}\rvert^2$,
with $N=2M+2H+1$. The window-input matrix of $S^\dagger S$ over all outputs is

$$
(S^\dagger S)_{\mathbf p\mathbf p''}=\sum_{\mathbf p'}\overline{s_{\mathbf p'-\mathbf p}}\,s_{\mathbf p'-\mathbf p''}
=C(\mathbf p-\mathbf p''),\qquad C(\mathbf d)=\sum_{\mathbf t}\overline{s_{\mathbf t}}\,s_{\mathbf t+\mathbf d}.
$$

With the forward FFT convention
$\hat x_k=\sum_nx_ne^{-2\pi ikn/N}$, $\mathrm{IFFT}(\lvert\hat x\rvert^2)_d=\sum_n\overline{x_n}x_{n+d}$
(circular), which is $C(\mathbf d)$ and not $C(-\mathbf d)=\overline{C(\mathbf d)}$.
$C$ is supported on $\lvert d\rvert\le2H$, and it is read for
$\lvert d\rvert\le2M$. Circular aliasing is absent if $N\ge4H+1$ (support
does not overlap itself) and $N\ge2M+2H+1$ (no read lag $d$ meets an image
$d-N$ inside the support). So
$N=\max(2H+2M+1,4H+1)$.

### Retained, outgoing and unresolved split

For a jump output $\phi$ in fibre $\boldsymbol\kappa'$, the dense absorbing
model splits $\phi$ into three parts:

- $P\phi$ is recycled.
- $(R-P)\phi$ is outgoing and Weyl measured.
- $(1-R)\phi$ is unresolved.

Choosing the outcome with probability proportional to these squared norms
reproduces the dense tallies, by the one-fibre theorem of
`cr-axial-absorbing-jump-unraveling`.

Lattice Kraus columns are truncated at $\lvert\mathbf t\rvert_\infty\le D$,
while $\Gamma_{\rm lat}$ carries the full loss. The represented channels are
$RL_j$, with rate $\lVert RL_j\chi\rVert^2$; their outputs have no
$(1-R)$ part. Everything else is one unresolved channel, consisting of the
outside-window parts of $L_j\chi$ plus the transfers beyond $D$:

$$
\Gamma_{\rm unres,lat}=\Gamma_{\rm lat}-\sum_jL_j^\dagger RL_j\ge0,
\qquad r_{\rm unres}=\langle\chi|\Gamma_{\rm lat}|\chi\rangle-\sum_j\lVert RL_j\chi\rVert^2 .
$$

This is the Kraus complement of the dense oracle's $E_O$, so the total rate
equals $\langle\chi|\Gamma_{\rm lat}+\Gamma_{\rm kick}|\chi\rangle$, which is
the norm-decay rate. A kick keeps its full output, and its unresolved weight
is $\lVert S_i\chi\rVert^2-\lVert R\,\mathrm{relabel}(S_i\chi)\rVert^2$.

For commensurate kicks, the fibres reachable from $\boldsymbol\kappa_0$ form
a finite set. The retained space is then the direct sum
$\bigoplus_fP(\boldsymbol\kappa_f)$, and the fibre-diagonal blocks satisfy
a finite GKSL equation that the dense oracle integrates.

## Filters

| Check | Result |
|---|---|
| Units | pass. $H$ eV, $\hbar v$ eV Å, columns Å$^{-1/2}$, rates and $\Gamma$ Å$^{-1}$, $q,\kappa,g$ Å$^{-1}$; the propagator is $e^{-i\lambda z/\hbar v}$. |
| Limits | pass. Zero channels leave every incident retained. Kick-free runs reduce to the one-fibre unraveling. For $q=\lvert\mathbf g\rvert$ on the axes, $\boldsymbol\kappa'=\boldsymbol\kappa$ and the result is an order shift by $\pm\hat{\mathbf x},\pm\hat{\mathbf y}$. |
| Signs and conventions | pass. Relabel is $+\mathbf n_0$; $C(\mathbf p-\mathbf p')$ is not transposed; the $-\tfrac{i}{2}\hbar v\Gamma$ term damps. |

## Source-to-code comparison

Code: `src/pyrite/montecarlo/channeling/axial_fiber_unraveling.py` at
`6925c8cf2`.

| Derivation | Code | Agreement |
|---|---|---|
| $N=\max(2H+2M+1,4H+1)$, $C=\mathrm{IFFT}\sum_i\lvert\hat s_i\rvert^2$, matrix $C(\mathbf p-\mathbf p')$ | `_autocorrelation_loss` L103–109 | match; kernel stored at $\mathbf t+H$ cancels in the lag |
| wrap-free size $2M+2D+1$, window offset $D$ | `_Engine.__init__` L131–133, `lattice_outputs` L166–169 | match |
| Parseval $N^{-2}\sum\lvert\hat s\rvert^2\lvert\hat\chi\rvert^2$ with $N=2M+2H+1$ | `kick_rates` L176–177 | match |
| source index $\mathbf p'-\mathbf n_0+M+H$ | `kick_output` L189: `axis_index - m - shift + m + h` | match |
| $\mathbf n_0=\mathrm{round}((\boldsymbol\kappa+\mathbf q)/\mathbf g)$, $\boldsymbol\kappa'=\boldsymbol\kappa+\mathbf q-\mathbf g_{\mathbf n_0}$ | L290–292 (`np.rint`) | match for rectangular lattices |
| $P[H-\tfrac{i}{2}\hbar v(\Gamma_{\rm lat}+\Gamma_{\rm kick})]P$ | `fiber` L151 | match (assumes orthonormal `spectrum` vectors) |
| rates $\lVert RL_j\chi\rVert^2$, lattice complement (clipped at 0), $\lVert S_i\chi\rVert^2$ | `_trajectory` L263–273 | match |
| split $\lVert P\phi\rVert^2$, $\lVert(R-P)\phi\rVert^2$, outside | `_split` L209–216; kick outside L295 | match |
| `retained` at the horizon | L262 | match; the closure reads the current carrier and jump counts (late binding) |
| streams | `fiber_stream` (Philox, domain `0x6230002`, version 1) for jumps, azimuth and split; `phase_space_stream` for the single measurement | independent per incident |

The cache key rounds $\boldsymbol\kappa$ to $10^{-12}$ Å$^{-1}$, and the
spectrum is evaluated at that key. This is harmless: the carrier is shifted
by at most $5\times10^{-13}$ Å$^{-1}$.

### Minor findings (no physics divergence)

1. Tie at the zone boundary (`axial_fiber_unraveling.py:291`). Suppose a
   commensurate kick lands exactly on $\pm g/2$, for example $q=g_y/2$ at
   $90^\circ$ from $\kappa_y=0$. Then `np.rint` chooses the window centring by
   round-half-even on floating noise. The images $\pm g/2$ are one physical
   fibre, but they give different windows $R$, spectra and cache keys, so
   the finite model is not uniquely defined. A dense oracle that picks the
   other image would disagree. The set has measure zero for continuous
   azimuths.
2. Unchecked shapes. `kick_q_inv_ang` and `kick_columns` lengths are not
   cross-validated. A shorter `q` raises `IndexError` only when that node
   fires (observed), and a longer one is silently ignored. An empty
   `kick_azimuths` fails at the first kick.
3. Silent clip. The lattice complement is clipped at 0 (L267) without
   checking the stated `lattice_loss - sum_j L_j^dagger R L_j >= 0`. A provider
   violating it biases the tallies silently instead of refusing.
4. The fold assumes a rectangular reciprocal lattice (per-axis
   `reciprocal_inv_ang`). The docstrings do not state this restriction.

## Independent numerical evidence (2026-10-31)

Scripts are in the session scratchpad (`val_fiber_model.py`,
`val_fiber_dense.py`, `val_fiber_continuous.py`). The model is my own:

- window $M=2$ (25 orders);
- $H(\boldsymbol\kappa)=0.8\lvert\boldsymbol\kappa+\mathbf g_{\mathbf p}\rvert^2$ plus a random Hermitian periodic potential;
- two random lattice columns ($D=1$), plus a random PSD beyond-$D$ loss;
- two random kick columns ($H=1$);
- $\hbar v=2$ eV Å and cut 5 eV.

The dense oracle is my own block-diagonal GKSL integration (DOP853,
`rtol` $10^{-10}$) built from direct-matrix convolutions. It shares no code
helpers.

- **Dense oracle**, 10,000 incidents per case. Cumulative outgoing and
  unresolved tallies at 2, 6 and 15 Å, plus horizon retained per fibre.
  Closure is $1\pm10^{-15}$.

  | Case | Max $\lvert z\rvert$ |
  |---|---|
  | $q=\sqrt2\,g$ at $45^\circ+k\,90^\circ$ (non-axis, single fibre, $\mathbf n_0=(\pm1,\pm1)$), $\boldsymbol\kappa_0=0$ | 1.35 |
  | Same kicks with off-centre $\boldsymbol\kappa_0=(0.21,-0.17)$ | 1.46 |
  | Rectangular $g=(1.3,0.9)$, $q=g_x/2$ at $0,\pi$, giving two fibres $\kappa_x=\pm0.325$ with shifts $0,\pm1$ | 1.04 |

  The incident weights agree to 12 digits. The mean kick count matches the
  dense $\int_0^{15}\operatorname{tr}(\Gamma_{\rm kick}\rho)\,dz$ to
  $\lvert z\rvert\le0.88$ in all three cases.
- **Component identities** on a random $\chi$ against direct matrices:
  - $\Gamma_{\rm kick}$ to $1\times10^{-16}$;
  - $RL_j\chi$ to $3\times10^{-16}$;
  - kick rates to $7\times10^{-15}$;
  - relabelled kick outputs for $\mathbf n_0\in\{0,(1,-1),(-2,1),(3,0)\}$ to $5\times10^{-16}$.
- **Continuous azimuth** ($q=0.77,1.9$ Å$^{-1}$ on the rectangular lattice).
  My own trajectory sampler (direct matrices, `expm`, bisection, floor-based
  fold; 3,658 distinct fibres visited) is compared with the code at 3,000
  incidents each:
  - cumulative tallies agree to $\lvert z\rvert\le1.84$ (correlated);
  - horizon retained 0.030 vs 0.031;
  - mean kicks 1.219 vs 1.216;
  - all folded carriers are in the zone.

  A repeat with fresh seeds at 8,000 incidents each gives:
  - tallies to $\lvert z\rvert\le0.78$;
  - retained 0.0304 vs 0.0308;
  - mean kicks 1.247 vs 1.195 ($z=2.6$; pooled over both runs about
    $2.4\sigma$).

  The commensurate dense kick counts above agree. The residual
  kick-count tension is therefore unattributed: chance, the reference
  sampler's 0.05 Å fixed-step bracket, or a continuous-only effect. It is
  recorded as an open low-weight concern, not a discrepancy.
- `pyrite-dev test tests/montecarlo/test_channeling_axial_fiber_unraveling.py tests/dev/test_validation_ledger.py`:
  24 passed.

**Verdict: `rederived`.** Human sign-off remains in #277.

## Task-owner response (2026-10-31)

The four minor findings are addressed. None of them changes the law the code samples.

1. Folding is now deterministic: `_fold` computes `floor(round(κ/g, 12) + 1/2)` per axis. Carriers therefore fold into the half-open zone `[-g/2, g/2)`, and a boundary tie maps to `-g/2` whatever the floating-point noise. Covered by `test_zone_boundary_ties_fold_to_the_half_open_zone`.
2. `_check_channels` refuses mismatched `kick_q_inv_ang`/`kick_columns`, an empty `kick_azimuths`, or a `lattice_loss` whose shape does not match the window. Covered by `test_inconsistent_channels_refuse`.
3. A negative lattice complement beyond `_COMPLEMENT_RTOL = 1e-9` (relative) now refuses instead of being clipped. Covered by `test_lattice_loss_below_its_kraus_outputs_refuses`. The Si [001] material channels pass this check.
4. The `FiberChannels` docstring now states the rectangular-cell and fold convention.

The kick-count tension in the continuous-azimuth comparison is still open and low weight. The commensurate dense oracles agree.

Material cross-check: at the Γ point with TDS only, the fibre engine's retained, outgoing and unresolved rate operators match `tds_outgoing_maps` (Si [001], 5 MeV, M=6, 95 eV cut, 77 retained states) to about 1e-15 relative. The traces, top eigenvalues and entry-state splits all agree. The TDS-only fibre tallies agree with the 20k Γ-point unraveling of `cr-axial-absorbing-jump-unraveling`. At 1e3 Å: seed 0, 4k incidents, 0.030 ± 0.003 (3.3σ); seed 1, 8k incidents, 0.024 ± 0.002 (1.7σ). At 1e4 Å and 3e4 Å both seeds agree within 2σ. Record: `docs/validation/check-records/axial-zone/fiber-local-4k.json`.

## Implementation change: outgoing order box (2026-10-31, awaiting re-verification)

**Change.** `FiberChannels.outgoing_order_max` = B ≥ M sets a centred outgoing box $\mathcal B=[-B,B]^2$. The retained window stays $R=[-M,M]^2$, with $R\subseteq\mathcal B$. Jump outputs are represented and Weyl-measured on $\mathcal B$ rather than on $R$. The default B = M reproduces the previous model exactly.

**Split.** For an output $\phi$ in fibre $\boldsymbol\kappa'$:

- $P(\boldsymbol\kappa')\phi$ is recycled ($P$ is supported on $R$);
- $(\mathcal B-P)\phi$ is outgoing;
- $(1-\mathcal B)\phi$ is unresolved.

These three parts are orthogonal and sum to $\phi$. Lattice channels are $\mathcal BL_j$. The unresolved lattice remainder is $\Gamma_{\rm lat}-\sum_jL_j^\dagger\mathcal BL_j$. A kick's unresolved part is $\lVert S_i\chi\rVert^2-\lVert\mathcal B\,\mathrm{relabel}(S_i\chi)\rVert^2$.

**Exactness condition $D\ge B+M$** (sufficient for every covariance, and necessary for some; see the re-verification below). Write $K(\mathbf t,\mathbf t')$ for the full transfer covariance and $K_D$ for its restriction to $\lvert\mathbf t\rvert_\infty,\lvert\mathbf t'\rvert_\infty\le D$, which the Kraus columns factor as $K_D=\sum_js_js_j^\dagger$. For window inputs $\mathbf p,\mathbf p'$:

$$
(\Gamma_{\rm lat})_{\mathbf p\mathbf p'}=\sum_{\mathbf o}K(\mathbf o-\mathbf p,\mathbf o-\mathbf p'),\qquad
\Big(\sum_jL_j^\dagger\mathcal BL_j\Big)_{\mathbf p\mathbf p'}=\sum_{\mathbf o\in\mathcal B}K_D(\mathbf o-\mathbf p,\mathbf o-\mathbf p').
$$

If $D\ge B+M$, then $\lvert\mathbf o-\mathbf p\rvert_\infty\le D$ for every $\mathbf o\in\mathcal B$ and every $\mathbf p\in R$, so $K_D=K$ on every box term. Two things follow:

- **Exact box outputs.** The represented outputs are the exact $\mathcal BL\chi$.
- **PSD remainder.** The remainder is $\sum_{\mathbf o\notin\mathcal B}M^{\mathbf o}$, a sum of PSD matrices $M^{\mathbf o}_{\mathbf p\mathbf p'}=K(\mathbf o-\mathbf p,\mathbf o-\mathbf p')$.

If $D<B+M$, box terms with some $\lvert\mathbf o-\mathbf p\rvert_\infty>D$ use $K_D$ with rows and columns zeroed. That is not a principal submatrix of $M^{\mathbf o}$, so the remainder can be indefinite and the outputs are wrong. The first test attempt hit exactly this case: a $D=2$ factorization cropped to $B=3$ refused with a negative complement. `_check_channels` now refuses $D<B+M$. The previous default, $B=M$ with $D=2M$, satisfies the condition, which is why the window-only model was exact. Kick columns define their own (truncated) model, and their rate is the Parseval total over all outputs, so a kick needs no such condition.

**Two-stage channel draw (stream version 2).** The channel is now drawn in two stages. The first stage picks the lattice group with weight $\langle\chi|\Gamma_{\rm lat}|\chi\rangle$, or kick node $i$ with weight $\lVert S_i\chi\rVert^2$. If the lattice group wins, the second stage draws $\mathcal BL_j$ with weight $\lVert\mathcal BL_j\chi\rVert^2$, or the remainder. The product of the two probabilities equals the one-stage law, so the sampled law is unchanged. Kraus outputs are now evaluated only at lattice jumps. The draw sequence differs, so `fiber_stream` now keys version 2.

**Evidence.**

- *Dense box oracle* (`test_outgoing_box_matches_dense_box_oracle`). The test builds full-output convolution matrices directly, without the window map builder, with M = 1, B = 3, D = 4 and H = 1. Both cases pass the 4.5σ tallies and the χ² joint Weyl law on the 7×7 box: lattice-only, and one-sided kicks (which fix the relabel sign on the box). `test_outgoing_box_moves_window_fold_outs_from_unresolved_to_outgoing` and the $D<B+M$ refusal are covered too.
- *Injected faults.* Reverting the kick source offset to the window offset fails the kick case. Shifting the lattice crop by one order fails both cases.
- *Material rates* (Si [001], 5 MeV, M = 6, ~95 eV cut, Γ point, mean over the 77 retained states). The retained rate is independent of B, as it must be: 5.071e-5 /Å throughout.

  | B (D = B + 6) | Kraus | Outgoing (/Å) | Unresolved (/Å) |
  |---:|---:|---:|---:|
  | 6 | 309 | 1.212e-5 | 2.846e-5 |
  | 12 | 557 | 3.470e-5 | 5.878e-6 |
  | 18 | 872 | 3.811e-5 | 2.472e-6 |
  | 24 | 1249 | 3.921e-5 | 1.368e-6 |

  From B = 12 to 24 the unresolved remainder falls by 4.3×. That is close to the factor 4 expected from the $Q^{-2}$ single-nuclear tail (the rate beyond $Q$ scales as $\int^\infty q\,dq\,q^{-4}\propto Q^{-2}$).

## Re-verification: outgoing order box (2026-10-31)

Fresh verification context, commit `dfe9dc9d1` (change `33ee0e040..dfe9dc9d1`).
The derivation below was written from the ledger row, the change summary and
the earlier sections of this page before the changed function bodies were read.

**Verdict: `rederived`.** Human sign-off remains in #277.

### Derivation

**(a) Exactness and positivity on a box.** Let the full lattice transfer
covariance be $K(\mathbf t,\mathbf t')=\sum_a\ell_a(\mathbf t)\overline{\ell_a(\mathbf t')}$, and
let the supplied columns $s_j$ factor its principal block on
$\lvert\mathbf t\rvert_\infty,\lvert\mathbf t'\rvert_\infty\le D$. Write
$\mathcal B=[-B,B]^2$ and $R=[-M,M]^2$. For an input $\chi$ on $R$, the box
output density of the true model and of the truncated columns are

$$
\rho^{\rm true}_{\mathbf o\mathbf o'}=\sum_{\mathbf p,\mathbf p'\in R}K(\mathbf o-\mathbf p,\mathbf o'-\mathbf p')\chi_{\mathbf p}\overline{\chi_{\mathbf p'}},
\qquad
\rho^{D}_{\mathbf o\mathbf o'}=\sum_{\mathbf p,\mathbf p'\in R}K_D(\mathbf o-\mathbf p,\mathbf o'-\mathbf p')\chi_{\mathbf p}\overline{\chi_{\mathbf p'}},
$$

with $\mathbf o,\mathbf o'\in\mathcal B$. The differences
$\mathbf o-\mathbf p$ range over $[-(B+M),B+M]^2$. So $\rho^D=\rho^{\rm true}$
for every $K$ if and only if $D\ge B+M$. For the remainder, define
$M^{\mathbf o}_{\mathbf p\mathbf p'}=\sum_a\overline{\ell_a(\mathbf o-\mathbf p)}\ell_a(\mathbf o-\mathbf p')$, which is a
Gram matrix and hence PSD. Then
$\Gamma_{\rm lat}=\sum_{\mathbf o}M^{\mathbf o}$ and
$\sum_jL_j^\dagger\mathcal BL_j=\sum_{\mathbf o\in\mathcal B}\tilde M^{\mathbf o}$.
Here $\tilde M^{\mathbf o}$ is the principal block of $M^{\mathbf o}$ on
$S_{\mathbf o}=\{\mathbf p:\lvert\mathbf o-\mathbf p\rvert_\infty\le D\}$,
padded with zeros. Hence

$$
\Gamma_{\rm lat}-\sum_jL_j^\dagger\mathcal BL_j
=\sum_{\mathbf o\notin\mathcal B}M^{\mathbf o}
+\sum_{\mathbf o\in\mathcal B}\begin{pmatrix}0&X_{\mathbf o}\\X_{\mathbf o}^\dagger&Y_{\mathbf o}\end{pmatrix},
$$

where the blocks are taken on $S_{\mathbf o}$ and its complement in $R$. If
$D\ge B+M$, every $S_{\mathbf o}=R$, the second sum vanishes, and the
remainder is PSD. If $D<B+M$, a block $\begin{pmatrix}0&X\\X^\dagger&Y\end{pmatrix}$
with $X\ne0$ is indefinite. A covariance can make it dominate, as this
construction shows. Take a rank-one $K$ with amplitude $a$ at
$\mathbf t=(D,0)$ and $b$ at $(D+1,0)$, with $\lvert a\rvert>\lvert b\rvert$.
On two $x$-adjacent window orders that both satisfy $p_x+D\le B$, the
remainder block is $\begin{pmatrix}\lvert b\rvert^2&\bar ab\\a\bar b&\lvert b\rvert^2\end{pmatrix}$,
whose determinant is $\lvert b\rvert^4-\lvert ab\rvert^2<0$. Such a pair
exists for every $D\le B+M-1$. So $D\ge B+M$ is necessary and sufficient
for both properties to hold for every covariance. For particular
covariances it is only sufficient. A diagonal (uncorrelated) $K$ has
$X_{\mathbf o}=0$, and its remainder stays PSD at any $D$, but transfers
beyond $D$ that land in $\mathcal B$ are then counted as unresolved rather
than outgoing. A $K$ that is genuinely supported within $\lvert\mathbf t\rvert_\infty\le D$
is exact at that $D$. The code's refusal is therefore conservative for such
providers, which is the correct side.

**(b) Two-stage draw.** The one-stage law has total rate
$r=\langle\chi|\Gamma_{\rm lat}|\chi\rangle+\sum_i\lVert S_i\chi\rVert^2$
and channel probabilities $\lVert\mathcal BL_j\chi\rVert^2/r$,
$(\langle\chi|\Gamma_{\rm lat}|\chi\rangle-\sum_j\lVert\mathcal BL_j\chi\rVert^2)/r$
and $\lVert S_i\chi\rVert^2/r$. The two-stage law draws the lattice group
with probability $\langle\chi|\Gamma_{\rm lat}|\chi\rangle/r$ and then a
member with conditional probability (member weight)$/\langle\chi|\Gamma_{\rm lat}|\chi\rangle$.
The product is identical. It needs the remainder to be nonnegative, which (a)
guarantees. The waiting law depends only on $r$ and is unchanged.

**(c) Index offsets.** A full linear convolution of the window input (index
$\mathbf p+M$) with a kernel (index $\mathbf t+D$) puts output order
$\mathbf o=\mathbf p+\mathbf t\in[-(M+D),M+D]^2$ at index $\mathbf o+M+D$,
and is wrap-free at FFT size $2M+2D+1$. The box crop copies full indices
$[M+D-B,\,M+D+B]$ to box indices $[0,2B]$, which needs $M+D\ge B$; that
holds whenever $D\ge B+M$. For a kick, an unrelabelled output $\mathbf u$
sits at full index $\mathbf u+M+H$. After the fold it becomes
$\mathbf u+\mathbf n_0$, at box index $\mathbf u+\mathbf n_0+B$. So box
index $\mathbf a$ reads full index $\mathbf a-B-\mathbf n_0+M+H$; indices
outside $[0,2M+2H]$ give zero. Window orders sit at box indices
$[B-M,B+M]$. The Weyl layout puts box order $\mathbf o$ at FFT index
$\mathbf o\bmod(2B+1)$.

**(d) Orthogonality and closure.** $P(\boldsymbol\kappa')$ projects onto
spectral vectors supported on $R\subseteq\mathcal B$, and $\mathcal B$ is
diagonal in orders. So $P\le\mathcal B\le1$, and $P$, $\mathcal B-P$ and
$1-\mathcal B$ are orthogonal projectors summing to one. The squared norms
therefore sum to $\lVert\phi\rVert^2$. Lattice outputs $\mathcal BL_j\chi$
have no $1-\mathcal B$ part; their out-of-box weight is in the remainder
channel. A kick splits $\lVert S_i\chi\rVert^2$ three ways, with unresolved
weight $\lVert S_i\chi\rVert^2-\lVert\mathcal B\,\mathrm{relabel}(S_i\chi)\rVert^2\ge0$.
Summed over channels and outcomes, the terminal plus recycled rates equal
$\langle\chi|\Gamma_{\rm lat}+\Gamma_{\rm kick}|\chi\rangle$, which is the
no-jump norm-decay rate. Closure therefore holds.

### Source-to-code comparison

`src/pyrite/montecarlo/channeling/axial_fiber_unraveling.py` at `dfe9dc9d1`:

| Derivation | Code | Agreement |
|---|---|---|
| refuse unless $D\ge B+M$ (default $B=M$ gives $D\ge2M$); $B\ge M$ integer | `_check_channels` L141–146 | match (shape proxy, see finding 1) |
| crop full index $\mathbf o+M+D$ to box index $\mathbf o+B$ | `_crop(m + d, self.box)` L126–130, L193; `lattice_outputs` L221–226 | match ($\min(M+D,B)=B$) |
| kick source index $\mathbf a-B-\mathbf n_0+M+H$, zero outside | `kick_output` L248–250 | match |
| window at box indices $[B-M,B+M]$, C order | `window_in_box` L189–192 | match |
| two-stage group/member law | `_trajectory` L342–354, `_lattice_channel` L287–294 | match; when the remainder is negative within `_COMPLEMENT_RTOL`, the normalization uses $\sum_j\lVert\mathcal BL_j\chi\rVert^2$ (roundoff only) |
| $P$, $\mathcal B-P$, $1-\mathcal B$ split; lattice outside $=0$, kick outside = rate minus box norm | `_split` L269–277, L354, L370–371 | match |
| box order $\mathbf o$ at FFT index $\mathbf o\bmod(2B+1)$, ortho inverse FFT | `_measure` L255–259 | match |
| `window` on the $(2B+1)^2$ grid | L434 | match |
| no-jump generator and kick rate use all outputs | `self.loss` L176, `kick_rates` L233–234 | unchanged, match |
| stream version 2 | `fiber_stream` L104 | match |

### Independent numerical evidence

Scripts are in the session scratchpad: `val_box_oracle.py`,
`val_box_psd.py`, `val_box_psd2.py` and the fault variants
`val_box_fault_*.py`. The model is my own. None of it uses the
implementation's map builders or the test's `_kraus`.

- **Model.** Rectangular cell $\mathbf g=(1.3,0.9)$ Å$^{-1}$ with an
  off-centre carrier $\boldsymbol\kappa_0=(0.21,-0.17)$.
  $H=0.8\lvert\boldsymbol\kappa+\mathbf g_{\mathbf p}\rvert^2$ plus a random
  periodic potential, with 5 retained states. There are three correlated
  lattice Kraus columns on $[-D,D]^2$. $\Gamma_{\rm lat}$ is their exact
  all-output sum plus a random PSD extra, so the remainder is nontrivial.
- **Kicks.** Two kick nodes, both commensurate and non-axis:
  $q=\lvert(g_x,g_y)\rvert$ and $q=2\lvert(g_x,g_y)\rvert$ at azimuths
  $\pm\arctan(g_y/g_x)$ and $\pi\pm\arctan(g_y/g_x)$. They fold back onto
  $\boldsymbol\kappa_0$ with shifts $(\pm1,\pm1)$ and $(\pm2,\pm2)$; the
  second node's outputs lie partly outside the box.
- **Oracle.** A dense GKSL integration (DOP853, `rtol` $10^{-11}$) in the
  retained space, built from direct-matrix convolutions. Cumulative outgoing
  and unresolved tallies are read at 1.5, 4 and 9 Å, plus retained at the
  horizon. The joint Weyl law uses a random window on the box grid, computed
  from the accumulated outgoing density.

| Case | Incidents | Max $\lvert z\rvert$ | Joint Weyl $\chi^2$ $p$ |
|---|---:|---:|---:|
| $M=1$, $B=2$, $D=3$, $H=1$, symmetric azimuths | 20,000 | 1.87 | 0.25 |
| $M=1$, $B=4$, $D=5$, $H=1$, symmetric, seed 7 | 20,000 | 1.33 | 0.013 |
| same, seed 11 | 30,000 | 1.50 | 0.97 |
| same, seed 12 | 30,000 | 2.57 | 0.76 |
| $M=2$, $B=3$, $D=5$, $H=2$, symmetric | 10,000 | 0.68 | 0.83 |
| $M=1$, $B=4$, $D=5$, one-sided azimuths $\pm\arctan(g_y/g_x)$ | 15,000 | 1.13 | 0.86 |

The tallies within a run are correlated. The $B=4$ cumulative-outgoing $z$
values at 4 Å are $+0.24$, $-1.50$ and $+2.57$ over three seeds (80k
incidents), which pools to about $+0.8\sigma$. The seed-7 joint
$p=0.013$ did not recur (seeds 11 and 12: $p=0.97$, $0.76$). The momentum
and position marginals pass in every run ($p\ge0.05$).

- **Power.** The oracle has power against the two offset faults. Flipping the
  oracle's relabel sign is detected only by the one-sided case: momentum
  marginal $p<10^{-3}$, joint $p<10^{-3}$, outgoing $z=-3.8$. The
  symmetric angle set is invariant under the flip. Shifting the oracle's
  lattice box by one order gives outgoing $z=-7.4$ and momentum $p=10^{-3}$.
- **Component identities** ($M=1$, $B=4$, $D=5$, $H=2$, random $\chi$):
  - `lattice_outputs` vs direct $\mathcal BL_j\chi$: $3.6\times10^{-15}$;
  - `kick_output` vs direct relabel-and-crop for $\mathbf n_0\in\{0,(1,-1),(-3,2),(4,4),(6,0),(-8,1)\}$: $\le2.8\times10^{-15}$, including partial and empty overlaps;
  - kick rate exact;
  - `window_in_box` maps every window order correctly;
  - columns with $D=4<B+M$ are refused.
- **Positivity.** I built Kraus factors of the principal $[-D,D]^2$ block of a
  full covariance. For $(M,B)\in\{(1,1),(1,2),(1,4),(2,3)\}$:
  - the rank-one edge construction of (a) gives a negative remainder
    eigenvalue ($-0.25$ to $-0.43$) for every $D<B+M$, and a nonnegative
    one at $D=B+M$;
  - random rank-2 covariances supported on $[-(D+1),D+1]^2$ are indefinite
    for $M=2$, $B=3$, $D\le2$, but often PSD otherwise;
  - full-rank random and diagonal covariances with wider support stay PSD
    for all tested $D$;
  - the box-output density error is $\le2\times10^{-15}$ for $D\ge B+M$ and
    $O(1)$ for $D<B+M$, even when the remainder stays PSD.

  Indefiniteness is therefore a worst-case property. The output-exactness
  failure is generic.
- `pyrite-dev test tests/montecarlo/test_channeling_axial_fiber_unraveling.py tests/dev/test_validation_ledger.py`:
  34 passed.

### Findings (no physics divergence)

1. *Shape proxy* (`axial_fiber_unraveling.py:145`). The $D\ge B+M$ refusal
   reads only `lattice_columns.shape[-1] // 2`. Two cases pass unchecked:
   - Columns zero-padded from a narrower truncation of a correlated
     covariance pass. They give inexact box outputs and possibly an
     indefinite remainder, caught only at a visited $\chi$ by the runtime
     complement check (`_lattice_channel` L290).
   - Non-square or even-width columns are not refused. In that case `fft2`
     with `s=(size, size)` (L180–182) would silently crop or pad axis $-2$.

   The covariance truncation is a documented provider obligation (the
   `FiberChannels` docstring), so this is a robustness note, not a law
   error.
2. *Wording.* The task-owner section calls $D\ge B+M$ the exactness
   condition and says the remainder "can be" indefinite below it. That is
   correct as a statement over all covariances. For a specific covariance,
   the condition is sufficient but not always necessary, per (a).
3. *Behaviour tightening.* With the default $B=M$, the check now refuses
   $D<2M$, which the previous version admitted subject to the runtime
   complement check. This is consistent with (a): such columns were already
   inexact on the window.

### Task-owner response to the re-verification findings (2026-10-31)

1. `_check_channels` now refuses lattice or kick column arrays that are not `(n, 2D+1, 2D+1)`, i.e. not square with odd width, so the FFT can no longer silently crop or pad an axis (`test_inconsistent_channels_refuse`). Zero-padded columns from a narrower factorization cannot be detected from their shape. The `FiberChannels` docstring now says that columns must factor the covariance truncated at their full width, and that the runtime complement refusal is the only backstop for violations.
2. The wording now reads: $D\ge B+M$ is sufficient for every covariance and necessary for some.
3. Default $B=M$ now refuses $D<2M$. This is intended: as finding (a) shows, such columns were already inexact on the window.
