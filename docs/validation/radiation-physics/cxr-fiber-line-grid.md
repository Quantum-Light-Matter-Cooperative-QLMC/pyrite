# Full-fiber roll quadrature: automatic line grid and truncation audit (#1489)

Validation: `cxr-fiber-intensity-average` (the #1489 extension only).

> Rebase note (2026-11-26): this review ran before the
> [#1480 write-up](cxr-fiber-intensity-average.md) reached the branch. Its
> statements that the #1480 parts lack a write-up and that the row stays
> `unverified` describe the pre-rebase branch; the
> [ledger](../ledger-core-coherent-physics.md#cxr-fiber-intensity-average)
> holds the current status. Commit hashes are pre-rebase.

This is a fresh-context verification of commit `c95e81407` against its parent
`d33a428e1`. It covers the roll-weighted truncation audit, the line-population
seeding, the automatic ceiling and the measured-grid cache key. It does **not**
re-derive the parts of the claim that #1480 and #1483 own: the projected
uniaxial $B(\mathbf g)$, the isotropic continuum share, or the quadrature
error of the sector midpoints against the full-circle average. Those parts
still have no fresh-context write-up.

Independence note: the claim is an algorithmic one (a quadrature and the
linearity of an audit), so the source is the ledger row and the issue text,
not a paper. Sections 1 to 4 were derived from that text. The commit diff was
open while §2 was written, because the audit's key names come from the diff.
The derivation does not rely on any code expression. §5 onward compares it
with the implementation.

## 1. Roll quadrature as a weighted sum of calls

Let the domain split into basal rows $\mathcal B=\{\mathbf g:\ g_\perp=0\}$
and the rest $\mathcal O$. With sector-midpoint rolls
$\phi_i=\phi_0+\tfrac{\pi}{3}\tfrac{i+1/2}{N}$, the scored density is

$$
I(E)=\sum_{\mathbf g\in\mathcal B} I_{\mathbf g}(E)
+\frac1N\sum_{i=0}^{N-1}\sum_{\mathbf g\in\mathcal O} I_{\mathbf g}(E;\phi_i).
$$

Distinct reflections add as intensities (claim). So any partition of a row
set into batches $\mathcal O=\bigsqcup_b \mathcal O_b$ gives
$\sum_{\mathbf g\in\mathcal O}I_{\mathbf g}=\sum_b I_{\mathcal O_b}$. Write
every scorer call $c$ as a pair (rows $S_c$, roll $\phi_c$) with spectrum
$v_c$. Then

$$
I=\sum_c w_c\,v_c,\qquad
w_c=\begin{cases}1 & S_c\subseteq\mathcal B\\ 1/N & S_c\subseteq\mathcal O\end{cases}
$$

The weight depends only on which class the call belongs to, not on the batch.
For any functional $\mathcal F$ that is linear in the spectrum,
$\mathcal F[I]=\sum_c w_c\,\mathcal F[v_c]$.

## 2. The audit quantities are linear in line weight

Each scored line $\ell$ has a resonance $E_\ell$, a sinc width parameter
$a_\ell$ and a production coefficient $c_\ell$. Its lineshape
$c_\ell\,\mathrm{sinc}^2(a_\ell(E-E_\ell))$ has total mass
$m_\ell=c_\ell\,\pi/a_\ell$. A spectrum call is linear in $\{c_\ell\}$.
Scaling the call by $w$ maps $c_\ell\to w c_\ell$ and leaves $E_\ell$ and
$a_\ell$ unchanged. For an axis $[S_-,S_+]$, the one-sided tail fraction is
$t_\ell(S)=\min\!\big(1,\,1/(\pi a_\ell (S-E_\ell))\big)$ for $S>E_\ell$,
and $1$ otherwise. This fraction depends on $(E_\ell,a_\ell)$ only. The
audit's sums are then:

$$
M=\sum_\ell m_\ell,\quad
A=\sum_\ell m_\ell t_\ell(S_+),\quad
L=\sum_\ell m_\ell t_\ell^{-}(S_-),\quad
A_k=\sum_\ell m_\ell t_\ell(S_k),
$$

$$
\mathcal M_{\rm band}=\sum_\ell c_\ell\!\int_{\rm band}\!\mathrm{sinc}^2(a_\ell(E-E_\ell))\,dE,\qquad
\mu_e=\sum_{\ell\in e} m_\ell .
$$

Each of these is homogeneous of degree one in $c_\ell$ and additive over
lines. So each is linear in the call weight, and the folded value
$\sum_c w_c Q_c$ equals $Q[I]$. The collected population
$\{(E_\ell,\pi/a_\ell,c_\ell)\}$ is a list of lines. The population of $I$ is
the union over calls of $\{(E_\ell,\pi/a_\ell,w_c c_\ell)\}$, so only the
coefficient column is scaled.

The derived gate quantities are not linear:
$f_+=A/M$, $f_-=L/M$, the extension stop $\min\{k: A_k/M\le\epsilon\}$, the
yield statistic $\mathrm{std}(\mu)\sqrt n/\sum\mu$, the maximum electron share
$\max_e\mu_e/\sum\mu$ and the measured stop (an edge chosen on the cumulative
weighted mass of the population). They are correct for $I$ only if they are
computed *after* folding, from the folded sums and the folded population. They
must not be folded themselves. They are also invariant under one overall
normalization ($1/N_e$), so un-normalized folding is enough.

## 3. Ceiling

The line kernels solve
$E_{\rm res}=\hbar c\,(\mathbf v\!\cdot\!\mathbf g)/(1-\hat{\mathbf n}\!\cdot\!\mathbf v)$.
With $\lvert\mathbf v\!\cdot\!\mathbf g\rvert\le\beta\lvert\mathbf g\rvert$
and $1-\hat{\mathbf n}\!\cdot\!\mathbf v\ge1-\beta$, the bound is

$$
E_{\rm res}\le\frac{\hbar c\,\beta\,\lvert\mathbf g\rvert_{\max}}{1-\beta}.
$$

The roll is a proper rotation about $\mathbf c^*$, so
$\lvert R_z(\phi)\mathbf g\rvert=\lvert\mathbf g\rvert$. The bound over the
domain's $\lvert\mathbf g\rvert$ therefore holds for every roll. It also holds
for every electron direction, which is what multiple scattering requires.
$\beta$ is evaluated at the largest incident energy. The bound is not a
recoil-exact one; a separate clamp at the beam energy handles energy
conservation.

## 4. Limiting cases

- $N=1$: one non-basal call per batch at weight $1$, at
  $\phi_0+\pi/6$. The fold is the identity, so the audit is that single
  orientation's ordinary audit.
- All-basal domain ($\mathcal O=\varnothing$): only weight-$1$ calls occur.
  The folded sums equal the ordinary basal audit of the same rows.
- Basal (non-fiber) case: no new code path is taken.

## 5. Comparison with the implementation

| Item | Derivation | Code | Result |
|---|---|---|---|
| Call weight | $w_c=1$ basal, $1/N$ otherwise, batch independent | `FiberReflectionSet._average` passes `1.0 / rolls` and adds `value / rolls`; `rolls` is 1 for basal and `azimuth_count` otherwise, and is the same for every batch | match |
| Fresh audit per call | each $Q_c$ is measured alone | `_roll_audit` copies `start_eV, stop_eV, band_eV, ceiling_eV, extension_stops_eV, n_electrons, mass_only` | match |
| Folded sums | $M,A,L,A_k,\mathcal M_{\rm band},\mu_e$ | `_AUDIT_SUMS` = `line_mass, mass_above, mass_below, mass_above_extension, band_mass, electron_mass` | match; this is every sum `_accumulate_edge_truncation` writes |
| Population | scale the coefficient column only | `(energy, width, weight * coefficient)` | match (`collect_lines` stores `(E_r, pi/a, weight)`) |
| Non-linear gates | computed after folding | `check_cxr_truncation`, `cxr_yield_statistics`, `extension_stop` and `case_cxr_stop_eV` read only the folded sums and population | match |
| Ceiling | $\hbar c\beta\lvert\mathbf g\rvert_{\max}/(1-\beta)$ over the domain rows | `kinematic_cxr_stop_eV` over `FiberCXRModel.prepare().hkls` (`fiber_line_axis`) | match |

All audit keys that exist were enumerated from their writers
(`_kernels.py::_accumulate_edge_truncation`, `_device_sizing.py::collect_lines`,
`_attribution.py`). No audit field is a maximum, a count or a ratio. The
remaining fields are `attribution`, which `_roll_audit` refuses, and
`device_bytes`, which is device-side state and is not folded. Copying `mass_only`
is harmless because line roulette is refused for fiber cases.

`device_bytes = None` on the caller is safe. Per-roll audits carry no budget,
so their chunks are host arrays. The fold appends them directly to the
caller's list without calling `collect_lines`. Clearing the caller's budget
keeps the whole population on the host. That is the "wholly on one side"
invariant `collect_lines` documents, and `join_resident_chunks` returns host
chunks unchanged. The cost is that fiber populations lose device residency.

Refusing `attribute` is necessary. Attribution records the heaviest lines and
per-electron masses as un-weighted records appended per call. The fold would
neither weight them nor merge them across rolls.

Bin-mean quadrature is consistent. `fiber_line_axis` forces it when the
per-call policy is unset. `_line_grid_case_keys` copies it into
`line_quadrature`. A user-set `quadrature="node"` survives and is then refused
by `fiber_emission._refusal`, so the route fails closed.

Skipping stored rows is correct. Those rows were derived for the basal
`hkl_list` and its lower ceiling. `basal_rows` is false for fiber sweeps, so
neither the stored row nor the bare-sweep `grid_model` tag is applied.

The cache key is correct. `_policy_cache_key` adds `cxr_fiber` only for
measured or node-local policies, and only when the key is present. Basal keys
are therefore byte-identical, and fiber records that differ in cutoff, $N$,
temperature or thermal source give different keys. The execution-only batch
size is not keyed, which is correct by §1.

## 6. Numerical checks

The probes are scratch files outside the repository. Both ran under
`pyrite-dev test` on HOPG with a 5000 Å slab, a 15° tilt and BremsLib.

- Manual quadrature, independent of `_average`, at 30 keV, cutoff 4 1/Å,
  $N=3$, 3 electrons: basal rows once plus three rolls of non-basal rows
  through the plain basal route, with weights applied by hand. The
  difference from the fiber route was $6.9\times10^{-18}$ (spectrum,
  peak-relative), and the folded `line_mass` agreed exactly.
- Mass closure: $\int I\,dE$ over $[100, 8000]$ eV divided by folded
  `line_mass`$/N_e$ was $0.99945$. The deficit lies inside the audited
  out-of-axis bounds ($1.0\times10^{-3}$ below and $8.5\times10^{-5}$ above).
  `electron_mass` summed to `line_mass`.
- Tail bound: on an axis cut at 2500 eV the true share above the cut, from the
  wide spectrum, was $2.97\times10^{-4}$. The audited `upper_fraction_bound`
  was $6.76\times10^{-4}$, so the bound holds.
- Ceiling, cutoff 4 1/Å: 34 rows (4 basal), $\lvert\mathbf g\rvert_{\max}=3.745$ 1/Å,
  and the roll mate's $\lvert\mathbf g\rvert$ agreed to $4.4\times10^{-16}$.
  The independent bound was 3613 eV at 30 keV and 25678 eV at 300 keV. The
  policy stops were 3700 eV and 25700 eV.
- End to end, 20 electrons, $N=2$:
  - 30 keV: the axis stops at the 3700 eV ceiling with an upper share of
    $2.94\times10^{-4}$. It is capped, so the route issues the ceiling-tail
    warning.
  - 300 keV: the measured stop is 11500 eV with an upper share of
    $5.0\times10^{-5}$ against a limit of $10^{-4}$.

  The implementer's 11.5 keV stop and both shares are reproduced. **The
  stated 300 keV ceiling of 41 keV is not.** The ceiling at the tested
  settings is 25.7 keV.
- The repository tests pass: `test_fiber_line_grid_1489.py`,
  `test_fiber_campaign_1483.py` and `test_fiber_model_1480.py` (46 passed, 3
  slow skipped), and `test-suite identity-pins` (26 passed).

## 7. Findings

1. **Evidence error (not code).** The #1489 progress note and the hand-off
   both give a 300 keV ceiling of 41 keV. At cutoff 4 1/Å the ceiling is
   25.7 keV, which equals $\hbar c\beta\cdot3.745/(1-\beta)$ rounded up to
   100 eV. A 41 keV ceiling would need $\lvert\mathbf g\rvert_{\max}\approx6.0$ 1/Å.
   The note should be corrected.
2. **Seeding gap outside the audit.** A per-call `windows` request gives a
   kinematic-ceiling bandwidth, and its kinematic seed provider
   (`cxr_seeds.py::_kinematic_provider`) reads `case["hkl_list"]`, which holds
   the basal rows. The fiber route does not refuse this. Non-basal lines would
   sit on the maximum-spacing backbone. Bin-mean scoring keeps their yield but
   not their shape. The fix is to refuse `windows` for fiber cases or to seed
   from the domain.
3. **30 keV acceptance.** The 30 keV case meets acceptance through the
   capped-at-ceiling warning branch, not through the tolerance: its share
   $2.9\times10^{-4}$ is above $10^{-4}$. Only $\mathrm{sinc}^2$ tails lie
   above the ceiling (§3), so this is the documented policy behaviour, but the
   acceptance wording should say so.
4. **Open risks.**
   - `np.asarray` in `_average` is safe on current routes, because
     `mc_cxr_spectrum` returns `_to_cpu(...)`. A future device-returning scorer
     would raise `TypeError` under CuPy, a loud failure rather than a silent
     one.
   - Population passes cost $N$ times the non-basal rows and are collected on
     the host. Host memory therefore grows as
     $(\lvert\mathcal B\rvert+N\lvert\mathcal O\rvert)$ times the segments.
     This has not been measured at production electron counts.

## 8. Verdict

The #1489 extension is **rederived**. Every fact in §1 to §4 matches the code,
and the numerical checks agree. There is no factor, weight or convention
discrepancy. The row as a whole stays `unverified` until the #1480 and #1483
parts (projected $B$, continuum share, quadrature-error convergence) get their
own fresh-context derivation. Human sign-off belongs to #277.
