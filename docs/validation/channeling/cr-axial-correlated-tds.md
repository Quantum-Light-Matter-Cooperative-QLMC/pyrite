# `cr-axial-correlated-tds`: correlated thermal displacements within an axial string

**Status:** task-owner derivation (2026-11-11, #1045); fresh-context verdict
`rederived` (below), anchored by the regression test. Code: `src/pyrite/montecarlo/channeling/axial_tds_correlation.py`.
Research path: `checks/axial_outgoing_maps.py::material_fiber_channels(correlated_partners=…)`
and `checks/axial_fiber_source.py` (`correlated_partners`, default 0).

## Why

`cr-axial-continuum-tds-jumps` treats every string atom as an independent
scatterer: the per-length dissipator is $\lambda$ times the single-atom
one. Real thermal displacements of neighbouring atoms in one string are
positively correlated. Andersen et al. (1982, Sec. 6) note that a perfectly
correlated pair scatters twice as strongly as two independent atoms, and
that correlations up to the third string neighbour raise incoherent
scattering in Si by 25–40%. The #992 widths sit 14–24% below Andersen's
Table 2 measurements.

## Derivation

**Model.** Atoms $i$ of one string sit at $z_i$ with transverse displacements
$u_i$. These are jointly Gaussian with

$$
\langle u_{i\alpha}u_{j\beta}\rangle=a^2c_{i-j}\,\delta_{\alpha\beta},\qquad c_0=1,
$$

which is isotropic in the transverse plane. That holds for the 4-fold
⟨100⟩ and 3-fold ⟨111⟩ string axes. The single-atom phase fluctuation at
transfer $t$ is $c(t)w_t(u_i)$, with
$w_t(u)=e^{-it\cdot u}-e^{-a^2t^2/2}$, as in `cr-axial-continuum-tds-jumps`.

**Sudden approximation.** The existing model treats each collision as a
transverse multiplication operator and keeps no longitudinal transfer
$q_z$. Summing over the atoms of a segment, the per-length jump term becomes

$$
\lambda\sum_j\mathbb E\big[f_0\,\rho\,f_j^\dagger\big],
$$

with axial phases $e^{iq_z(z_0-z_j)}$ set to 1. This is valid while
$q_z|z_0-z_j|\ll1$. For 2p→1s-type transfers in Si at 4 MeV,
$q_z=\Delta E_\perp/\hbar v\approx0.02$–0.05 Å⁻¹, and over the retained
partners ($|\Delta z|\lesssim27$ Å) the phase stays below about 1 rad.
The partner sum is therefore truncated at the coherently scattering range,
about $1/(\alpha Z)\approx10$ string atoms (Andersen Sec. 6). The truncation
is the model's stand-in for the dephasing that grows with $|\Delta z|$.

**Pair kernel.** For partners with correlation $c$,

$$
\mathbb E\big[e^{-it\cdot u_0}e^{it'\cdot u_j}\big]=e^{-a^2(t^2+t'^2)/2+c\,a^2\,t\cdot t'}
\;\Rightarrow\;
\mathbb E\big[w_t(u_0)\,w_{t'}(u_j)^*\big]=\chi_t\chi_{t'}\big(e^{c\,a^2\,t\cdot t'}-1\big),
$$

with $\chi_t=e^{-a^2t^2/2}$. Setting $c=1$ gives the self kernel
$e^{-a^2|t-t'|^2/2}-\chi_t\chi_{t'}$ of the independent model.

**Exact Kraus form.** Take $0\le c\le1$, a common mode $v\sim N(0,c\,a^2I)$
and

$$
w^{(c)}_t(v)=e^{-(1-c)a^2t^2/2}\big(e^{-it\cdot v}-e^{-c\,a^2t^2/2}\big).
$$

Then
$\mathbb E_v[w^{(c)}_tw^{(c)*}_{t'}]=e^{-(1-c)a^2(t^2+t'^2)/2}\big(e^{-c\,a^2|t-t'|^2/2}-e^{-c\,a^2(t^2+t'^2)/2}\big)=\chi_t\chi_{t'}(e^{c\,a^2\,t\cdot t'}-1)$.
The pair kernel is therefore exactly the independent-string kernel of a
string with

- phase $\varphi_c(q)=\varphi(q)\,e^{-(1-c)a^2q^2/2}$,
- displacement variance $c\,a^2$ per axis, and
- linear density $m\lambda$, where $m$ is the number of partners per atom at
  that $c$ (2 for $\pm j$ on a uniform string).

It is a sum of positive-semidefinite terms. In the Taylor series,
$(t\cdot t')^n$ is a PSD kernel and $c^n\ge0$. Each shell is therefore a
completely positive channel, and so is any truncated sum of shells. Gain
and loss come from one kernel, so the loss is that string's
$\Gamma_{ab}=m\lambda e^{i(g_b-g_a)\cdot r}D_c(|g_a-g_b|)/(A(2\pi)^2)$, with

$$
D_c(G)=\int d^2k\,\varphi(|k-\tfrac G2|)\varphi(|k+\tfrac G2|)\,e^{-a^2(k^2+G^2/4)}\big(e^{c\,a^2(k^2-G^2/4)}-1\big).
$$

This is `overlap_transform` evaluated with $(\varphi_c,ca^2)$. The sampler,
displacement law, box decision and convolution of
`cr-axial-continuum-tds-jumps` apply unchanged, because each shell is one
more `TdsString`. $|\varphi_c|$ is nonincreasing whenever $|\varphi|$ is.

**Tail records.** A tail event from a shell string records the common
mode $v$ as its displacement. The tail handoff uses that displacement only
as a classical launch offset, $v\sim N(0,ca^2)$, which lies within the
thermal spread.

**Limits.**

- $c=0$ (or $m=0$): the kernel vanishes. `correlated_strings` returns the
  input string alone, so the model is bitwise the uncorrelated one.
- $c=1$, $m=1$ (a perfectly correlated pair): $\varphi_1=\varphi$ and
  $a_1^2=a^2$, so $\Gamma_{\rm pair}=\Gamma_{\rm self}$ exactly and the
  scattering doubles (Andersen Sec. 6).
- Small $c$: the pair kernel is $\approx c\,a^2(t\cdot t')\chi\chi'$,
  linear in $c$, but below $c\times$ the self kernel because
  $e^{x}-1>x$. This is consistent with Andersen's
  $1+0.7(C-1)$ rather than $C$.
- Negative $c$ is refused: the shell kernel would not be PSD on its own.

## Si correlation input

Primary source: Nielsen & Weber, J. Phys. C 13, 2449 (1980) (Zotero
`PQ7CVU28`, scanned PDF). It gives transverse correlation functions from the
adiabatic bond-charge model (BCM) at 300 K. Their Fig. 3(b)/4(b) quantity
$\langle\Delta u_\perp^2\rangle/\langle\Delta u_\perp^2\rangle_{\rm uncorr}$,
Eq. (15b), equals $1-c$ for a partner on the string, since the separation
vector is then the string axis. Values were digitized from a 400 dpi render
(frame-calibrated pixels, ±0.005):

| Partner (a/4 units) | String | $1-c$ (300 K) | $c$ |
|---|---|---|---|
| (111), √3 | ⟨111⟩ 1st | 0.760 (Fig. 3b at 293–300 K: 0.761–0.760) | 0.240 |
| (400), 4 | ⟨100⟩ 1st | 0.929 (Fig. 3b: 0.930) | 0.071 |
| (800), 8 | ⟨100⟩ 2nd | 0.972 | 0.028 |

Further 300 K points from Fig. 4(b) are not string partners: (220) 0.105,
(311) 0.086, (331) 0.075, (224) 0.062, (440) 0.054, (551) 0.038,
(660) 0.030. Together with (400) and (800) they fit
$c=0.381\,r^{-1.186}$ ($r$ in $a/4$, 8% relative rms). N&W state that the overall
decay roughly follows the Debye $1/r$ law. String partners that N&W did not
compute use this envelope:

- ⟨100⟩: $3a,4a,\dots$
- ⟨111⟩: $3\sqrt3,4\sqrt3,\dots$ in $a/4$; the string has alternating spacings $\sqrt3a/4$ and $3\sqrt3a/4$.

**Cross-check with Andersen Table 2.** Over the six nearest string atoms
(Andersen's definition) this input gives $C=1.238$ for ⟨100⟩ (Andersen:
1.25) and $C=1.420$ for ⟨111⟩ (Andersen: 1.40). Andersen's ref. [19] is
Nielsen & Weber (1980). Ten partners (≈ $1/\alpha Z$) give 1.288 and 1.480.
The Debye model is not used: N&W and Andersen both note that it overstates
transverse correlations in Si: the 300 K Debye curve of Fig. 4(b) lies below 0.7 at (111) ($c>0.3$) against the BCM 0.24.
The input is for room temperature only.

**Effect on Si rates (4 MeV, window M = 6; plane wave and the 1s and 2p states at κ = 0).**

| Axis, partners | Plane wave | 1s | 2p | $K_{1s}+K_{2p}$ | Andersen $1+0.7(C-1)$ |
|---|---|---|---|---|---|
| ⟨100⟩, 6 | +14.1% | +10.2% | +15.1% | +11.2% | +16.7% (C = 1.238) |
| ⟨100⟩, 10 | +17.0% | +12.3% | +18.3% | +13.6% | |
| ⟨111⟩, 6 | +25.7% | +19.9% | +27.5% | +21.8% | +29.4% (C = 1.420) |
| ⟨111⟩, 10 | +29.2% | +22.6% | +31.3% | +24.7% | |

(`scratch` script: `StringChannel.rate` summed over the channels of
`material_fiber_channels(..., correlated_partners=n)`.) Tightly bound 1s
gains less than the plane-wave rate. Its transfers are large, so $e^{x}-1$
is far from linear there.

## Evidence

All in `tests/montecarlo/quantum/test_channeling_axial_tds_correlation.py`:

- **Zero-correlation bitwise limit.** `correlated_strings` returns the
  identical object, and a 300-history string unraveling with a zero shell
  gives event reprs identical to the uncorrelated run.
- **Perfect-pair doubling.** The $c=1$, $m=1$ shell has a loss array equal
  to the self loss, and the total rate is $2\times$ the independent rate
  (1e-15).
- **Pair kernel closed form.** For a Gaussian phase,
  $D_c(G)=\varphi_0^2e^{-(B+a^2/2)G^2/2}[e^{-ca^2G^2/4}\pi/(2B+a^2-ca^2)-\pi/(2B+a^2)]$.
  The shell string's `overlap_transform` and full `StringChannel.loss`
  Toeplitz match it to 1e-9 for $c=0.05,0.3,0.7$.
- **Explicit-atom independent sum.** A 16-atom ring with circulant
  covariance ($c_{1,2,3}=0.35,0.15,0.05$) is sampled by Cholesky, 200k
  configurations. The estimate of $\mathbb E[F_tF_{t'}^*]/N$, with
  $F=\sum_iw_t(u_i)$, equals the summed self and shell kernels within
  $5\sigma$ at three non-collinear transfers. The self kernel alone is
  rejected (>20σ).
- **Si input.** The six-partner $C$ is within 0.025 of Andersen Table 2. The
  nearest shells are N&W's direct values.
- **Refusals.** Other axes, negative partner counts, partner counts beyond
  the enumerated string range, and $c\notin[0,1]$.
- **Injected faults, all detected:** shell damping $(1-c)\to c$; shell
  variance $ca^2\to a^2$; partner multiplicity dropped; zero shells not
  skipped; and a perturbed N&W (400) value.

## Limitations

- Sudden approximation: $q_z\to0$ inside the truncated partner range. The
  range (default 10 atoms in the research runs; 6 for Andersen's $C$)
  stands in for longitudinal dephasing.
- Intra-string correlations only. Correlations between neighbouring strings
  are omitted (optional in #1045).
- Si at 300 K only (N&W BCM); no temperature dependence.
- Correlation is isotropic in the transverse plane, and $c\ge0$.

## Fresh-context verification (2026-11-11)

Independent verifier in a fresh context, at commit `c29a7339f`. The
derivation was written first, from the ledger row, the docstrings,
Andersen et al. (1982) Sec. 6 and Table 2, and Nielsen & Weber (1980)
Eq. (15b) and Figs. 3(b) and 4(b). Only then were the implementation bodies,
the wiring and the task-owner derivation read. No files were edited.

- **Claim**: `cr-axial-correlated-tds`, `montecarlo/channeling/axial_tds_correlation.py::{PairCorrelation,correlated_strings,silicon_string_correlations}`.
- **Filters**: units pass; limits pass; signs and conventions pass.
- **Re-derivation**: `matches`.
- **Verdict**: `rederived`. The anchor test is green (12 passed), so the
  claim moves to `anchored`. Human sign-off remains in #277.

### Independent derivation

**Pair kernel.** In the sudden approximation the string phase fluctuation is
$\delta V(t)=\varphi(t)e^{-it\cdot\mathbf r}\sum_jw_t(u_j)$. The Born–Markov
per-length kernel is
$\lambda\varphi\varphi'e^{-i(t-t')\cdot\mathbf r}\sum_j\mathbb E[w_t(u_0)w_{t'}(u_j)^*]$,
summed over ordered pairs, so a shell with $m$ partners per atom counts $m$
times. With $\mathrm{Var}(t\cdot u_0-t'\cdot u_j)=a^2(t^2+t'^2-2c\,t\cdot t')$,
the pair term is $\chi_t\chi_{t'}(e^{ca^2t\cdot t'}-1)$.

**Common-mode form.** Take $v\sim N(0,ca^2)$ and
$w^{(c)}_t=e^{-(1-c)a^2t^2/2}(e^{-it\cdot v}-e^{-ca^2t^2/2})$. These
reproduce the pair term exactly. Setting $t=k-G/2$ and $t'=k+G/2$ gives the
stated $D_c(G)$, which is `overlap_transform` of $(\varphi_c,ca^2)$. Gain
and loss come from one Kraus family.

**Complete positivity.** For $c\ge0$ the shell is a Schur-product series
of PSD kernels. For $c<0$ the diagonal is negative, so the refusal is
correct. For $c>1$ the pair covariance is not PSD.

**Limits.**
- $c=0$ gives zero.
- $c=1$, $m=1$ gives Andersen's factor of two.
- Small $c$ gives a contribution linear in $c$.

**Si data.** N&W Eq. (15b) gives $1-c$ for string partners, and Andersen's
ref. [19] is N&W.

**Diamond geometry.** Explicit lattice enumeration in $a/4$ units gives
$r^2$ (partners):

- ⟨100⟩: 16 (2), 64 (2), 144 (2), 256 (2), …
- ⟨111⟩: 3 (1), 27 (1), 48 (2), 75 (1), 147 (1), 192 (2), 243 (1), 363 (1).

The sets are the same from A and B sites and for equivalent axes.
`_string_partners` matches this enumeration for $r^2\le400$.

### Comparison with the implementation

All items match.

- Shell phase $\varphi e^{-(1-c)a^2q^2/2}$: a default-argument closure, so no late binding.
- Variance $ca^2$, density $m\lambda$, same position: `dataclasses.replace`.
- $c=0$ or $m=0$ returns the input object.
- Refusals.
- Loss through `overlap_transform`.
- N&W values and envelope.
- Wiring (`checks/axial_outgoing_maps.py`): the same shells go on every cell string; [110] and `tds="lattice"` are refused; the default 0 builds the strings as before.
- Tail-launch indices stay consistent (`self.strings = channels.tds_strings`).

### Independent numerical evidence

Own scripts, in the session scratchpad (`v/check_*.py`).

| Check | Result |
|---|---|
| Shell `StringChannel.loss` vs own adaptive `dblquad` of the original pair kernel ($\varphi=1.7(1+q^2/2)^{-3/2}$, $a^2=0.05$, off-centre string, $g=(0.8,1.25)$ Å⁻¹; $(c,m)$ = (0.05,2), (0.3,2), (0.7,1), (0.95,3)) | max rel. 1.0e-14 to 1.4e-15 (self 2.0e-9) |
| Total rate, self + shell, random window state | rel. 2e-10 to 8e-10 |
| PSD of the shell Γ | $c=0.05$–0.95: min eigenvalue > 0; $c=-0.3$: all negative (refusal justified) |
| $c=1$, $m=1$ | loss `array_equal`; rate ratio exactly 2 |
| Explicit 25-atom chain, ⟨111⟩ alternating geometry, OU correlation $e^{-\Delta z/3\,\text{Å}}$, Cholesky, A/B average, vs the sum of 13 shell kernels | $\lvert z\rvert\le1.48$ (1M and 2.5M samples, three transfer pairs including $t\cdot t'<0$); self alone rejected at 31–189σ |
| Si $C$, six partners, own enumeration and digitization | ⟨100⟩ 1.238; ⟨111⟩ 1.4199 (Andersen 1.25, 1.40); ten partners 1.288, 1.480 |
| Wiring, 4 MeV plane-wave rate ratio, six partners | ⟨100⟩ 1.1409, ⟨111⟩ 1.2568 |
| Own N&W digitization (300 dpi): (111) / (400) / (800) | $c$ = 0.239–0.240 / 0.070–0.071 / 0.028–0.029; own envelope $0.382r^{-1.186}$ agrees with the code's within ≤1e-4 at every envelope partner |

### Findings (none blocking), and how they were resolved

1. A tail record from a shell string carries the common mode $v$, not an
   atom position. This is now stated in the module docstring and the ledger
   Notes. Its effect is at most a sub-$a$ launch offset.
2. `silicon_string_correlations` silently capped `partners` at the
   enumerated range (24 for ⟨100⟩, 48 for ⟨111⟩). It now refuses, and the
   anchor test covers the refusal.
3. The sudden-approximation truncation is adequately stated. The farthest
   of ten partners reaches about 1 rad of $q_zz$, but has $c\le0.03$.
4. Wording: the source of Andersen's $C$ is now definite (ref. [19]); the
   0.7 factor is now "consistent with"; the envelope rms is now 8%.
5. Negative $c$ is a representational limit only. All N&W string-partner
   values are positive.
