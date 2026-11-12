# `cr-capture-yield` re-derivation

Claim: `montecarlo/channeling/emission.py::{emission_lines,EmissionLines.rates,channeling_spectrum}`. Sources: maintainer decision 2026-10-15, proposal §11 decision 4 option (b), walkthrough §9 item 2, and `cr-dipole-rate`.

Fresh-context verification, 2026-10-15. The derivation below was written from the ledger row before the bodies were read.

## Independent derivation

A tracked above-barrier state $i$ ($U_{\rm barrier}\le E_i<U_{\rm barrier}+\text{margin}$, inside the solver subspace $S$) decays radiatively into a bound $f$ through the same first-order dipole operator as a bound line. Write $a_{fi}=\sqrt{dN_{fi}/dz}\;x_{fi}/\lvert x_{fi}\rvert$ (`cr-dipole-rate`), and sum the transitions into one final state over a set $c$ of initial states, $d_c=\sum_{i\in c}a_{fi}\lvert f\rangle\langle i\rvert$. The expected photon rate per path is

$$
\langle d_c^\dagger d_c\rangle_\rho=\sum_{i,i'\in c}\bar a_{fi}\,a_{fi'}\,\rho_{i'i}
=\sum_{i,i'\in c}\bar a_{fi'}\,a_{fi}\,\rho_{ii'} .
$$

It is linear in $\rho$, so it is unbiased on any ensemble. It needs only the free–free block $\rho_{FF}$, so it vanishes when no free state is tracked. It also vanishes with no bound $f$.

In the Markov (flat-density) limit, the frequency-integrated rate operator is $\sum_f D_f^\dagger D_f$, where $D_f$ sums over *all* initial states. Cross terms between two initial states resolved in energy ($\lvert E_i-E_{i'}\rvert\gg$ width) multiply $\rho_{ii'}$. That coherence rotates at $(E_i-E_{i'})/\hbar v$ and averages out of $\int dz$. Clustering only unresolved initial energies is therefore the standard secular choice, and the residual is $O(\text{width}/\delta)$. Cross terms inside a cluster must be kept. They vanish by parity only at $k=0$ and $g/2$, as the earlier `cr-capture-bound` verification found. The photon energy is $(E_i-E_f)/D$, at or above the capture edge $(U_{\rm barrier}-E_f)/D$.

## Independent numerical checks

`scratch/cr505reval/capture.py` uses my own commutator dipoles, my own depth history from `DensityMatrixSolver.propagate`, and three estimators: diagonal, clustered within 0.5 eV, and fully coherent per $f$ (no clustering). Configuration: Si(110), 17 MeV, TDS + electronic, 2.2 mrad, 2 µm, $dz=100$ Å, sampling every 500 Å.

| $n_{\rm cells}$, $H$, margin | capture impl | own diagonal | own fully coherent | capture / line, impl | own |
| --- | --- | --- | --- | --- | --- |
| 4, 8, 0 eV | 0 | 0 | 0 | 0 | 0 |
| 4, 8, 30 eV | 8.0537e-6 | 8.0537e-6 | 8.0526e-6 | 0.3804 | 0.3804 |
| 4, 8, 100 eV | 8.6191e-6 | 8.6191e-6 | 8.6171e-6 | 0.3900 | 0.3900 |
| 4, 8, 300 eV | 8.6191e-6 | 8.6191e-6 | 8.6171e-6 | 0.3900 | 0.3900 |
| 4, 16, 30 eV | 8.1036e-6 | 8.1036e-6 | 8.1026e-6 | 0.3806 | 0.3806 |
| 4, 16, 300 eV | 8.8798e-6 | 8.8798e-6 | 8.8778e-6 | 0.3915 | 0.3915 |
| 8, 8, 30 eV | 8.0216e-6 | 8.0216e-6 | 8.0200e-6 | 0.3799 | 0.3799 |
| 16, 8, 30 eV | 8.0052e-6 | 8.0052e-6 | 8.0033e-6 | 0.3795 | 0.3795 |

The claimed 0.38–0.39 over margins 30–300 eV, $H=8$–16 and $n_{\rm cells}=4$–16 is reproduced. Margin 100 and 300 coincide at $H=8$ because that band ends below 100 eV above the barrier. With every coherence of a final state kept, the yield moves by at most $2.3\times10^{-4}$, so the choice of clustering does not affect the yield.

**Capture line shapes and the stated limit.** `scratch/cr505reval/capture_width.py` takes the four strongest capture groups at $n_{\rm cells}=4$ and 16 (margin 30 eV). For each it seeds the coherence and propagates it with the full solver, which keeps transfer into every other band-index pair. The full $G(z)$ equals the group modes within 1.8% of $\lvert G_0\rvert$ over three mode decay lengths; for example 4→3 at 16 cells gives $\lvert G\rvert/\lvert G_0\rvert$ 0.154/0.114/0.041 (full) against 0.156/0.117/0.039 (modes). The dropped transfer into other pairs is therefore small on these grids. Its sign is not definite: the summed gain $\langle i\vert L^\dagger d^\dagger L\vert f\rangle$ has no fixed sign. The ledger's "capture widths are upper estimates" is not a theorem; it should read "approximate (≤2% in $G$ on the checked grids)". Yields do not depend on the widths. No group mixes bound and capture kinds on either grid. 86 (4 cells) and 422 (16 cells) capture rows include 16 multi-member clusters. Their shape uses the strongest member's pair and amplitude. In the independent assembly this changes the capture spectrum by 2.4e-3 of its peak (`cr-yield-assembly`).

## Diff against the implementation

- `EmissionLines.rates` computes `sum(same * amp[:,None] * conj(amp[None,:]) * rho[upper, upper])` with `same` the shared-final mask, i.e. $\sum_f\sum_{i,i'}a_{fi}\bar a_{fi'}\rho_{ii'}$. This matches.
- `emission_lines` sets `initial_ok = bound | (subspace & ~bound)` and labels a transition capture when its upper state is not bound. Free→free is excluded by requiring `bound[lower]`. Clusters are consecutive runs within a block sharing `lower`, with link tolerance $\tfrac12(h_a+h_b)$ on the initial-energy difference. This matches.
- Yield and spectrum separation: `total_capture_yield` sums the `KIND_CAPTURE` rows, and `capture_spectrum` takes groups by their first member's kind. Both are consistent here because no group mixes kinds. A guard would be needed if the top band crosses the barrier between blocks.

## Verdict

`rederived`. Suggested Notes edit: replace "capture widths are upper estimates" with "approximate; full-dynamics $G(z)$ agrees within 2% on 4- and 16-cell grids".

Toctree: this page is new. `docs/validation/index.md` needs a `channeling/cr-capture-yield` entry next to `channeling/cr-capture-bound`; the verifier did not edit the index.

## Upper band straddling the barrier (#1091)

Line groups are keyed by the band pair $(u, l)$ across Bloch blocks. When band
$u$ is bound in some blocks and above the barrier in others, the group holds
bound→bound rows and capture rows. The coherence-transfer generator
$K_{k'k}=\langle f_{k'}|\mathcal J(|f_k\rangle\langle i_k|)|i_{k'}\rangle$
acts on the band-pair coherence whatever the barrier label of $i_k$, so the
rows keep their shared modes $K=V\,\mathrm{diag}(\mu)\,V^{-1}$.

The field radiated by mode $m$ is $\sum_k a_k^*\sigma_k$, so the weight
$w_m=(a^\dagger V)_m(V^{-1}s)_m$ is linear in the radiating amplitudes. For a
partition of the rows into kinds $P$, set $a_P$ equal to $a$ on $P$ and zero
elsewhere:

$$
w_m^{P}=(a_P^\dagger V)_m\,(V^{-1}s)_m,\qquad
\sum_P w_m^{P}=w_m,\qquad
\sum_m w_m^{P}=a_P^\dagger s=\sum_{k\in P}Y_k .
$$

Because each mode profile has unit area, the line and capture spectra
integrate to exactly the line and capture yields of `EmissionLines.rates`,
and their sum is the unsplit group spectrum. Before #1091 the whole group
went to the kind of its first row. At Si(110), 12.5 MeV, 4 cells, $H=8$,
300 eV margin, 1 mrad divergence and 5000 Å, groups (3,0), (3,2) and (3,1)
mix kinds. Capture was overstated by 15% and bound→bound understated by
5.7%. The total changes at rounding level only ($\le 10^{-15}$ relative).

A cluster inside one block takes its strongest member's kind and can, in
principle, merge an initial state just below the barrier with one just above
it. The anchor case has no such cluster.

### Fresh-context verification of the split (2026-11-11)

**Derivation.** By the quantum regression theorem the group spectrum per
transverse eV is

$$
S(E)=\frac{1}{\pi\hbar v}\,\mathrm{Re}\int_0^\infty d\tau\,
e^{-iE\tau/\hbar v}\,a^\dagger e^{K\tau}s
=\frac{1}{\pi\hbar v}\,\mathrm{Re}\,a^\dagger\left(\frac{iE}{\hbar v}-K\right)^{-1}s ,
$$

with $s_k=Y_k/a_k^*$. The source $\sigma_k\rho$ lives in block $k$ only, so
$C(0)=a^\dagger s=\sum_k Y_k$ has no cross terms. With
$K=V\,\mathrm{diag}(\mu)\,V^{-1}$ the resolvent is a sum of poles with
weights $(a^\dagger V)_m(V^{-1}s)_m$. Replacing $a$ by $a_P$ at the
radiating (late-time) end is linear, so $\sum_P w_m^P=w_m$ and
$\sum_m w_m^P=a_P^\dagger s=\sum_{k\in P}Y_k$. Attributing the photon to
the source end instead, $(a^\dagger V)_m(V^{-1}s_P)_m$, gives the same
per-kind integral $a^\dagger s_P=\sum_{k\in P}Y_k$. Only the per-kind line
shapes depend on the choice: cross-block terms $a_{k'}^*\,(e^{K\tau})_{k'k}\,s_k$
with $k'$ and $k$ of different kinds have no unique kind label. The
radiating-end convention credits the dipole that radiates at detection. It
is a definition, not an observable; the per-kind integrated yields are
convention-free and equal `EmissionLines.rates`.

**Shared modes.** The off-diagonal elements
$K_{k'k}=\langle f_{k'}|\mathcal J(|f_k\rangle\langle i_k|)|i_{k'}\rangle$ are
the gain from scattering kicks that move band-pair coherence between Bloch
blocks. The diagonal already carries the full loss $(\Gamma_i+\Gamma_f)/2$.
Keying groups on kind would drop the gain between kinds while keeping the
loss, which broadens the lines and changes the total spectrum. The barrier
label is an energy classification, not a dynamical boundary of $\mathcal J$.

**Numbers** (Si(110), 12.5 MeV, 4 cells, $H=8$, 300 eV margin, 293.15 K,
1 mrad, 5000 Å, $dz=250$ Å, no absorption; independent verifier
script, not committed). The three mixed groups are $(3,0)$ and
$(3,2)$ (kinds C C L C over blocks 0–3) and $(3,1)$ (C L C over blocks 1–3).

- $K$ rebuilt from `solver.jump`/`loss_rate` agrees with
  $V\,\mathrm{diag}(\mu)\,V^{-1}$ to $2\times10^{-10}$ relative. Off-diagonal
  elements reach 0.11–0.18 of the largest diagonal damping.
- Per-kind spectra from the resolvent above (no eigendecomposition) match
  `kind_spectra` to $\le 2\times10^{-10}$ of peak for every mixed group and
  kind. A direct $\int d\tau\,e^{-iE\tau/\hbar v}\,a_P^\dagger e^{K\tau}s$
  quadrature at the strongest mode agrees to $3\times10^{-5}$ (step-limited),
  which confirms the transform sign.
- $a_P^\dagger s=a^\dagger s_P=\sum_{k\in P}Y_k$ to printed precision. Over
  $\pm 2$ MeV the line and capture spectra integrate to their yields to
  $-1.2\times10^{-4}$ (Lorentzian tails outside the window), and
  line + capture equals the unsplit sum to $8\times10^{-13}$ of peak.
- Old attribution: line yield $-5.69\%$, capture $+14.70\%$, which reproduces
  the write-up.
- Source-end and radiating-end per-kind shapes differ by $\le 1.8\%$ of peak
  in $(3,0)$ and $(3,2)$.
- Of 151 rows (20 multi-member), no cluster mixes bound and free initial
  states.

**Residuals.** A per-kind part of a mixed group is not positive-definite. In
$(3,1)$ the bound row ($Y=2\times10^{-18}$) has a dispersive lobe of $-0.40$
of its own peak, but its magnitude is $3\times10^{-12}$ of the line-spectrum
peak, and the summed line and capture spectra stay non-negative. The
Gaussian $\gamma$-spread exponent is strength-weighted over the whole group
and shared by both kinds, which affects shapes only for
`sigma_gamma_rel > 0`. Cluster kind mixing is still assigned to the strongest
member. It is absent here.

**Verdict:** `rederived`.
