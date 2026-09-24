# PENELOPE shell hard-loss sampling: independent verification

Validation: `penelope-shell-hard-loss-sampling`. Verdict: **discrepancy in the cited distant-loss sampling convention**. This verifies the host-side loss and energy ledger, not recoil angles, event scheduling, or relaxation.

## Sources, quantity, and assumptions

The [PENELOPE-2024 manual](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf), §§3.2.2–3.2.5, Eqs. 3.76, 3.87, 3.94–3.96, 3.124–3.125, supplies the shell differential cross sections and hard-event procedure. The [Geant4 Penelope reference](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/electron_incident/ionisation/penelope_ionisation.html) independently states the inner- and outer-shell secondary convention. The code anchor is `src/pyrite/montecarlo/transport/shell_sampling.py::sample_shell_hard_loss`.

Inputs are a material oscillator set, its closed soft/hard partition, two uniforms in $[0,1)$, and a separate secondary production threshold in eV. The result has oscillator and branch labels, energy transfer $W$ in eV, optional emitted secondary kinetic energy, local deposited energy, optional binding reserve, and an inner-shell vacancy identifier. Every sampled hard loss has $W>W_c$, including substituted inner shells. A bound-shell hard transfer is assumed to ionize its oscillator. Recoil and azimuth are outside this claim.

## Independent derivation and cheap filters

For branch $b$ of oscillator $k$, let $h_{kb}$ be its nonnegative hard zeroth-moment cross section per formula unit. PENELOPE Eq. 3.124 gives the shell sum; resolving its branch terms gives

$$
H=\sum_{k,b}h_{kb},\qquad P(k,b\mid\text{hard})=\frac{h_{kb}}{H}.
$$

This ratio is dimensionless. For a continuous branch on $[L,R]$, a uniform $\xi$ must satisfy

$$
\xi=\frac{\int_L^W g_b(x)\,dx}{\int_L^R g_b(x)\,dx},\qquad 0\le\xi<1.
$$

Equation 3.76 gives $p_{\rm dis}(W)=2(D-W)/(D-U)^2$ on $U\le W\le D$, where $U$ is binding and $D=W_{\rm dis}=\min(3W_k-2U,E)$ after the manual's near-threshold resonance adjustment. Taking the *energy-loss DCS as written in Eqs. 3.94–3.95* yields $g_{\rm dis}(W)\propto p_{\rm dis}(W)/W$ for either distant branch. The conditional primitive, with the constant prefactor cancelled, is

$$
A_{\rm dis}(W)=D\ln W-W,\qquad
\xi=\frac{D\ln(W/L)-(W-L)}{D\ln(R/L)-(R-L)}.
$$

For every bound shell, $L=\max(U,W_c)$. The upper bound is $R=\min(D,(E+U)/2)$; the last term is the electron-exchange limit of Eq. 3.88. The conduction-band distant branch is instead a point mass at its resonance $W_k$.

For the close branch, Eqs. 3.87 and 3.96 give $g_{\rm clo}(W)\propto F^{(-)}(E+U,W)/W^2$. Its physical interval is $L=\max(Q_k,W_c)$ and $R=(E+U)/2$, with $Q_k=U$ for a bound shell and $Q_k=W_k$ for the conduction band. An exact inverse CDF is the unique root of the integral equation above; an analytic antiderivative or converged root solve implements the same distribution. Every continuous sample must remain in $[L,R]$. Positivity of the DCS makes the CDF monotone.

For a substituted inner shell, the secondary has kinetic energy $T_s=W-U$ and the vacancy reserves $U$; for an outer shell or conduction band, the Geant4 Penelope convention uses the proxy $T_s=W$ and no vacancy. If $T_s$ is below the production threshold it is deposited locally. Thus, in both cases,

$$
E_{\rm local}+E_{\rm emitted}+E_{\rm reserved}=W.
$$

All four terms are in eV. The limiting checks are the conduction-band delta, $\xi=0\Rightarrow W=L$, $\xi\to1\Rightarrow W\to R$, and this energy identity for emitted and locally deposited secondaries.

## Source discrepancy and code comparison

The manual is not internally consistent about bound distant loss. Eqs. 3.94–3.95 write $p_{\rm dis}(W)/W$, but Eq. 3.125 explicitly samples $p_{\rm dis}(W)$ alone, with inverse

$$
W=D-\sqrt{(D-L)^2-\xi(R-L)(2D-L-R)}.
$$

Its earlier Eqs. 3.81–3.82 also give $p_{\rm dis}(W)/W_k$, which is proportional to the triangle rather than $p_{\rm dis}(W)/W$. The first divergent term is **$1/W$ versus a constant $1/W_k$**. For the illustrative interval $U=L=1$ eV and $D=R=3$ eV, the median from the DCS expression is $1.4344719278$ eV; Eq. 3.125 gives $1.5857864376$ eV. This difference cannot be removed by normalization.

The implementation selects flattened oscillator/branch probabilities from the partition's hard zeroth moments. Its bound distant integral uses the $D\ln W-W$ primitive, and its close integral uses the Møller $J_0^{(-)}$ primitive, each inverted over the branch interval by `brentq`. These match the ledgered DCS-based conditional laws and reproduce the partition's own zeroth moments. They **do not match the explicit PENELOPE Eq. 3.125 distant sampler**. The conduction-band distant result is its resonance. The inner and outer secondary rules, vacancy label, threshold deposition, and binding reserve match the derived energy identity. The reserve remains an obligation of issue #94's relaxation handoff.

The code's distant and close bounds implement the intervals above. Its `binding_reserve_eV` is $U$ only for substituted inner shells; `local_deposit_eV` is $T_s$ only when no secondary is emitted. The total is $W$ in either threshold case. These observations do not independently validate the upstream partition, closure factors, or EEDL rate substitution.

## Verdict and suggested ledger edit

Units, support limits, branch normalization, and energy signs **pass** against the ledgered DCS. The distant source convention **fails to reconcile**: PENELOPE Eq. 3.125 and Eqs. 3.81–3.82 imply a triangular conditional law, whereas Eqs. 3.94–3.95 and this code use the triangle divided by $W$. Suggested ledger status: `discrepancy`, with this precise source conflict in Notes. A human should adjudicate the intended reference law before advancing the status; only a human can mark `signed-off`.
