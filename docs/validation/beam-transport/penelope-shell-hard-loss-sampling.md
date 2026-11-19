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

## Adjudication

On 2026-09-25 the owner chose $p_{\rm dis}(W)/W$ (Eqs. 3.94–3.95 and 3.104) as the reference distant law for this PENELOPE-like model. It keeps sampling, moments and closed hard rates on one DCS. The model therefore intentionally differs from PENELOPE's Eq. 3.125 sampler. Adopting Eq. 3.125 later would require changing the moments, rates and sampler together. Under this choice, the checks above pass. The ledger status is `rederived`; only a human can mark it `signed-off`.

## Independent verification of the safeguarded Newton inversion (#916, fresh context 2026-11-19)

Scope: only the root finder of the transport twins (`hard_inelastic.py::_hard_cdf_density`, `_sample_hard_transfer_eV`; CUDA `_jit_shell_device.py::_hard_cdf_value`, `_hard_cdf_slope`, `_sample_hard_transfer_eV`). The distribution (Eqs. 3.76, 3.87, 3.92, 3.94, 3.96, 3.104; adjudicated $p_{\rm dis}(W)/W$ law) is unchanged by #916. Disclosure: the verifier was handed the diff with the task, so the code was seen before this derivation; the $J$ and $J'$ expressions below were derived from the source DCS shapes and checked against arbitrary-precision references that use no implementation helper.

### Derivation

The hard loss $W$ on $[L,R]$ solves $J(W)-J(L)=u\,[J(R)-J(L)]\equiv T$ with $J'(W)=g(W)>0$ the unnormalized density. Define the residual $r(W)=J(W)-J(L)-T$. Because $g>0$, $r$ is strictly increasing, $r(L)=-T<0$ for $u>0$ and $r(R)=(1-u)[J(R)-J(L)]\ge0$, so a unique root lies in $(L,R]$.

**Distant triangle.** $g=(D-W)/W$ gives $J=D\ln W-W$ and $J'=D/W-1$. $J$ is concave, $J'>0$ for $W<D$, and $J'(D)=0$. The code's $J=D\ln(w/L)-(w-L)$ differs from mine by the constant $D\ln L-L$ only.

**Møller.** The standard close DCS is

$$
\frac{F^{(-)}}{W^2}=\frac1{W^2}+\frac1{(E'-W)^2}-\frac{1-a}{W(E'-W)}+\frac{a}{E'^2},\qquad a=\Big(\frac{E}{E+m_ec^2}\Big)^2,\ E'=E+U,
$$

(the $a$ built from $E$ and the $W$, $E'-W$ terms from $E'$ is the convention of the rederived `penelope-shell-soft-hard-partition`; #916 does not touch it). Partial fractions $1/[W(E'-W)]=(1/E')(1/W+1/(E'-W))$ give

$$
J^{(-)}_0=-\frac1W+\frac1{E'-W}+\frac{1-a}{E'}\ln\frac{E'-W}{W}+\frac{aW}{E'^2}.
$$

**Bhabha.** Integrating $F^{(+)}/W^2$ (Eq. 3.91, $x=W/E$) term by term gives $J^{(+)}_0=-1/W-(b_1/E)\ln W+b_2W/E^2-b_3W^2/(2E^3)+b_4W^3/(3E^4)$ and

$$
J^{(+)\prime}_0=\frac1{W^2}-\frac{b_1}{EW}+\frac{b_2-b_3x+b_4x^2}{E^2}.
$$

The code's `slope` is $-b_1/W+(b_2-b_3x+b_4x^2)/E$, and it returns $1/W^2+\text{slope}/E$, which is the same. Every sign, power and $1/E$ factor matches.

**Safeguarded Newton.** The iterate is $W\leftarrow W-r(W)/J'(W)$, which needs $J'>0$. $F^{(-)}>0$ and $\min F^{(+)}=0.267$ (see `bhabha-close`) make that true on the interior. The triangle has $J'>0$ for $W<D$, and the interior satisfies $W<R\le D$. Bracket invariant: $[\text{lo},\text{hi}]$ starts as $[L,R]$ with $r(\text{lo})<0\le r(\text{hi})$. Each evaluation sets $\text{lo}=W$ if $r<0$ and $\text{hi}=W$ if $r>0$, which preserves the sign change because $r$ is monotone. A Newton step outside the open bracket is replaced by the midpoint, so the bracket shrinks at least as fast as bisection. Termination: the cap of 200 steps equals the old bisection cap, and each step either halves the bracket or is a Newton step into it. A step with $|\Delta W|\le2^{-50}W$ (about 4 ulp) returns, and so does a collapsed bracket (midpoint equal to an endpoint, which is still inside $[\text{lo},\text{hi}]$).

**Limits.**

- $u=0$ gives $T=0$, so `not target > 0` returns $L$ exactly.
- $u\to1$ gives $T\to J(R)-J(L)$ and the iterate converges to $R$ (or to its round-off root).
- A degenerate interval $R\le L$ returns $L$ before any evaluation.
- Conduction-band distant ($U=0$, branch $\ne2$) returns $W_{cb}$ before any evaluation.
- The initial guess for close losses, $1/W=1/L-T$ from the leading $-1/W$ term, falls back to $R$ if $1/L-T\le0$. It is only a starting point because the bracket test guards it.

### Numerical evidence (scratch scripts, not committed)

- $J,J'$ against SymPy antiderivatives evaluated at 60 digits (mpmath) from my own $F^{(-)}$ and $F^{(+)}$: 825 points over $E=2\,{\rm keV}$–$10\,{\rm MeV}$, $U\in\{0,99,1839,69525\}$ eV, all six branches, $W_c\in\{50,1000\}$ eV. The largest $|J-J_{\rm ref}|/\text{range}$ is $2.2\times10^{-14}$ and the largest $|J'-F/W^2|/|F/W^2|$ is $1.5\times10^{-14}$.
- Roots against 220-step 60-digit bisection of my own $J$: 1925 cases, $u\in\{10^{-12},10^{-6},0.1,0.5,0.9,1-10^{-6},1-10^{-12}\}$. The worst Newton relative error is $3.7\times10^{-10}$, at the distant law's flat top ($u=1-10^{-12}$). The old bisection is $1.3\times10^{-9}$ for the same case. So Newton is at least as accurate as the retained oracle.
- 60 000 random cases from 30 eV to 10 GeV with $u$ down to $10^{-16}$ and up to nextafter(1, 0): zero out-of-bracket or non-finite results. The largest Newton–bisection difference is $2\times10^{-7}$ relative (Bhabha, $E\sim10^{10}$ eV, $1-u\sim10^{-16}$); there Newton is $6.7\times10^{-8}$ from the 80-digit root and bisection $1.4\times10^{-7}$, so both are limited by FP64 conditioning of $J$, not by the algorithm.
- Repository tests `test_hard_loss_newton.py` and `test_shell_partition.py`: 103 passed (1.8 s).

### Device twin (read only; CUDA not runnable here)

The device sampler follows the host line for line: the same bounds inlining, kind selection (0 triangle, 1 Møller, 2 Bhabha, from branch and positron tag), $J(L)$, $T$, `not target > 0` return, initial guess, bracket test, loop body, residual-zero return, bracket update, Newton step with midpoint fallback, termination test and 200-step cap (`_NEWTON_STEPS`, `_NEWTON_RTOL` are imported, not duplicated). `_hard_cdf_value` and `_hard_cdf_slope` repeat the host expressions term by term and in the same order. There are two benign rounding-order differences, so the twins are not bitwise identical: host `b4 * x**3 / 3` against device `b4*x*x*x/3`, and host `** 2` against explicit products (which are equal). Both are at the ulp level. The device `peak_end` is zero for close branches and is unused there, as on the host. Not checked: CuPy JIT compilation and run-time behavior (early `return` inside `while`, variable scoping of `inverse`) and CUDA agreement of the sampled $W$. The existing hardware parity checks must cover that.

### Verdict

$J$, $J'$, bracket maintenance, termination and the $u=0$, $u\to1$, degenerate-interval limits match the independent derivation. The distribution is unchanged, so the earlier `rederived` verdict carries over. Suggested ledger edit: keep `rederived` and replace the pending-verification sentence with a reference to this section. Open: CUDA execution of the twin. Only a human may mark `signed-off`.
