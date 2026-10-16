# `cr-capture-bound` re-derivation

Claim: `montecarlo/channeling/emission.py::capture_bound` (helper `_capture_rates`). Sources: proposal §11 decision 4, walkthrough §9 item 2, and `cr-dipole-rate`.

## Independent derivation (before reading the body)

Radiative capture of an above-barrier state $i$ ($E_i\ge U_{\rm barrier}$) into a bound $f$ uses the same first-order operator as a bound line, at $\hbar\omega=(E_i-E_f)/D\ge(U_{\rm barrier}-E_f)/D$. In the dipole limit its rate per path is $dN_{fi}/dz$ of `cr-dipole-rate`, and the total per initial state is

$$
\Gamma_i^{\rm cap}=\sum_{f\,\rm bound}\frac43\alpha\gamma^2\Bigl(\frac{E_i-E_f}{\hbar c}\Bigr)^3\frac{\lvert x_{fi}\rvert^2}{\beta}.
$$

The expected yield is $\int dz\,\langle\sum_f d_f^\dagger d_f\rangle$. It is diagonal only if no two free states feeding the same $f$ are degenerate within the widths. **That condition does not hold in general.** At $k=0$ and $k=g/2$ the above-barrier bands come in pairs split by small zone-centre or zone-boundary gaps; in Si(110) at 17 MeV the splittings are 0.0001–0.03 eV, far below the ~0.3 eV widths. The pairs have opposite parity at those two symmetric points, so for a centrosymmetric plane only one member of each pair has $x_{fi}\ne0$ to a given $f$ and the cross terms vanish. The diagonal form is therefore exact at symmetric $k$ by parity, not by nondegeneracy. Away from symmetric $k$ the pairs split linearly in $k$.

Estimate: $\int dz\,\sum_{i\in S,\,\rm free}\rho_{ii}(z)\,\Gamma_i^{\rm cap}$. Upper bound: every non-bound electron (free in $S$, free flux, overflow) captures at most at $\max_i\Gamma_i^{\rm cap}$ over the plane-wave basis, giving $\max_i\Gamma_i^{\rm cap}\int dz\,(1-P_{\rm bound})$. This bounds the model's capture only within the basis, and only up to cluster cross terms. By Cauchy–Schwarz a cluster contributes at most (cluster strength) × (its population), and with parity-forbidden cross terms that equals the diagonal term. With margin 0 no free state lies in $S$, so the estimate is exactly 0 while the bound stays finite. With zero bound states there is no $f$, and both are 0.

## Independent numerical checks

These were run in the Kephart configuration: Si(110), 17 MeV, TDS + electronic, $n_{\rm cells}=4$, $H=8$, 2 µm, 2.2 mrad divergence, $dz=100$ Å, sampling every 500 Å. I computed my own $\Gamma^{\rm cap}$ from the solver eigenvectors (own commutator dipoles) and my own depth history from `propagate`.

| margin | $\Gamma^{\rm cap}$ max rel. diff | estimate (impl / mine) | upper bound (impl / mine) | line yield | estimate / lines | bound / lines |
| --- | --- | --- | --- | --- | --- | --- |
| 0 eV | $1.1\times10^{-9}$ | 0 / 0 | $5.134\times10^{-5}$ / same | $1.608\times10^{-5}$ | 0 | 3.19 |
| 30 eV | $1.1\times10^{-9}$ | $8.054\times10^{-6}$ / same | $4.810\times10^{-5}$ / same | $2.117\times10^{-5}$ | **0.380** | 2.27 |
| $10^5$ eV | $1.1\times10^{-9}$ | $8.619\times10^{-6}$ / same | $4.752\times10^{-5}$ / same | $2.210\times10^{-5}$ | **0.390** | 2.15 |

Cross terms between free states within 0.5 eV, included explicitly over the whole history, change the margin-30 estimate by a factor of **1.000**: the parity cancellation holds on this grid. The free states nearest the barrier (0.34 eV above it) carry $\Gamma^{\rm cap}$ equal to 0.24 of the strongest line strength, which makes a capture-to-line ratio of tens of percent plausible at 2.2 mrad, where a large fraction of electrons enters above the barrier. The forward edge into the ground state, 37.29 keV, lies above the highest line, 35.74 keV. That check is automatic, since $U_{\rm barrier}-E_0>E_i-E_0$ for every bound $i$. Capture into the top bound level starts far lower, inside the line band, as the Notes state.

## Diff against the implementation

- `_capture_rates` forms `dipole_rate_per_ang(E_free - E_bound, x[bound, free], gamma)` and sums over bound states, which equals $\Gamma_i^{\rm cap}$.
- `capture_bound` integrates `history.capture` (diagonal $\rho$ times $\Gamma$) and `max_rate * trapz(1 - P_bound)` by the trapezoid rule. Both are reproduced to machine precision.
- The docstring justification, "Above-barrier Bloch states at one `k` are nondegenerate, so the expected value is diagonal" (`emission.py:945`), is inaccurate. The diagonal result holds by parity at the symmetric blocks $k=0,g/2$, which are the only near-degenerate ones on the coarse grid. On fine $k$ grids, near-degenerate pairs at small $k\ne0$ lose that protection. I recommend changing the wording or adding a degeneracy guard. The numbers above are unaffected.
- "Upper bound" is a model bound: it is relative to the basis states and the diagonal estimator. It is not a rigorous bound on physical capture.

## Promotion decision

The estimate is 0.38–0.39 of the line yield, about 13 times the 3% rule, so the rule fires. Scoring capture conflicts with the v1 bound→bound maintainer decision and is correctly left as an open maintainer decision.

## Verdict

`rederived`. Revise the docstring's nondegeneracy wording to cite parity.

## Resolution (worker, after the verdict)

- The docstring now says that the diagonal form holds by parity only at `k = 0, g/2` and that this protection is lost on fine `k` grids.
- Following the maintainer decision of 2026-10-15, capture is now scored (`cr-capture-yield`). Capture transitions into one final state are clustered when their initial energies are unresolved within the widths, and their cross terms are kept. `capture_bound().estimate` is now that scored yield.
- The upper bound uses `λ_max` of the clustered capture-rate operator over all above-barrier states, instead of the largest diagonal rate.
- At the Kephart configuration the ratio is unchanged (0.38–0.39). At the entry state the cross terms change the capture rate by less than 2e-9, consistent with ×1.000 above.
