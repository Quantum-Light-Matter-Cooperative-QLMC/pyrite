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

## Re-validation (2026-10-15, fresh context, after the resolution)

**Derivation of the new bound.** Let $R=\sum_c d_c^\dagger d_c$ act on the above-barrier states of one block. It is positive semidefinite and vanishes on bound states, so for any $\rho$, including its bound–free coherences, $\mathrm{Tr}[\rho R]=\mathrm{Tr}[\rho_{FF}R]\le\lambda_{\max}(R)\,\mathrm{Tr}\,\rho_{FF}$. The density matrix is block-diagonal over Bloch blocks. Assume a dechanneled electron also sits in above-barrier basis states. Then the capture rate is at most $\max_k\lambda_{\max}(R_k)\,(1-P_{\rm bound})$, and the claimed bound follows. It is valid for the scored estimator when the bound's clusters are *coarser* than the scored ones: the pinching inequality gives $\lambda_{\max}(\text{block-diagonal part})\le\lambda_{\max}(\text{full})$ for a PSD matrix, and a principal submatrix interlaces. It is a model bound. With every free level of a final state in one cluster (no clustering) $\lambda_{\max}$ is 1.9× larger.

**Checks** (`scratch/cr505reval/capture.py`, `bound_struct.py`). Kephart configuration: Si(110), 17 MeV, 2.2 mrad, 2 µm, $dz=100$ Å, sampling every 500 Å, my own amplitudes and my own depth history.

| $n_{\rm cells}$, $H$, margin | estimate impl / own | line yield impl / own | ratio | upper bound (× line) | $\lambda_{\max}$ impl | max over $z$ of rate$/(\lambda(1-P_b))$ |
| --- | --- | --- | --- | --- | --- | --- |
| 4, 8, 0 eV | 0 / 0 | 1.6081e-5 / same | 0 | 5.134e-5 (3.19) | 3.0365e-9 | 0 |
| 4, 8, 30 eV | 8.0537e-6 / same | 2.1171e-5 / same | 0.3804 | 4.810e-5 (2.27) | 3.0365e-9 | 0.207 |
| 4, 8, 300 eV | 8.6191e-6 / same | 2.2103e-5 / same | 0.3900 | 4.752e-5 (2.15) | 3.0365e-9 | 0.211 |
| 4, 16, 300 eV | 8.8798e-6 / same | 2.2683e-5 / same | 0.3915 | 4.710e-5 (2.08) | 3.0383e-9 | 0.212 |
| 16, 8, 30 eV | 8.0052e-6 / same | 2.1093e-5 / same | 0.3795 | 4.830e-5 (2.29) | 3.0466e-9 | 0.206 |

- The estimate equals the scored capture yield (`cr-capture-yield`) and my own diagonal estimator to 5 significant figures.
- The bound holds pointwise in depth with a factor of about 5 to spare.
- `bound_struct.py` checks the dominance condition on 4 and 16 cells. No scored capture cluster straddles two runs of `_capture_rate_bound`. $\lambda_{\max}$ of the scored clustered operator is 1.000× (4 cells) and 0.997× (16 cells) of the bound's $\lambda_{\max}$.

**Caveat (no failing case found).** `capture_bound` sets the run tolerance to `lines.hwhm_eV[is_capture].max()`, the maximum of the *rate-weighted cluster means* (`emission.py:1175`). The scored links use the pair tolerance $\tfrac12(h_{\rm last}+h_{\rm item})$ of individual pairs (`emission.py:676–680`). A link between two broad pairs inside a cluster dominated by a narrow strong member could exceed the bound's tolerance, split a scored cluster and break the coarser-cluster condition. The max over pair half widths would make the bound structural. With cross terms below $2\times10^{-4}$ of the yield the numerical effect is nil here.

**Verdict:** `rederived`.
