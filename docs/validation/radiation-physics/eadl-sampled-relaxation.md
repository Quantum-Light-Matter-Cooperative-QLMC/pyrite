# Sampled EADL single-vacancy relaxation

Validation: `eadl-sampled-relaxation`.

## Source and independence

This derivation was written in a fresh verification context before inspecting
the implementation bodies. The initial inputs were the ledger record and
AST-extracted signatures/docstrings of `sample_relaxation_cascade`,
`_sample_transition`, and `_sampled_energy_budget`.

The primary source is [ENDF-102, File 28, §§28.1–28.3, printed pp. 227–229
(2023 manual)](https://www.nndc.bnl.gov/endfdocs/ENDF-102-2023.pdf).
Its fields specify subshell binding energy (`EBI`), daughter subshells (`SUBJ`,
`SUBK`), emitted transition energy (`ETR`), and branch weight (`FTR`).
The [April 2001 edition, File 28](https://www.oecd-nea.org/dbdata/data/manual-endf/endf102_2001.pdf)
uses §§28.2–28.4 for the corresponding description, format, and procedures.
The section-number difference is editorial. The zero value of `SUBK` identifies
a radiative transition; a nonzero value identifies two daughter holes and an
electron. Both binding and emitted energies use eV.

The intended quantity is one realized cascade starting from a specified vacancy
in validated, cascade-ordered neutral-atom data. The model uses independent
single-vacancy probabilities, retaining repeated daughter holes separately.
It omits occupancy depletion, shifts caused by multiple vacancies, directions,
lifetimes, particle transport, and deposition. Independent uniform inputs and
depth-first traversal are replay conventions supplied by this API, rather than
requirements of the ENDF format. The pinned EADL2025 tables are inputs; this
verification does not independently certify their evaluated values.

## Independent derivation

Let $B_i$ be the binding of a vacancy in subshell $i$. For its ordered transition
list let $w_{it}\geq0$ be the printed fractional probability, with positive total
$W_i$. Normalization of the printed values gives

$$
W_i=\sum_t w_{it},\qquad p_{it}=\frac{w_{it}}{W_i},\qquad
C_{it}=\sum_{s\leq t}p_{is}.
$$

For a uniform $u\in[0,1)$ select the first positive-weight transition whose
cumulative probability is strictly greater than $u$. Thus the branch interval
is left closed and right open. Zero weights have zero interval width, including
at $u=0$. Exactly one uniform is needed per decaying vacancy; traversal of a
terminal vacancy needs none. A singleton positive branch is certain.

The emitted energy $\epsilon_t$ is the tabulated `ETR`. The daughter multiset is

$$
D_t=\begin{cases}
\{j\},&\text{radiative},\\
\{j,k\},&\text{nonradiative}.
\end{cases}
$$

The multiset is essential: when $j=k$, the nonradiative event creates two holes,
each of which is subsequently propagated. Neither the probability nor the
daughter count receives a factorial correction: the table already describes
the entire transition. The radiative particle is a photon with energy
$\epsilon_t$; the nonradiative particle is an electron with kinetic energy
$\epsilon_t$. Electron rest energy is not an additional relaxation expenditure.

For the API's finite positive cutoff $E_c$, a vacancy is terminal when
$B_i\leq E_c$ or its transition list is empty. Its full $B_i$ remains reserved.
Otherwise sample its transition, then visit the first daughter and its entire
descendant tree before the second daughter. Strictly increasing daughter ranks
in a finite subshell list bound every path's depth. Even with binary splitting,
the resulting tree is finite. Tape exhaustion must fail rather than silently
turn an unprocessed branch into a terminal hole.

Binding differences need not equal the independently tabulated emission energy.
Define the signed local defect

$$
\delta_t=B_i-\sum_{d\in D_t}B_d-\epsilon_t.
$$

This is bookkeeping for the tabulated single-vacancy model, not an extra emitted
particle or a prescribed deposited energy. It may be negative; clamping would
change the stated energy identity. Rearrangement at each internal tree vertex
gives $B_i=\epsilon_t+\sum_d B_d+\delta_t$. Summing over vertices cancels each
nonterminal daughter's binding against its own parent term, leaving

$$
\boxed{B_p=\sum_{\gamma}E_\gamma+\sum_e T_e
+\sum_{v\in\mathcal T}B_v+\sum_t\delta_t.}
$$

Here $p$ is the primary vacancy and $\mathcal T$ is the multiset of terminal
vacancies. Cancellation works with repeated daughter labels because it counts
vacancy instances. Every term has units eV. Branch probabilities and uniform
inputs are dimensionless. Raising the cutoff above the primary binding gives
no emissions or defects, one terminal vacancy, and no random draws. With exact
binding-difference transition energies every defect vanishes. A zero-energy
electron transition can still have a nonzero branch probability and two holes.

## Independent numerical example

Consider bindings $B_1=100$, $B_2=30$, $B_3=10$ eV. Subshell 1 has a photon
branch to 2 of energy 69 eV and weight 0.2499999, and an electron branch to two
instances of 2 of energy 41 eV and weight 0.7499997. Subshell 2 has a certain photon
branch to 3 of energy 21 eV. Subshell 3 is terminal. Normalized primary branch
probabilities are $1/4$ and $3/4$.

For the photon branch the totals are photon energy 90 eV, terminal binding
10 eV, defect zero, and two draws. The two defects are $+1$ and $-1$ eV.
For the electron branch the totals are photon energy 42 eV, electron kinetic
energy 41 eV, terminal binding 20 eV, defect $-3$ eV, and three draws. Its defects
are $-1$, $-1$, and $-1$ eV. Both paths sum to 100 eV. The analytic mean photon
energy is 54 eV, mean electron energy 30.75 eV, mean terminal binding 17.5 eV,
and mean defect $-2.25$ eV. Their sum is again 100 eV. Equality at primary
uniform $1/4$ belongs to the electron branch.

## Implementation comparison

Body inspection followed the written derivation. All three implementation
owners contain the matching validation marker. The comparison finds:

- `_sample_transition` concatenates radiative then electron probabilities,
  accumulates the unnormalized weights, and compares against the draw times
  their total. This is algebraically the normalized CDF above. Right-sided
  search implements the strict upper bound, skipping zero-width intervals.
  Clipping the threshold to the preceding representable value below the total
  prevents multiplication rounding from falling beyond the last branch.
  The loader's near-unit-sum contract is enforced with tolerance $10^{-5}$;
  arbitrary unnormalized weights outside that input contract are rejected.
- `sample_relaxation_cascade` tests the terminal condition before obtaining a
  draw. Its stack receives daughters in reverse order, so popping visits the
  first daughter first. A repeated daughter is appended twice, with the same
  creating-transition ordinal. Every realized daughter must have a strictly
  later subshell rank. Finite tape exhaustion raises an error.
- `_sampled_energy_budget` sums photon energies, electron kinetic energies,
  and terminal binding energies separately. Each defect is exactly parent
  binding minus the sum of daughter bindings minus emitted energy. No sign
  clamp or binding-difference substitution appears.

An independent direct numerical check constructed the three-subshell example
above without using implementation probability or budget helpers. Primary
draws 0.1 and 0.25 reproduced the two analytic energy partitions and two/three
draw counts exactly. The next untouched tape values were respectively 0.6 and
0.8. A primary cutoff of 100 eV consumed no draw and reserved 100 eV. A cutoff
of 30 eV on the electron branch retained both daughter holes, giving electron
energy 41 eV, terminal binding 60 eV, and defect $-1$ eV. This also confirms the
inclusive cutoff boundary. These are algorithm checks with independently
chosen synthetic data, not an external EADL experimental anchor.

The focused `tests/montecarlo/test_sampled_relaxation.py` run passed all 32 tests
(1.63 s). Its Si/Cu/Au ensembles provide additional implementation evidence.
The historical `vacancy_cascade` builds its daughter matrix from printed weights,
whereas this sampler divides each decaying row by its printed sum. Thus their
ensemble expectations agree exactly only for unit printed sums. Comparisons
need a rounding allowance in addition to sampling uncertainty; agreement with
the historical helper alone is not an independent probability-law derivation.
This difference does not change the per-history signed-defect identity, which
telescopes for every selected tree regardless of its branch probabilities.

Verdict: **rederived**. Units, limiting cases, and signs/conventions pass;
no divergent term was found within the stated single-vacancy model.
Human sign-off and validation of production transport integration remain
outside this claim. The task owner checks the documentation build and rendered
math before accepting this write-up.
