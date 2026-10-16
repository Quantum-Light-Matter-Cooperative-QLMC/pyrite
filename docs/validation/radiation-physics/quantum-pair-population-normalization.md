# `quantum-pair-population-normalization`

## Claim and source

Research claim (#501): for $N$ electrons and the first-order one-photon
channel operator $\hat M$ acting on one electron, the emitted photon number in
one initially empty mode is a one-body plus a two-body reduced-density-matrix
term; the separable i.i.d. limit, the classical-mixture reduction to
`spxr-physical-population`, a Cauchy–Schwarz ceiling $\langle n\rangle\le
N^2\operatorname{Tr}(\rho^{(1)}\hat M^\dagger\hat M)$, and a fermion-exchange
term dropped at low beam degeneracy.

Source cited by the ledger: expansion of $\hat A^\dagger\hat A$ with
$\hat A=\sum_a\hat M^{(a)}$; Karnieli et al., Phys. Rev. Lett. 127, 060403
(2021) for entangled pairs; Zhang et al., arXiv:2401.05978 for the
modulated-wavepacket $N^2$ term. There is no production implementation; the
"implementation" compared in the last section is
`docs/research/physics/quantum-coherent-emission.md`.

This is a fresh-context derivation written before that research note was read.

## Setup

Let $\mathcal H_e$ be the one-electron space and $\rho_N$ the $N$-electron
state on $\mathcal H_e^{\otimes N}$, written in first quantization with labels
$a=1,\dots,N$ (antisymmetric for identical fermions, arbitrary for
distinguishable electrons). The photon mode starts in $|0\rangle$. To first
order in the coupling, the interaction-picture evolution maps

$$
|\Psi_e\rangle|0\rangle\;\mapsto\;|\Psi_e\rangle|0\rangle
+\hat A|\Psi_e\rangle|1\rangle+O(g^2),\qquad
\hat A=\sum_{a=1}^N\hat M^{(a)},
$$

where $\hat M=-\tfrac{i}{\hbar}\int dt\,\langle1|\hat H_{\rm int}(t)|0\rangle$
acts on $\mathcal H_e$ only and $\hat M^{(a)}$ is $\hat M$ on factor $a$. The
lowest-order photon number is then

$$
\langle n\rangle=\operatorname{Tr}\!\left(\rho_N\hat A^\dagger\hat A\right)+O(g^4).
$$

$\hat M$ is dimensionless (a probability amplitude), so $\langle n\rangle$ is
dimensionless per mode; per unit $\omega$ and solid angle it carries the mode
density, the same units as one channel's spectral yield.

## Reduced-density-matrix form

Expanding the product of one-body sums,

$$
\hat A^\dagger\hat A=\sum_a(\hat M^\dagger\hat M)^{(a)}
+\sum_{a\ne b}\hat M^{(a)\dagger}\hat M^{(b)} .
$$

Define unit-trace labelled marginals
$\rho^{(1)}_a=\operatorname{Tr}_{\neg a}\rho_N$ and
$\rho^{(2)}_{ab}=\operatorname{Tr}_{\neg a,b}\rho_N$ (first factor $a$, second
$b$). Then

$$
\boxed{\langle n\rangle=\sum_a\operatorname{Tr}\!\left[\rho^{(1)}_a\hat M^\dagger\hat M\right]
+\sum_{a\ne b}\operatorname{Tr}\!\left[\rho^{(2)}_{ab}\left(\hat M^\dagger\otimes\hat M\right)\right]}
$$

with the sum over **ordered** pairs ($N(N-1)$ terms). The $(a,b)$ and $(b,a)$
terms are complex conjugates, so the pair sum is real and equals
$2\operatorname{Re}\sum_{a<b}$. With the second-quantized normalization
$\gamma=\sum_a\rho^{(1)}_a$ ($\operatorname{Tr}\gamma=N$) and
$\Gamma=\sum_{a\ne b}\rho^{(2)}_{ab}$ ($\operatorname{Tr}\Gamma=N(N-1)$) the
same result reads
$\langle n\rangle=\operatorname{Tr}(\gamma\hat M^\dagger\hat M)
+\operatorname{Tr}[\Gamma(\hat M^\dagger\otimes\hat M)]$ with no explicit sums.
The ledger's form is consistent provided its $\rho^{(1)}_a$, $\rho^{(2)}_{ab}$
are unit-trace and its pair sum is ordered; mixing the two conventions would
introduce a spurious factor $N(N-1)$ or $2$.

This expression is exact at this order for any $\rho_N$: distinguishable,
symmetric, antisymmetric, mixed or entangled. For identical particles the
exchange physics is already inside $\rho^{(2)}$; nothing is "dropped" at this
stage.

## Separable i.i.d. limit

For $\rho_N=\rho^{\otimes N}$, $\rho^{(1)}_a=\rho$ and
$\rho^{(2)}_{ab}=\rho\otimes\rho$, so

$$
\operatorname{Tr}[(\rho\otimes\rho)(\hat M^\dagger\otimes\hat M)]
=\operatorname{Tr}(\rho\hat M^\dagger)\operatorname{Tr}(\rho\hat M)
=\left|\operatorname{Tr}(\rho\hat M)\right|^2,
$$

using $\operatorname{Tr}(\rho\hat M^\dagger)=\overline{\operatorname{Tr}(\rho\hat M)}$
for Hermitian $\rho$. Hence

$$
\langle n\rangle=N\operatorname{Tr}(\rho\hat M^\dagger\hat M)
+N(N-1)\left|\operatorname{Tr}(\rho\hat M)\right|^2 .
$$

A product state is not antisymmetric; this limit treats electrons as
distinguishable, which is the low-degeneracy limit below.

## Fermion exchange term

For a Slater determinant, or more generally a quasi-free (Gaussian) fermion
state, Wick's theorem gives $\Gamma=(\gamma\otimes\gamma)(1-\hat P)$ with
$\hat P$ the swap. Then

$$
\operatorname{Tr}[\Gamma(\hat M^\dagger\otimes\hat M)]
=\left|\operatorname{Tr}(\gamma\hat M)\right|^2
-\operatorname{Tr}\!\left(\gamma\hat M^\dagger\gamma\hat M\right).
$$

The exchange term is
$-\operatorname{Tr}(\gamma\hat M^\dagger\gamma\hat M)
=-\|\gamma^{1/2}\hat M\gamma^{1/2}\|_{\rm HS}^2\le0$: it always **reduces** the
photon number (Pauli antibunching). It is bounded by

$$
0\le\operatorname{Tr}(\gamma\hat M^\dagger\gamma\hat M)
\le\|\gamma\|_{\rm op}\operatorname{Tr}(\gamma\hat M^\dagger\hat M)
=\delta\,\operatorname{Tr}(\gamma\hat M^\dagger\hat M),
$$

where $\delta=\|\gamma\|_{\rm op}\le1$ is the maximum natural-orbital
occupation, i.e. the beam degeneracy. So the exchange term is at most $\delta$
times the incoherent (linear) term, and negligible against it when
$\delta\ll1$; for a pure Slater determinant $\delta=1$ and it is not negligible.
Writing $\gamma=N\rho$, the direct term is $N^2|\operatorname{Tr}(\rho\hat M)|^2$
rather than $N(N-1)|\cdot|^2$; the difference is the quasi-free state's
sub-Poissonian number statistics, $\langle N(N-1)\rangle=N^2-\operatorname{Tr}\gamma^2$,
with $\operatorname{Tr}\gamma^2\le\delta N$. Both corrections are $O(\delta)$ of the
linear term, so the i.i.d. form is the $\delta\to0$ limit. The scale "degeneracy
$\ll1$" is correct; "dropped" should be read as "bounded by $\delta$ times the
linear term", not as small compared to the coherent term.

## Cauchy–Schwarz ceiling, any state

On operators on $\mathcal H_e^{\otimes2}$ define the positive semidefinite form
$\langle X,Y\rangle_{\rho^{(2)}}=\operatorname{Tr}(\rho^{(2)}X^\dagger Y)$. With
$X=\hat M\otimes1$ and $Y=1\otimes\hat M$, $X^\dagger Y=\hat M^\dagger\otimes\hat M$, so

$$
\left|\operatorname{Tr}[\rho^{(2)}_{ab}(\hat M^\dagger\otimes\hat M)]\right|^2
\le\operatorname{Tr}(\rho^{(1)}_a\hat M^\dagger\hat M)\,
\operatorname{Tr}(\rho^{(1)}_b\hat M^\dagger\hat M).
$$

No separability is used; the bound holds for entangled and antisymmetrized
$\rho^{(2)}$. With equal marginals (automatic for identical particles, or any
label-symmetric state) $|\cdot|\le\operatorname{Tr}(\rho^{(1)}\hat M^\dagger\hat M)\equiv T$
and

$$
\langle n\rangle\le NT+N(N-1)T=N^2T .
$$

For unequal marginals the general result is
$\langle n\rangle\le\left(\sum_a\sqrt{T_a}\right)^2\le N\sum_aT_a$. The equality
$X|\cdot\rangle\propto Y|\cdot\rangle$ on the support of $\rho^{(2)}$ saturates
it, e.g. electrons that are common eigenstates of $\hat M$ with equal
eigenvalues. Entanglement can therefore redistribute but never exceed the
$N^2$ ceiling set by the single-electron $\operatorname{Tr}(\rho^{(1)}\hat M^\dagger\hat M)$.
Positivity of $\langle n\rangle$ gives the matching lower bound on the pair sum,
$\sum_{a\ne b}\ge-NT$.

## Classical-mixture reduction and what "c-number" must mean

Let $\rho=\sum_jp_j|j\rangle\langle j|$ with $|j\rangle$ normalized trajectory
wavepackets (not necessarily orthogonal; the trace formula is linear in
$\rho$, so any decomposition works). Suppose the emission acts as

$$
\hat M|j\rangle=E_j|j'\rangle,\qquad\langle j'|j'\rangle=1,
$$

where $E_j$ is the classical point-charge field amplitude of trajectory $j$
(including its arrival phase) and $|j'\rangle$ is the recoiled state
(momentum lowered by $\hbar\mathbf k$). Then:

- self term, exact: $\operatorname{Tr}(\rho\hat M^\dagger\hat M)=\sum_jp_j|E_j|^2=\langle|E|^2\rangle$;
- pair term: $\operatorname{Tr}(\rho\hat M)=\sum_jp_jE_j\langle j|j'\rangle$.

The pair term equals $\langle E\rangle$ only if $\langle j|j'\rangle=1$ for
every $j$, i.e. the recoil is negligible and $|j\rangle$ is (to this accuracy)
an eigenvector of $\hat M$. Since $|\langle j|j'\rangle|\le1$, the general
quantum pair term is never larger than the classical one, and a $j$-dependent
phase of $\langle j|j'\rangle$ suppresses it further.

For the usual form $\hat M\simeq E\,e^{-i\mathbf k\cdot\hat{\mathbf r}}$ (dipole
current in a fixed trajectory field) and a wavepacket of density
$|\psi_j(\mathbf r)|^2$ centred at $\mathbf r_j$,
$\langle j|j'\rangle e^{i\mathbf k\cdot\mathbf r_j}$ is the single-electron
form factor $\int|\psi_j|^2e^{-i\mathbf k\cdot(\mathbf r-\mathbf r_j)}d^3r$,
e.g. $e^{-k_\parallel^2\sigma^2/2}$ for a Gaussian of width $\sigma$ along
$\mathbf k$. The precise c-number conditions are therefore: (i) each component
is a pure wavepacket localized below the reduced wavelength along $\mathbf k$,
$k\sigma\ll1$ (equivalently momentum spread $\gg\hbar k$, so the recoiled state
overlaps the original); (ii) the mixture over $j$ is classical (independent
trajectory sampling); (iii) electrons are i.i.d. Then

$$
\langle n\rangle=N\langle|E|^2\rangle+N(N-1)|\langle E\rangle|^2,
$$

which is `spxr-physical-population`. A momentum-diagonal (energy-eigenstate)
mixture has $\operatorname{Tr}(\rho\hat M)=0$ under energy conservation and
gives only the linear term; the $N^2$ term requires coherence between momenta
separated by the recoil, which is the modulated-wavepacket mechanism. The
ledger phrase "classical mixture of c-number-field (trajectory) states" is
correct only with conditions (i)–(ii) made explicit; without them it overclaims
(the self term matches unconditionally, the pair term does not).

## Limits

| check | result |
| --- | --- |
| $N=1$ | no ordered pairs; $\langle n\rangle=\operatorname{Tr}(\rho\hat M^\dagger\hat M)$ |
| $\operatorname{Tr}(\rho\hat M)=0$, i.i.d. | linear term $N\operatorname{Tr}(\rho\hat M^\dagger\hat M)$ only |
| identical pure eigenstates, $\hat M\lvert\psi\rangle=E\lvert\psi\rangle$ | $N\lvert E\rvert^2+N(N-1)\lvert E\rvert^2=N^2\lvert E\rvert^2$, saturating the ceiling |
| identical pure non-eigenstates | $N\lVert\hat M\psi\rVert^2+N(N-1)\lvert\langle\psi\vert\hat M\vert\psi\rangle\rvert^2<N^2\lVert\hat M\psi\rVert^2$ |
| classical trajectory mixture, $k\sigma\ll1$ | `spxr-physical-population` |
| Slater determinant, $N=1$ | direct minus exchange $=0$, no pair term |

The ledger's "identical c-number-like pure states give $N^2$" is correct only
for eigenstates (c-number in the sense above); a generic identical pure state
gives the strictly smaller middle row.

## Numerical spot check

A scratch script (not committed) built $\hat A=\sum_a\hat M^{(a)}$ explicitly
on $(\mathbb C^4)^{\otimes3}$ with a random non-Hermitian $\hat M$ and compared
$\operatorname{Tr}(\rho_N\hat A^\dagger\hat A)$ with the reduced-density-matrix
form:

| state | $\operatorname{Tr}(\rho_N\hat A^\dagger\hat A)$ | prediction |
| --- | --- | --- |
| random mixed antisymmetric | 9.09333326 | RDM form 9.09333326; ceiling holds |
| random entangled, label-symmetric | 21.4366937 | RDM form 21.4366937; $\lvert{\rm pair}\rvert\le T$ and $\le N^2T$ hold |
| Slater determinant | 9.18813921 | direct − exchange 9.18813921 |
| i.i.d. product | 18.0649332 | i.i.d. formula 18.0649332 |

All agree to $10^{-14}$ relative.

## Comparison with the research note

Read after the derivation above was fixed:
`docs/research/physics/quantum-coherent-emission.md` §1, §4, §7, §8 and
`docs/research/physics/superradiant-pxr.md`.

| item | note | this derivation | agreement |
| --- | --- | --- | --- |
| RDM form, `eq-qce-pair` | ordered $a\ne b$, $\rho^{(2)}_{ab}$ per ordered pair | same, unit-trace marginals | matches; the note does not state unit-trace normalization explicitly |
| i.i.d. limit, `eq-qce-iid` | $N\operatorname{Tr}(\rho\hat M^\dagger\hat M)+N(N-1)\lvert\operatorname{Tr}(\rho\hat M)\rvert^2$ | same | matches |
| classical reduction | c-number field $\mathcal E_j$, "point charge, recoil neglected", "exactly" | needs $\langle j\vert j'\rangle=1$: $k\sigma\ll1$ and negligible recoil | matches in substance; "exactly" holds only inside that idealization |
| quantum content | $\operatorname{Tr}(\rho\hat M)$ needs coherence between a state and its recoiled state | same (form factor; momentum-diagonal mixture gives zero) | matches |
| Cauchy–Schwarz ceiling | $\langle n\rangle\le N^2\operatorname{Tr}(\rho^{(1)}\hat M^\dagger\hat M)$ for any state, equal marginals | same; general $(\sum_a\sqrt{T_a})^2$ | matches |
| exchange sign/size | subtracts $\operatorname{Tr}(\rho\hat M^\dagger\rho\hat M)$ per pair, bounded by degeneracy | $-\operatorname{Tr}(\gamma\hat M^\dagger\gamma\hat M)\le\delta\operatorname{Tr}(\gamma\hat M^\dagger\hat M)$ | matches with $\gamma=N\rho$ to leading order in $1/N$ |
| exchange "vanishes for orthogonal orbitals" | stated | false as written: for orthogonal occupied orbitals $\phi,\chi$ the exchange is $\lvert\langle\chi\vert\hat M\vert\phi\rangle\rvert^2+\lvert\langle\phi\vert\hat M\vert\chi\rangle\rvert^2$ | **wording error**: it vanishes when $\hat M$ has no matrix elements between distinct occupied orbitals (phase-space separation larger than the recoil/emission bandwidth), not for orthogonality |
| limit "identical pure states with $\hat M\lvert\psi\rangle\propto\lvert\psi'\rangle$ give $N^2$" | stated (also in the ledger Checks) | $N^2$ needs $\lvert\psi'\rangle=\lvert\psi\rangle$ up to phase (eigenstate); otherwise $N\lVert\hat M\psi\rVert^2+N(N-1)\lvert\langle\psi\vert\hat M\psi\rangle\rvert^2$ | **wording too loose**: $\hat M\lvert\psi\rangle\propto\lvert\psi'\rangle$ holds for every state |
| "redistribute emission between channels up to $N^2$" | stated | ceiling is per channel; no inter-channel sum rule is derived | loose wording, not a factor error |
| §8 prebunched pair term | $N(N-1)\lvert\operatorname{Tr}(\rho\hat M)\rvert^2\lvert b(\omega)\rvert^2$ | follows from `eq-qce-iid` if arrival time is independent of the transverse state | matches under that stated independence |
| `spxr-physical-population` | $N\langle\lvert\mathcal E\rvert^2\rangle+N(N-1)\lvert\langle\mathcal E\rangle\rvert^2$, ordered pairs | same | matches |
| §4 status label | "research claim, `rederived`" | ledger row is `unverified` at time of review | the note's label was ahead of the ledger |

## Adjudication

The load-bearing identities match: the reduced-density-matrix expansion, the
i.i.d. limit, the classical reduction under explicit no-recoil and point-charge
conditions, the Cauchy–Schwarz $N^2$ ceiling for arbitrary (including
entangled and antisymmetrized) states, and the sign and degeneracy scale of the
exchange term. No factor, sign or exponent differs. Three wording fixes are
needed, listed in the table above: the "vanishes for orthogonal orbitals"
exchange statement, the "$\hat M\lvert\psi\rangle\propto\lvert\psi'\rangle$
gives $N^2$" limit, and the explicit no-recoil/localization condition for
"c-number". Verifier verdict: `rederived`, provided those wording fixes are
applied. Human sign-off is still required.
