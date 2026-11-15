# `cr-submev-dirac-spin-correction` re-derivation

Claim: `checks/channeling_submev_w001_spin.py::dirac_terms`, with the
Klein–Gordon separation of `checks/channeling_submev_w001_kinematics.py`.
Recorded in the {ref}`finite-β audit <sec-submev-w001-finite-beta>`.
Research record for #585; the production transverse Hamiltonian is spin-free
and drops $-V^2/2E$.

Fresh context, 2026-11-15, commit `180691b42`.

## Klein–Gordon separation

$$
(E-V)^2-c^2p_\perp^2-(E^2-m^2c^4-2E\varepsilon)-m^2c^4
=2E\Bigl(\varepsilon-\frac{c^2p_\perp^2}{2E}-V+\frac{V^2}{2E}\Bigr).
$$

The separation is exact for a potential independent of $z$. The transverse
mass $E/c^2$ is not a paraxial approximation.

## Spinor product

Take
$u=\sqrt{(E+mc^2)/2E}\,\bigl(\chi,\ c\boldsymbol\sigma\cdot\mathbf p\,\chi/(E+mc^2)\bigr)$.
Then $u^\dagger u=1$ on shell. With
$(\boldsymbol\sigma\cdot\mathbf a)(\boldsymbol\sigma\cdot\mathbf b)=\mathbf a\cdot\mathbf b+i\boldsymbol\sigma\cdot(\mathbf a\times\mathbf b)$:

$$
u^\dagger(\mathbf p')u(\mathbf p)=\frac{E+mc^2}{2E}\Bigl[1+\frac{c^2(\mathbf p'\cdot\mathbf p+i\boldsymbol\sigma\cdot(\mathbf p'\times\mathbf p))}{(E+mc^2)^2}\Bigr].
$$

**Darwin term.** With $|\mathbf p'|=|\mathbf p|$, the spin-free part minus 1
is $-c^2|\mathbf p'-\mathbf p|^2/4E(E+mc^2)$. In real space that is
$+\hbar^2c^2\nabla^2V/4E(E+mc^2)$, the Darwin term. Its non-relativistic limit
is $\hbar^2\nabla^2V/8m^2c^2$. It is positive at the string core and raises
1s.

**Spin–orbit term.** With $\mathbf p'-\mathbf p=\hbar\mathbf q$ and
$\nabla V\leftrightarrow i\mathbf qV_q$, the longitudinal part is
$\hbar c\beta/2(E+mc^2)\,\boldsymbol\sigma\cdot(\nabla V\times\hat z)$. That
equals the $g=2$ BMT energy $(\hbar/2mc)\,\beta/(\gamma+1)\,|\nabla V|$. The
transverse part gives the $\sigma_zL_z$ splitting of 2p.

## Consistency of the on-shell form

Eliminating the lower component at fixed $p_z$ gives exactly

$$
[(E-V)^2-m^2c^4-c^2p^2]\phi=-i\hbar c^2(\boldsymbol\sigma\cdot\nabla V)(E+mc^2-V)^{-1}(\boldsymbol\sigma\cdot\mathbf p)\phi .
$$

Divided by $2E$, this adds
$c^2[-i\hbar\nabla V\cdot\mathbf p+\hbar\boldsymbol\sigma\cdot(\nabla V\times\mathbf p)]/2E(E+mc^2)$
to $H_\text{KG}$, up to a relative $O(V/(E+mc^2))$.

- The Hermitian part of $-i\hbar\nabla V\cdot\mathbf p$ is the on-shell
  Darwin form.
- The anti-Hermitian part is proportional to $[p_\perp^2,V]$, a commutator
  with $H_0$, so it gives no first-order shift.

The on-shell construction is therefore the consistent first-order correction,
and its sign is definite. A common $p_z$ adds a spurious
$\{c^2p_\perp^2/2E,V\}/2(E+mc^2)$ term, of order $V^2/E$. That variant is
inconsistent.

## Code and hand check

These match in `dirac_terms`:

- the row/column convention (rows are $\mathbf p'$);
- `scale` $=1/2E(E+mc^2)$;
- the cross products and eV·Å units;
- the Hermiticity check;
- `plain` reproduces `axial_states`.

Hand check at 30 keV: $226.5\ \text{eV}\times(-3.0\ \text{to}\ -1.9)\times10^{-4}/(1-\beta)$
gives −0.10 to −0.064 eV in the line, matching the record.

## Findings and adjudication

- The docstring gave the non-relativistic limit as $\hbar^2/4m^2c^2$; it is
  $\hbar/4m^2c^2$. *Addressed.*
- The page and docstring called the result a sign-ambiguous bound on
  magnitude. They also said the omitted second-order Born terms were of the
  order of $-V^2/2E$. Both are wrong: Klein–Gordon is exact for the spin-free
  part, and what remains is a relative $O(V/(E+mc^2))\lesssim10^{-3}$
  correction to the Darwin and spin terms. *Addressed*: the page states the
  signed result and this argument.

Verdict: `rederived` for the matrix element, the coefficient identification,
the Klein–Gordon separation, the consistency argument and the code. This is a
static check at $k=0$, with no longitudinal harmonics. Human sign-off is
tracked in #277.
