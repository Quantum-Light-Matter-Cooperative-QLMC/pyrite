# `cr-dipole-matrix-element` re-derivation

Claim: `materials/channeling/transverse.py::TransverseStates.dipole_ang`. Source: the commutator identity, with Andersen et al. (1983).

## Independent derivation (before reading the body)

Take $H=\frac{p^2}{2\gamma m}+U(x)$ with $p=-i\hbar\,d/dx$ and $[x,p]=i\hbar$. Then $[U,x]=0$ and

$$
[H,x]=\frac{1}{2\gamma m}[p^2,x]=\frac{1}{2\gamma m}\bigl(p[p,x]+[p,x]p\bigr)=-\frac{i\hbar p}{\gamma m}.
$$

For eigenstates $H\lvert n\rangle=E_n\lvert n\rangle$,

$$
\langle f\vert[H,x]\vert i\rangle=(E_f-E_i)\,x_{fi}=-\frac{i\hbar}{\gamma m}\langle f\vert p\vert i\rangle .
$$

For Bloch states at the same $k$, with $p\to\hbar(k+mg)$ on each plane wave,

$$
\boxed{x_{fi}=-i\frac{\hbar^2}{\gamma m}\sum_m c_{f,m}^*\,(k+mg)\,c_{i,m}\;\frac{1}{E_f-E_i}}
$$

Here $\hbar^2/(\gamma m)=2\,(\hbar^2/2m_e)/\gamma$. Units: $\mathrm{eV\,\AA^2}\cdot\mathrm{\AA^{-1}}/\mathrm{eV}=$ Å. The state labels need $E_f\ne E_i$.

Caveats: the identity holds for the same $k$ because $p$ is diagonal in $k$. The off-$k$ interband position is ill-defined on periodic functions; this is why the commutator form is used. For a real, even $U$ at $k=0$, eigenvectors have $c_{-m}=\pm c_m$. The sum $\sum_m (mg)\,c_{f,m}^*c_{i,m}$ is then odd in $m$ and vanishes for equal parities, so same-parity elements vanish without being imposed. With real $c$ (the real gauge) $x_{fi}$ is purely imaginary, because the plane-wave basis is complex. Only $\lvert x_{fi}\rvert$ is gauge independent.

## Independent numerical check (Si(110), 40 MeV, $k=0$, $M=48$)

Direct quadrature of $\langle f\vert x\vert i\rangle$ over one centred period $[-d/2,d/2)$ with 4000 points versus the commutator formula:

| $(f,i)$ | commutator form | quadrature |
| --- | --- | --- |
| (1,0) | $-0.087991\,i$ Å | $-0.087991\,i$ Å |
| (2,1) | $-0.150169\,i$ Å | $-0.150169\,i$ Å |
| (3,0) | $-0.007027\,i$ Å | $-0.007027\,i$ Å |
| (3,2) | $-0.217952\,i$ Å | $-0.217989\,i$ Å (quadrature differs by $2\times10^{-4}$ relative) |
| (2,0), same parity | $<10^{-16}$ | $<5\times10^{-12}$ |

Phase factors differ by the eigenvector gauge, but the magnitudes are independent of it, and in this gauge the quadrature agrees too. The (3,2) difference of $2\times10^{-4}$ relative is plausibly finite-cell truncation and quadrature error of the weakly confined upper states; I did not isolate it, and the ledger's 2e-3 deep-state tolerance covers it.

## Diff against the implementation

`dipole_ang` raises when $\lvert\Delta E\rvert<10^{-12}$ eV or $f=i$. Otherwise it returns $-i\cdot 2(\hbar^2/2m_e)/\gamma\cdot\langle c_f\vert(k+mg)\vert c_i\rangle/\Delta E$, where `np.vdot` conjugates the first argument. This is the derived expression with the same signs and factor of two.

Numeric diff for pairs (1,0), (2,1), (3,2), (5,4): $\lvert x_{fi}\rvert$ differs from my own code by $\le 6\times10^{-17}$ Å.

## Verdict

`rederived`. Gauge note: the phase of $x_{fi}$ depends on the eigenvector sign convention of `eigh`, so consumers should use $\lvert x_{fi}\rvert^2$ (as in the emission-rate claim, #505).
