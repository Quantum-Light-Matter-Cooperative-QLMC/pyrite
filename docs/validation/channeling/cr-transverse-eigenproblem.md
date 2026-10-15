# `cr-transverse-eigenproblem` re-derivation

Claim: `materials/channeling/transverse.py::{transverse_states,TransverseStates.wavefunction}`.
Source: Andersen, Bonderup, Lægsgaard and Sørensen, Phys. Scr. 28, 308 (1983); Azadegan, Comput. Phys. Commun. 184, 1064 (2013), Eqs. 2–6.

## Independent derivation (written before reading the implementation body)

Dropping $O((U/E)^2)$ terms and spin from the squared Dirac equation, the longitudinal motion separates and the transverse part obeys

$$
\Bigl[-\frac{\hbar^2}{2\gamma m}\frac{d^2}{dx^2}+U(x)\Bigr]\psi=E_\perp\psi .
$$

For $U(x+d)=U(x)$ with $g=2\pi/d$, write $U(x)=\sum_j U_j e^{ijgx}$. A real $U$ gives $U_{-j}=U_j^*$. Bloch's theorem gives

$$
\psi_{k}(x)=d^{-1/2}\sum_m c_m e^{i(k+mg)x}.
$$

Projecting on $e^{i(k+mg)x}$ gives

$$
\sum_{m'}\Bigl[\frac{\hbar^2}{2\gamma m}(k+mg)^2\delta_{mm'}+U_{m-m'}\Bigr]c_{m'}=E_\perp c_m .
$$

The matrix is Hermitian, since $U_{m'-m}=U_{m-m'}^*$. Because $k$ and $k+g$ are equivalent, $k$ may be folded into $(-g/2,g/2]$. The convention $\int_0^d|\psi|^2dx=\sum_m|c_m|^2=1$ holds for unit-norm columns. Setting $U_0=0$ fixes the energy origin at the mean potential. A state is bound when $E_\perp<\max U$.

Units: $\hbar^2/2m_e=(\hbar c)^2/(2m_ec^2)=1973.2698^2/(2\times 510998.95)=3.80998\ \mathrm{eV\,\AA^2}$, and $(k+mg)^2$ in $\mathrm{\AA^{-2}}$ gives eV.

## Cheap filters (independent code, own Hermitian build and `numpy.linalg.eigh`)

| Filter | Result |
| --- | --- |
| $U=0$: eigenvalues equal $\frac{\hbar^2}{2\gamma m}(k+mg)^2$ sorted | exact; no state below $\max U=0$ |
| Cosine $U=2V_0\cos gx$, $V_0=20$ eV, $\gamma=200$: Mathieu with $q=4V_0/(a g^2)$, $a=\hbar^2/2\gamma m$ gives spacings $\hbar\omega_n=2\sqrt{aV_0g^2}-\tfrac{ag^2}{4}(n+1)+\dots$ | residual $-7\times10^{-4}$ eV on a 4.04 eV spacing for $n=0$, growing with $n$ as the next order $O(q^{-1/2})$ |
| $\gamma$ scaling at Si(110): bound count at 40/160/640 MeV | 6 / 12 / 24 |
| Parity at $k=0$ | dipole same-parity elements $<10^{-11}$ (see `cr-dipole-matrix-element`) |
| Norm per period by quadrature (4001 points) | $1-6\times10^{-15}$ |

## Numeric reference

The potential is `planar_profile('silicon',(1,1,0),293.15)` (verified separately; $d=1.92011$ Å, $\max U=6.624$ eV, well $-14.416$ eV at $x=0$). At 40 MeV, with my own solver and $M=48$:

- Bound count is 6 for every $k$ tried ($0$, $0.4$, $1.3089$ and $-1.6361$ Å$^{-1}$).
- $\varepsilon=-10.7706,\,-4.6694,\,-0.5618,\,2.3004,\,4.4302,\,5.8675$ eV at $k=1.3089$ Å$^{-1}$.
- $2\gamma^2(\varepsilon_1-\varepsilon_0)=76.692$ keV, versus Azadegan's 76.2071 keV (0.64% high). Using the exact $1/(1-\beta)$ factor gives 76.689 keV.

The 0.6% gap equals the ledger's note. It is attributable to the potential model (Doyle–Turner versus the PyRITE form factors, and the thermal amplitude), which is verified in the profile claim and not here. Azadegan lists six bound states, matching.

## Diff against the implementation

`_hamiltonian` builds $\frac{\hbar^2}{2m_e}\frac{1}{\gamma}(k+mg)^2$ on the diagonal and $U_{m-m'}$ with the conjugate for negative lag. This is identical term by term to the derivation. `transverse_states` folds $k$ with `round`, and moves the $-g/2$ edge to $+g/2$, as in the stated convention. It takes `barrier` from `profile.extrema()` and calls `eigh`. `wavefunction` returns $d^{-1/2}\sum_m c_m e^{i(k+mg)x}$.

Numeric diff over $k\in\{0,0.4,1.3089,-1.6361\}$ Å$^{-1}$ at $M=48$: maximum eigenvalue difference to my solver is $0.0$ eV, and the bound counts agree.

## Verdict

`rederived`. No divergent factor, sign or convention. Caveats, none affecting the claim: the quoted 0.6% offset from Azadegan rests on the potential, and `n_plane_waves` must exceed the highest relevant order (basis convergence $M=16$ vs 48 is covered in the ledger anchors).
