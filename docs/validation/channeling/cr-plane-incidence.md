# `cr-plane-incidence` re-derivation

Claim: `materials/channeling/entry.py::plane_incidence`. Source: reciprocal-lattice geometry; electron refraction by the mean inner potential (Spence & Zuo 1992).

## Independent derivation (before reading the body)

**Plane normal.** Planes $(hkl)$ have normal along $\mathbf g_{hkl}$, so in the sample frame $\mathbf n=R\mathbf g/|\mathbf g|$, with $R$ the lattice-to-sample rotation. The sign of $\mathbf n$ fixes the sign of the plane-local angle only.

**Beam direction.** Let the face normal in the lab be $\mathbf N=(\sin t\cos a,\sin t\sin a,\cos t)$ and let the sample frame be reached by the minimal rotation $R_t=R_z(a)R_y(t)R_z(-a)$ with $R_t\mathbf e_z=\mathbf N$. The beam is $\mathbf e_z$ in the lab, so in the sample frame

$$
\mathbf b=R_t^{T}\mathbf e_z=(-\sin t\cos a,\,-\sin t\sin a,\,\cos t).
$$

(The tangential sign is tied to the minimal-rotation convention; I compare against `tilted_geometry` numerically.)

**Refraction.** Let $p(T)=\sqrt{T(T+2mc^2)}/\hbar c$ [$\mathrm{\AA^{-1}}$]. Vacuum: $\mathbf p=p(T)\,\mathbf b$. At the face (normal $\mathbf e_z$) the tangential component is conserved, $\mathbf p_\parallel=p(T)(b_x,b_y)$. Inside, the energy is $T-U_0$, so

$$
p_{\rm in}=p(T-U_0),\qquad p_{{\rm in},z}=\sqrt{p_{\rm in}^2-p_\parallel^2},\qquad
\mathbf p_{\rm in}=(\mathbf p_\parallel,\,p_{{\rm in},z}).
$$

Since $U_0<0$, $p_{\rm in}>p$ and $p_{{\rm in},z}>p_z$. The transverse wavevector is

$$
k_x=\mathbf p_{\rm in}\cdot\mathbf n=\mathbf p_\parallel\cdot\mathbf n_\parallel+p_{{\rm in},z}\,n_z .
$$

Reported angles: $\psi_{\rm geo}=\arcsin(\mathbf b\cdot\mathbf n)$; $\psi_{\rm int}=\arcsin(k_x/p_{\rm in})$; $\psi_{\rm eff}=k_x/p(T)$, so $\psi_{\rm eff}\,p(T)=k_x$ and `entry_populations` (with $k_x=p\psi$) reproduces $k_x$ exactly.

**Limits.** (i) $U_0\to0$: $p_{\rm in}=p$, $\mathbf p_{\rm in}=\mathbf p$, $k_x=p\,\mathbf b\cdot\mathbf n$, and $\psi_{\rm int}=\psi_{\rm geo}$ with $\psi_{\rm eff}=\sin\psi_{\rm geo}$. (ii) $n_z=0$: $k_x=\mathbf p_\parallel\cdot\mathbf n_\parallel$ is unchanged by refraction. (iii) Normal incidence on planes inclined by $\vartheta$ to the face ($n_z=\cos\vartheta$): $\delta k_x\approx n_z\,(p_{\rm in}-p)=n_z(-U_0)/(\hbar c\,\beta)$, since $dp/dT=1/(\hbar c\beta)$. (iv) Tilt within a plane family leaves $\psi=0$ (tilt axis lies in the planes); tilt across gives $\psi=t$.

## Independent numeric check

Own construction: $\mathbf b=R_z(a)R_y(-t)R_z(-a)\mathbf e_z$, $\mathbf n=R\,\mathbf{hkl}/|\cdot|$ (cubic, $\mathbf g\propto(h,k,l)$), $R$ from `resolve_crystal_cut(CrystalCut(surface_hkl=(0,0,1), miscut_deg=m), None, (4,4,0)).rotation(lattice)` (the orientation itself is the separate #25 claim), and $U_0$ from the code (the $U_0$ claim is checked in [its own write-up](cr-mean-inner-potential.md)).

Scan: Si, $m\in\{0,1^\circ\}$; planes $(110),(004),(220),(111),(1\bar10)$; $T=40$ MeV and $30$ keV; $(t,a)\in\{(0,0),(0.3^\circ,0),(0.3^\circ,90^\circ),(0.5^\circ,37^\circ),(-0.4^\circ,0),(2^\circ,60^\circ)\}$. Compared: beam direction, plane normal, $\psi_{\rm geo}$, $\psi_{\rm int}$, $\psi_{\rm eff}$. Maximum absolute difference $4\times10^{-16}$.

Further checks:

- `tilted_geometry(0.01, 0.3°, 37°)` returns beam $(-0.004182,-0.003151,0.999986)$, equal to my $\mathbf b$, so the tilt-sign convention matches.
- $(110)$ planes on the $(001)$ face ($n_z=0$): $\psi_{\rm eff}$ with and without refraction is identical (difference exactly 0) at 30 keV and 40 MeV, with tilt $0.5^\circ$, azimuth $37^\circ$.
- $U_0\to0$ (`refraction=False`), $1^\circ$ miscut $(004)$: $\psi_{\rm eff}=\sin\psi_{\rm geo}$ and $\psi_{\rm int}=\psi_{\rm geo}$ to machine precision.
- Magnitude of refraction for the $1^\circ$ miscut, $(004)$ planes, normal tilt: $\psi_{\rm eff}-\sin\psi_{\rm geo}=2.3649\times10^{-4}$ at 30 keV against estimate (iii) $2.3703\times10^{-4}$ (0.2% from the next-order term), and $3.41\times10^{-7}$ at 40 MeV against $3.414\times10^{-7}$. Scaling by $n_z=\sin1^\circ=0.01745$ for planes standing nearly perpendicular to the face (the $(100)$-type case in the ledger note) gives $5.96\times10^{-9}$ rad at 40 MeV, consistent with the ledger's "about $6\times10^{-9}$ rad".
- Convention note: $\psi_{\rm eff}=k_x/p_{\rm vac}$ differs from $\sin\psi_{\rm int}$ by the factor $p_{\rm in}/p$ ($\approx1+10^{-5}$ at 40 MeV and $\approx10^{-3}$ at 30 keV), as intended by the docstring.

## Diff against the implementation

Code: `normal = R g/|g|`; `face_lab = (sin t cos a, sin t sin a, cos t)`, `beam = _rotation_between(e_z, face_lab).T @ e_z`; `p_par = p_out*beam[:2]`; `p_in = p(T-U_0)`; `p_z = sqrt(p_in^2 - p_par^2)`; `k_x = p_vec . n`; `geometric = asin(b.n)`, `internal = asin(k_x/p_in)`, `effective = k_x/p_out`. This is term-for-term my derivation. It also raises when `beam_z <= 0`. The only difference is the finite-difference $U_0$ of the other claim (-0.2%).

**Verdict: `rederived`.**
