# surface-hkl-orientation

> Ownership refresh (2026-10-20, #599): current entry points replace retired
> monolithic module paths below. Dated source excerpts, line numbers, numerical
> evidence and verdicts describe their original revision; this reference repair
> does not independently revalidate those equations or extend their evidence.

- **Claim id**: `surface-hkl-orientation`
- **Anchor**: `src/pyrite/montecarlo/geometry.py::_orientation_R`; plumbing through `src/pyrite/campaign/sweep.py`, `src/pyrite/montecarlo/spectrum/cxr/_spectrum.py`, `src/pyrite/montecarlo/detector.py`, `src/pyrite/montecarlo/runner/case_driver.py`
- **Source**: standard reciprocal-lattice geometry (Ashcroft & Mermin ch. 5; International Tables A, §1.1) and the Rodrigues rotation formula
- **Verifier**: independent fresh context (did not write the implementation), 2026-09-13. Sections 1--3 were derived **before** reading the body of `_orientation_R`; only the ledger row, the module docstring of `montecarlo/geometry.py`, and the signature + derivation docstring of `_orientation_R` were read first, per the [validation methodology](../methodology.md).

## 1. What is claimed

`_orientation_R(lattice, beam_uvw, azimuth_rad, recip_miscut_rad=None, surface_hkl=None)` returns the single rotation matrix $R$ applied to *every* reciprocal vector before transport, such that

1. the cleavage-plane normal selected by `surface_hkl` $=(h,k,l)$, namely the reciprocal-lattice vector

   $$
   \mathbf{g}_{hkl}=h\,\mathbf{b}_1+k\,\mathbf{b}_2+l\,\mathbf{b}_3,
   $$

is carried onto the sample $+z$ axis (the outward slab normal) by the **minimal** (smallest-angle) rotation;
2. that minimal rotation is **proper**, $R R^{\mathsf T}=I$ and $\det R=+1$, so crystal handedness is preserved;
3. a right-handed roll by `azimuth_rad` $=\psi$ about sample $+z$ is applied **after** the minimal rotation;
4. the alternative selector `beam_uvw` $=(u,v,w)$ does the same for the *direct*-lattice direction $u\,\mathbf{a}_1+v\,\mathbf{a}_2+w\,\mathbf{a}_3$, and the two selectors are mutually exclusive;
5. for an orthogonal cell a one-axis `surface_hkl` reduces to the parallel direct axis (so the reciprocal path is a strict generalization of the legacy direct-axis path);
6. `recip_miscut_rad=None` is a bit-for-bit no-op.

Intended quantity: a dimensionless element of $SO(3)$. Inputs: integer Miller indices (dimensionless), lattice lengths in Å and angles in degrees, azimuth in radians. The claim under test is purely geometric, so the cheap filters are dimensional triviality plus sign/handedness/limiting-orientation checks.

## 2. Cheap filters

**Dimensions.** $\mathbf{b}_i$ carry Å$^{-1}$ (or are dimensionless per Å if the crystallographic normalization is used); $h,k,l$ are pure integers, so $\mathbf{g}_{hkl}$ carries Å$^{-1}$. $R$ acts on $\hat{\mathbf{g}} =\mathbf{g}/\lVert\mathbf{g}\rVert$, which is dimensionless, and $R$ itself is dimensionless. $\psi$ enters only through $\cos\psi,\sin\psi$. Pass, and note that the overall scale of $\mathbf{g}$ — hence the entire $2\pi$-vs-no-$2\pi$ question — cannot affect $R$, because only $\hat{\mathbf{g}}$ enters. Pass.

**Limiting orientations.**

- $\mathbf{g}\parallel+\hat{\mathbf{z}}$ already: the minimal rotation must be $R_{\min}=I$ (rotation angle $0$).
- $\mathbf{g}\parallel-\hat{\mathbf{z}}$: the minimal rotation angle is $\pi$ and the axis is *any* unit vector in the $xy$ plane — the construction is genuinely degenerate and a naive Rodrigues expression divides by zero. A correct implementation must special-case it, and must not fall back on $-I$, which has $\det=-1$ and is not a rotation.
- Cubic (or any orthogonal) cell with $(h,k,l)=(1,0,0)$: $\mathbf{b}_1 \parallel\mathbf{a}_1$, so the reciprocal and direct paths must return the same $R$.
- $\psi=0$: $R=R_{\min}$, the minimal-rotation in-plane convention.

**Signs / handedness.** A right-handed roll about $+z$ by $\psi$ sends $+\hat{\mathbf{x}}\to\cos\psi\,\hat{\mathbf{x}}+\sin\psi\,\hat{\mathbf{y}}$, i.e. $+x\to+y$ for small $\psi>0$ — the same CCW-about-$+z$ sense the module docstring and [tilt convention](../../physics/geometry/tilt-convention.md) assign to `tilt_azim_rad`. Pass (checked numerically in §5.4).

## 3. Independent derivation

### 3.1 Direct basis, metric tensor, and the Cartesian embedding

A rotation matrix is only defined once the lattice is embedded in Cartesian space. Take the standard crystallographic setting used by essentially every CIF consumer: with lengths $a,b,c$ and angles $\alpha=\angle(\mathbf{a}_2, \mathbf{a}_3)$, $\beta=\angle(\mathbf{a}_1,\mathbf{a}_3)$, $\gamma=\angle(\mathbf{a}_1,\mathbf{a}_2)$,

$$
\mathbf{a}_1=(a,0,0),\qquad
\mathbf{a}_2=(b\cos\gamma,\;b\sin\gamma,\;0),
$$

$$
\mathbf{a}_3=\left(c\cos\beta,\;
c\,\frac{\cos\alpha-\cos\beta\cos\gamma}{\sin\gamma},\;
c\,\frac{\sqrt{\Delta}}{\sin\gamma}\right),
$$

$$
\Delta=1-\cos^2\alpha-\cos^2\beta-\cos^2\gamma
+2\cos\alpha\cos\beta\cos\gamma,
$$

so that $\mathbf{a}_1\parallel\hat{\mathbf{x}}$, $\mathbf{a}_2$ lies in the $xy$ plane with a positive $y$ component, and $V=\mathbf{a}_1\cdot
(\mathbf{a}_2\times\mathbf{a}_3)=abc\sqrt{\Delta}>0$ (right-handed). The
metric tensor is $M_{ij}=\mathbf{a}_i\cdot\mathbf{a}_j$.

### 3.2 Reciprocal basis and the normalization convention

The reciprocal basis is defined by the duality relation

$$
\mathbf{a}_i\cdot\mathbf{b}_j=\kappa\,\delta_{ij},
\qquad
\mathbf{b}_1=\kappa\,\frac{\mathbf{a}_2\times\mathbf{a}_3}{V},\quad
\mathbf{b}_2=\kappa\,\frac{\mathbf{a}_3\times\mathbf{a}_1}{V},\quad
\mathbf{b}_3=\kappa\,\frac{\mathbf{a}_1\times\mathbf{a}_2}{V},
$$

with $\kappa=2\pi$ in the solid-state ("physics") convention and $\kappa=1$ in the crystallographic convention. Correspondingly $d_{hkl}=\kappa/ \lVert\mathbf{g}_{hkl}\rVert$, i.e. $2\pi/\lVert\mathbf{g}\rVert$ or $1/\lVert\mathbf{g}\rVert$.

The choice of $\kappa$ rescales every $\mathbf{b}_i$ by a common positive factor, hence rescales $\mathbf{g}_{hkl}$ by that same factor and leaves $\hat{\mathbf{g}}_{hkl}$ — the only thing $R$ depends on — **invariant**. So $R$ is convention-independent; only the magnitude of $\mathbf{g}$ used downstream (dispersion relation, Bragg condition) is convention-sensitive. For PyRITE the diffraction physics uses $\lvert\mathbf{g}\rvert=2\pi/d$, so $\kappa=2\pi$ is the expected convention; this is checked against `reciprocal_g_vector` in §5.1.

### 3.3 Why the plane normal is $\mathbf{g}_{hkl}$ and not $h\mathbf{a}_1+k\mathbf{a}_2+l\mathbf{a}_3$

By the Miller definition, the $(hkl)$ plane cuts the axes at $\mathbf{a}_1/h$, $\mathbf{a}_2/k$, $\mathbf{a}_3/l$. Two independent in-plane vectors are

$$
\mathbf{u}=\frac{\mathbf{a}_2}{k}-\frac{\mathbf{a}_1}{h},\qquad
\mathbf{v}=\frac{\mathbf{a}_3}{l}-\frac{\mathbf{a}_1}{h}.
$$

Using $\mathbf{a}_i\cdot\mathbf{b}_j=\kappa\delta_{ij}$,

$$
\mathbf{g}_{hkl}\cdot\mathbf{u}
=\kappa\left(\frac{k}{k}-\frac{h}{h}\right)=0,
\qquad
\mathbf{g}_{hkl}\cdot\mathbf{v}
=\kappa\left(\frac{l}{l}-\frac{h}{h}\right)=0,
$$

so $\mathbf{g}_{hkl}\perp(hkl)$ exactly, for every lattice, with no orthogonality assumption. (Zero indices are handled by the limit: a zero index means the plane is parallel to that axis, and $\mathbf{g}\cdot\mathbf{a}_i =\kappa\,(hkl)_i=0$ reproduces exactly that statement.)

The direct-space combination $\mathbf{t}_{hkl}=h\mathbf{a}_1+k\mathbf{a}_2 +l\mathbf{a}_3$ is a *lattice translation*, not a normal. Writing $\mathbf{a}_i=\kappa^{-1}\sum_j M_{ij}\mathbf{b}_j$ and $\mathbf{m}=(h,k,l) ^{\mathsf T}$,

$$
\mathbf{t}_{hkl}
=\kappa^{-1}\sum_j\big(M\mathbf{m}\big)_j\,\mathbf{b}_j ,
\qquad
\mathbf{g}_{hkl}=\sum_j m_j\,\mathbf{b}_j .
$$

Because $\{\mathbf{b}_j\}$ is a basis, $\mathbf{t}_{hkl}\parallel \mathbf{g}_{hkl}$ **iff**

$$
\boxed{\,M\mathbf{m}=\lambda\,\mathbf{m}\quad\text{for some }\lambda>0\,}
$$

i.e. iff $(h,k,l)$ is an eigenvector of the metric tensor. Special cases:

| lattice | $(h,k,l)$ | $M\mathbf{m}\parallel\mathbf{m}$? |
| --- | --- | --- |
| cubic, $M=a^2 I$ | any | yes (every direction) |
| orthogonal, $M=\operatorname{diag}(a^2,b^2,c^2)$ | one-axis $(1,0,0)$ etc. | yes |
| orthogonal | $(1,1,0)$ with $a\neq b$ | no |
| hexagonal, $\gamma=120^\circ$ | $(0,0,1)$ | yes ($c\perp$ basal plane) |
| hexagonal | $(1,0,0)$ | no: $M\mathbf{m}=(a^2,-a^2/2,0)$ |
| monoclinic $\beta\neq90^\circ$ | $(1,0,-1)$ | no |

So "the normal is $\mathbf{g}$" and "the normal is $\mathbf{t}$" coincide exactly when $\mathbf{m}$ is a metric eigenvector — in practice: any index in a cubic cell, and a single-nonzero index in an orthogonal cell. This is precisely the reduction asserted in the docstring, and it is why $(00l)$ cuts of the hexagonal catalog crystals (`hopg`, `hbn`) happen to agree with the legacy direct-axis path while e.g. `gep`'s monoclinic $(1,0,\bar1)$ does not.

### 3.4 Minimal rotation carrying $\hat{\mathbf{g}}$ onto $+\hat{\mathbf{z}}$

Let $\hat{\mathbf{g}}=\mathbf{g}/\lVert\mathbf{g}\rVert$ and $\hat{\mathbf{z}}=(0,0,1)$. Any rotation taking $\hat{\mathbf{g}}\to \hat{\mathbf{z}}$ can be written as $R_z(\psi)R_{\min}$; the minimal one rotates about the axis perpendicular to both, by the angle between them. Set

$$
\mathbf{k}=\hat{\mathbf{g}}\times\hat{\mathbf{z}},\qquad
s=\lVert\mathbf{k}\rVert=\sin\theta,\qquad
c=\hat{\mathbf{g}}\cdot\hat{\mathbf{z}}=\cos\theta,
$$

with $\theta=\operatorname{atan2}(s,c)\in[0,\pi]$. With $K=[\mathbf{k}]_\times$ the skew matrix of $\mathbf{k}$ (so $K\mathbf{x} =\mathbf{k}\times\mathbf{x}$), the Rodrigues formula for a rotation by $\theta$ about $\hat{\mathbf{k}}=\mathbf{k}/s$ is

$$
R_{\min}
=I+\sin\theta\,[\hat{\mathbf{k}}]_\times
+(1-\cos\theta)\,[\hat{\mathbf{k}}]_\times^2
=I+K+\frac{1-c}{s^2}\,K^2
=I+K+\frac{K^2}{1+c},
$$

using $s^2=1-c^2=(1-c)(1+c)$. Verification that this maps $\hat{\mathbf{g}}$ onto $\hat{\mathbf{z}}$, using the BAC--CAB identity:

$$
K\hat{\mathbf{g}}=(\hat{\mathbf{g}}\times\hat{\mathbf{z}})\times
\hat{\mathbf{g}}=\hat{\mathbf{z}}-c\,\hat{\mathbf{g}},\qquad
K^2\hat{\mathbf{g}}=K(\hat{\mathbf{z}}-c\hat{\mathbf{g}})
=(-\hat{\mathbf{g}}+c\hat{\mathbf{z}})-c(\hat{\mathbf{z}}-c\hat{\mathbf{g}})
=(c^2-1)\hat{\mathbf{g}},
$$

hence

$$
R_{\min}\hat{\mathbf{g}}
=\hat{\mathbf{g}}+\hat{\mathbf{z}}-c\hat{\mathbf{g}}
+\frac{(c^2-1)}{1+c}\hat{\mathbf{g}}
=\hat{\mathbf{g}}+\hat{\mathbf{z}}-c\hat{\mathbf{g}}+(c-1)\hat{\mathbf{g}}
=\hat{\mathbf{z}} .
$$

**Properness.** $R_{\min}=\exp(\theta[\hat{\mathbf{k}}]_\times)$ is the exponential of a real skew matrix, so $R_{\min}^{\mathsf T} =\exp(-\theta[\hat{\mathbf{k}}]_\times)=R_{\min}^{-1}$ and $\det R_{\min}=\exp(\theta\operatorname{tr}[\hat{\mathbf{k}}]_\times) =\exp(0)=+1$. Equivalently $\operatorname{tr}R_{\min}=1+2c$, the standard $1+2\cos\theta$. Handedness is therefore preserved; no reflection is introduced.

**Degenerate cases.** The axis $\mathbf{k}$ vanishes when $s=0$:

- $c=+1$ ($\hat{\mathbf{g}}=+\hat{\mathbf{z}}$): $\theta=0$ and the correct answer is $R_{\min}=I$. The closed form $I+K+K^2/(1+c)$ already returns $I$ here since $K=0$, so this case is benign but should be guarded for exactness.
- $c=-1$ ($\hat{\mathbf{g}}=-\hat{\mathbf{z}}$): $\theta=\pi$, $K=0$, and the $1/(1+c)$ term is $0/0$. The rotation axis is undefined — every axis in the $xy$ plane works — so the implementation must *choose* one. Any choice $\hat{\mathbf{k}}=(\cos\varphi_0,\sin\varphi_0,0)$ gives the proper matrix

  $$
  R_{\min}=I+2[\hat{\mathbf{k}}]_\times^2
  =2\hat{\mathbf{k}}\hat{\mathbf{k}}^{\mathsf T}-I ,
  $$

e.g. $\varphi_0=0$ gives $R_{\min}=\operatorname{diag}(1,-1,-1)$. All such choices differ from one another by a roll about $z$, i.e. by a relabelling of the azimuth origin, which is exactly the freedom the `azimuth_rad` argument parametrizes. The forbidden answer is $-I$: it maps $-\hat{\mathbf{z}}$ to $+\hat{\mathbf{z}}$ but has $\det=-1$ and flips handedness.

### 3.5 Azimuth composition and handedness

The residual freedom after $R_{\min}$ is a roll about $+z$. The right-handed (CCW seen from $+z$) rotation by $\psi$ is

$$
R_z(\psi)=
\begin{pmatrix}
\cos\psi & -\sin\psi & 0\\
\sin\psi & \phantom{-}\cos\psi & 0\\
0 & 0 & 1
\end{pmatrix},
$$

and the claimed composition, "minimal rotation first, then the roll", acting on column vectors means

$$
\boxed{\,R=R_z(\psi)\,R_{\min}\,}\qquad
\mathbf{g}'=R\,\mathbf{g}.
$$

The order matters: $R_{\min}R_z(\psi)$ would roll the crystal about the *pre-rotation* $z$ axis, which is not the slab normal, and would in general no longer put $\mathbf{g}$ on $+z$. The chosen order does:

$$
R\,\mathbf{g}=R_z(\psi)\big(\lVert\mathbf{g}\rVert\hat{\mathbf{z}}\big)
=\lVert\mathbf{g}\rVert\,\hat{\mathbf{z}},
$$

since $R_z$ fixes $\hat{\mathbf{z}}$. $\det R=\det R_z\cdot\det R_{\min} =+1$, and $R R^{\mathsf T}=I$.

For the `beam_uvw` path the derivation is identical with $\hat{\mathbf{g}}$ replaced by the unit vector along $u\mathbf{a}_1+ v\mathbf{a}_2+w\mathbf{a}_3$.

### 3.6 Summary of the independent expression

$$
R(h,k,l,\psi)=R_z(\psi)\left[I+K+\frac{K^2}{1+\hat{\mathbf{g}}\cdot
\hat{\mathbf{z}}}\right],\qquad
K=[\hat{\mathbf{g}}\times\hat{\mathbf{z}}]_\times,\qquad
\hat{\mathbf{g}}=\frac{\sum_i(hkl)_i\mathbf{b}_i}
{\lVert\sum_i(hkl)_i\mathbf{b}_i\rVert},
$$

with the $\hat{\mathbf{g}}=-\hat{\mathbf{z}}$ case replaced by a $\pi$-rotation about a chosen in-plane axis, and $R=\mathrm{None}$ (identity, construction frame) permitted when no orientation and no azimuth are set.

## 4. Comparison with the implementation

Read after §3. The implementation is `_orientation_R` plus two helpers it delegates to, `materials/crystal.py::_reciprocal_basis` / `::reciprocal_g_vector` and `materials/crystal.py::_rotation_between`.

Literal source (the load-bearing lines):

```python
    elif surface_hkl is not None:
        axis, axis_norm = reciprocal_g_vector(surface_hkl, lattice)
        R = _rotation_between(axis / axis_norm, np.array([0.0, 0.0, 1.0]))
    if azimuth_rad:
        ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
        Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        R = Rz if R is None else Rz @ R
```

```python
        B = 2.0 * np.pi * np.array([_cross3(a2, a3), _cross3(a3, a1), _cross3(a1, a2)]) / V
...
    g_vec = np.asarray(hkl, dtype=float) @ _reciprocal_basis(lattice)
```

```python
    c = float(np.dot(u_hat, t_hat))
    axis = _cross3(u_hat, t_hat)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel: 180 deg about any axis perpendicular to u_hat
        tmp = np.array([1.0, 0.0, 0.0]) if abs(u_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _cross3(u_hat, tmp)
        axis /= np.linalg.norm(axis)
        K = np.array(...)
        return np.eye(3) + 2.0 * K @ K
    axis = axis / s
    K = np.array(...)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)
```

Term-by-term:

| derivation (§3) | implementation | agrees |
| --- | --- | --- |
| $\mathbf{b}_i=\kappa(\mathbf{a}_j\times\mathbf{a}_k)/V$, $\kappa=2\pi$ | `B = 2.0 * np.pi * [...] / V`, rows $\mathbf{b}_1,\mathbf{b}_2,\mathbf{b}_3$ | yes, $\kappa=2\pi$ (physics convention, $\lvert\mathbf{g}\rvert=2\pi/d_{hkl}$) |
| $\mathbf{g}_{hkl}=\sum_i(hkl)_i\mathbf{b}_i$ | `hkl @ _reciprocal_basis(lattice)` (row vector times row-stacked basis) | yes |
| Cartesian embedding of §3.1 | `_direct_lattice_vectors`, `"general"` branch, with $c_z=\sqrt{c^2-c_x^2-c_y^2}$ | yes: $\sqrt{c^2-c_x^2-c_y^2}=c\sqrt{\Delta}/\sin\gamma$ identically |
| $R_{\min}=I+K+K^2/(1+c)$, $K=[\hat{\mathbf{g}}\times\hat{\mathbf{z}}]_\times$ | `I + s*K + (1-c)*(K@K)` with `K` the skew of the **normalized** axis | yes, algebraically identical: with $K=s\hat{K}$, $s^2/(1+c)=(1-c^2)/(1+c)=1-c$. The code's form is the numerically better-conditioned one (no $1/(1+c)$) |
| direction: $R\hat{\mathbf{g}}=\hat{\mathbf{z}}$ | `_rotation_between(g_hat, z)` documented as `R @ u_hat = t_hat` | yes (not the inverse) |
| $c=+1$ case $\to I$ | `if c > 0: return np.eye(3)` | yes |
| $c=-1$ case $\to 2\hat{\mathbf{k}}\hat{\mathbf{k}}^{\mathsf T}-I$, proper | `np.eye(3) + 2.0 * K @ K` with $\hat{\mathbf{k}}\perp\hat{\mathbf{g}}$; for $\hat{\mathbf{g}}=-\hat{\mathbf{z}}$ the tie-break `tmp = x_hat` gives $\hat{\mathbf{k}}=-\hat{\mathbf{y}}$ and $R=\operatorname{diag}(-1,1,-1)$ | yes; $I+2\hat K^2=2\hat{\mathbf{k}}\hat{\mathbf{k}}^{\mathsf T}-I$, and it is **not** $-I$ |
| $R_z(\psi)$ right-handed about $+z$ | `[[ca,-sa,0],[sa,ca,0],[0,0,1]]` | yes, $+x\to+y$ for $\psi>0$ |
| composition $R=R_z(\psi)R_{\min}$ | `R = Rz @ R` | yes (left-multiplication, column-vector action) |
| action on reciprocal vectors | `g_vec = R_orient @ g_vec` in `spectrum/cxr/_per_hkl.py:665`, `_batched.py:235`, and `detector.py::mosaic_psi_rad` | yes, consistent column-vector convention everywhere |
| `recip_miscut_rad=None` is a no-op | `if recip_miscut_rad is not None:` and then `if mp:` | yes; `(0.0, anything)` also collapses to the `None` path |
| mutual exclusivity | `raise ValueError("beam_uvw and surface_hkl are mutually exclusive")` | yes |

No symbolic divergence in any factor, sign, exponent, or convention.

Two convention notes (not discrepancies):

1. The $\hat{\mathbf{g}}=-\hat{\mathbf{z}}$ tie-break picks $\hat{\mathbf{k}}$ from `u_hat[0]`, so the in-plane azimuth origin of the antiparallel branch is an arbitrary (but proper and deterministic) choice, and it does not agree with the limit of the generic branch as $\hat{\mathbf{g}}\to-\hat{\mathbf{z}}$ along an arbitrary path. That discontinuity is intrinsic to the degeneracy, not to this code: only `azimuth_rad` can fix the in-plane origin there.
2. The branch threshold is $s<10^{-12}$, i.e. within $\sim10^{-12}$ rad of (anti)parallel. Because the generic branch uses $(1-c)\hat K^2$ rather than $K^2/(1+c)$, it stays well-conditioned right up to the threshold, so the cut is not a numerical cliff.

## 5. Numeric checks

All numbers below come from an independent NumPy construction — my own `direct_vectors`/`recip_vectors`/Rodrigues from §3, fed with the six lattice parameters read directly from the bundled CIFs — compared against `_orientation_R`. No PyRITE helper is used on the independent side.

Cells used: `hopg` ($a=2.461$, $c=6.711$ Å, $\gamma=120^\circ$), `hbn` ($a=2.504$, $c=6.661$ Å, $\gamma=120^\circ$), `gep` (monoclinic $C2/m$, $a=15.1948$, $b=3.6337$, $c=9.1941$ Å, $\beta=101.239^\circ$), `silicon` (cubic).

### 5.1 Normalization convention

$\lvert\mathbf{g}_{001}\rvert$ for `hopg`: independent $2\pi$-convention value $0.936251722125$ Å$^{-1}$, `reciprocal_g_vector` $0.936251722125$ Å$^{-1}$, $2\pi/c=0.936251722125$ Å$^{-1}$; componentwise agreement $2.2\times10^{-16}$. Same for `hbn`. So $\kappa=2\pi$ is confirmed, and $d_{001}=2\pi/\lvert \mathbf{g}\rvert=c$ as required. For `gep`, $\lvert\mathbf{g}_{001}\rvert =0.696755$ Å$^{-1}\neq2\pi/c=0.683393$ Å$^{-1}$ — correct, since $d_{001}=c\sin\beta$ for a monoclinic cell, another independent confirmation of the reciprocal construction.

(Incidentally, I typed $a=5.431$ Å for silicon while the CIF has $5.4309$ Å. The $\lvert\mathbf{g}\rvert$ comparison then differs by $2\times10^{-5}$, but every rotation matrix below still matches to $10^{-16}$ — a free empirical confirmation of §3.2's claim that $R$ is invariant under a uniform rescaling of $\mathbf{b}_i$.)

### 5.2 Reciprocal vs direct normal on non-orthogonal cells

Angle between $\mathbf{g}_{hkl}$ and $\mathbf{t}_{hkl}=h\mathbf{a}_1+ k\mathbf{a}_2+l\mathbf{a}_3$, with $\lVert M\mathbf{m}\times\mathbf{m}\rVert$ as the metric-eigenvector residual from §3.3:

| cell | $(hkl)$ | $\angle(\mathbf{g},\mathbf{t})$ | $\lVert M\mathbf{m}\times\mathbf{m}\rVert$ |
| --- | --- | --- | --- |
| hopg | $(001)$ | $0.00000^\circ$ | $1.4\times10^{-15}$ |
| hopg | $(100)$ | $30.00000^\circ$ | $3.03$ |
| hopg | $(10\bar1)$ | $55.36424^\circ$ | $39.2$ |
| hbn | $(100)$ | $30.00000^\circ$ | $3.14$ |
| gep | $(001)$ | $11.23900^\circ$ | $27.2$ |
| gep | $(10\bar1)$ | $28.10376^\circ$ | $146.4$ |
| silicon | any of $(001),(100),(10\bar1)$ | $0.00000^\circ$ | $0$ |

The zero-angle rows are exactly the metric-eigenvector rows, as derived. Two independent closed-form confirmations: the hexagonal $(100)$ offset is exactly $30^\circ$ (the angle between $\mathbf{a}^*$ and $\mathbf{a}$ for $\gamma=120^\circ$), and the monoclinic $(001)$ offset is exactly $\beta-90^\circ=11.239^\circ$. The catalog's `gep` entry uses `surface_hkl = [1, 0, -1]`, so the legacy direct-axis path would have been wrong there by $28.1^\circ$.

### 5.3 Rotation correctness, independent vs implementation

For every combination of $\{$`hopg`, `hbn`, `gep`, `silicon`$\}\times \{(001),(100),(10\bar1),(00\bar1)\}$ and $\psi\in\{0^\circ,37^\circ\}$:

- $R\,\hat{\mathbf{g}}=(0,0,1)$ with transverse residual $\le4\times10^{-16}$ and $z$-component $1.000000000000000$;
- $\lVert RR^{\mathsf T}-I\rVert_\infty\le1.1\times10^{-15}$;
- $\det R=1.000000000000000$ (worst case $1.000000000000001$);
- $\lVert R_{\text{independent}}-R_{\text{code}}\rVert_\infty\le 6.7\times10^{-16}$.

The hardest case, `gep` $(10\bar1)$ at $\psi=37^\circ$ (monoclinic, non-eigenvector index, non-zero azimuth), matches at $6.7\times10^{-16}$.

### 5.4 Azimuth handedness and composition order

$R_z(30^\circ)\hat{\mathbf{x}}=(0.866025403784439,\,0.5,\,0)$ from both the independent construction and `_orientation_R(..., azimuth_rad=\pi/6, ...)` with no orientation set: $+x\to+y$, a right-handed / CCW roll about $+z$, matching the module docstring and the `tilt_azim_rad` sense in [tilt-convention](../../physics/geometry/tilt-convention.md).

Composition order, `gep` $(10\bar1)$, $\psi=37^\circ$:

| candidate | $\lVert R_{\text{code}}-\text{candidate}\rVert_\infty$ |
| --- | --- |
| $R_z(\psi)R_{\min}$ (derived) | $6.7\times10^{-16}$ |
| $R_{\min}R_z(\psi)$ (wrong order) | $1.10$ |

So the order is resolved unambiguously and matches §3.5.

### 5.5 Orthogonal one-axis reduction, and its failure off-axis

| cell | `surface_hkl` vs `beam_uvw` | $\lVert R_{hkl}-R_{uvw}\rVert_\infty$ |
| --- | --- | --- |
| silicon | $(001)$ vs $[001]$ | $0$ |
| silicon | $(100)$ vs $[100]$ | $6.1\times10^{-17}$ |
| hopg | $(001)$ vs $[001]$ | $0$ |
| hopg | $(100)$ vs $[100]$ | $0.500$ |

The first three confirm the docstring's "for an orthogonal cell a one-axis `surface_hkl` reduces to its parallel direct axis" (and the hexagonal $c$ axis, which is also a metric eigenvector). The last confirms the reduction genuinely fails off the eigenvector — the $0.5$ is the matrix-element signature of the $30^\circ$ difference from §5.2, i.e. the exact error the reciprocal path removes.

### 5.6 Degenerate antiparallel case

`surface_hkl = (0, 0, -1)` on every cell tested returns

```text
R = [[-1.,  0.,  0.],
     [ 0.,  1.,  0.],
     [ 0.,  0., -1.]]
```

with $\det R=+1$, $RR^{\mathsf T}-I=0$ exactly, $R\hat{\mathbf{g}}=(0,0,1)$ exactly, and `R == -I` is `False`. This is the $\pi$ rotation about $\hat{\mathbf{y}}$, i.e. $2\hat{\mathbf{k}}\hat{\mathbf{k}}^{\mathsf T}-I$ with $\hat{\mathbf{k}}=\pm\hat{\mathbf{y}}$ — exactly the admissible family derived in §3.4, and specifically not the improper $-I$.

### 5.7 Guards and defaults

- `beam_uvw` and `surface_hkl` together raise `ValueError: beam_uvw and surface_hkl are mutually exclusive`.
- `(None, None, azimuth_rad=0.0)` returns `None` (construction frame).
- `recip_miscut_rad=(0.0, 0.0)` returns `None` — the claimed bit-for-bit no-op.

## 6. Downstream consistency

`_orientation_R` defines the *sample* frame; `tilted_geometry` assumes that frame has the slab normal along $+z$ and rotates the beam and detector into it via `R.T`. The two are consistent by construction: $R_{\text{orient}}$ puts $\mathbf{g}_{hkl}$ on $+z$, which is what `tilted_geometry`'s docstring calls "slab normal and crystal construction frame along +z".

The consumers apply $R$ identically and on the correct side:

- `montecarlo/spectrum/cxr/_per_hkl.py:665` and `_batched.py:235` — `g_vec = R_orient @ g_vec`;
- `montecarlo/detector.py::mosaic_psi_rad` — `g_vec = R @ g_vec`, with `beam_dir`, `n_hat` taken from `tilted_geometry` in the same frame.

Tilt-sign check with a *non-orthogonal* orientation, `gep` with `surface_hkl = (1,0,-1)` and its $(20\bar2)$ reflection, at $\theta_{\rm obs}=119^\circ$:

| $\theta_{\rm tilt}$ | $\hat{\mathbf{g}}\cdot\hat{\mathbf{n}}$ (this run) | documented value |
| --- | --- | --- |
| $+10^\circ$ | $-0.3256$ | $-0.326$ |
| $-10^\circ$ | $-0.6293$ | $-0.630$ |

These reproduce the numbers in [tilt-convention](../../physics/geometry/tilt-convention.md) exactly, so positive polar tilt still moves $\mathbf{g}$ toward the detector after the reciprocal orientation rotation. The rotated $(20\bar2)$ direction is $(6.2\times10^{-17},\,-2.7\times10^{-33},\,1)$, i.e. $\mathbf{g}\parallel \hat{\mathbf{n}}$ at zero tilt as the tilt convention assumes. No sign or handedness conflict between `_orientation_R` and the downstream tilt convention.

## 7. Verdict

The independent derivation reproduces the implementation exactly: the same $\kappa=2\pi$ reciprocal basis, the same reciprocal (not direct) plane normal, the same Rodrigues minimal rotation with $R\hat{\mathbf{g}}=\hat{\mathbf{z}}$, the same proper $\det=+1$ antiparallel special case (not $-I$), the same right-handed azimuth, and the same composition order $R=R_z(\psi)R_{\min}$ acting on column vectors. Agreement is $\le7\times10^{-16}$ in every matrix element across four crystal systems including two hexagonal cells and one monoclinic cell with a non-eigenvector Miller index. The convention is consistent with the downstream `tilted_geometry` tilt sign.

**Status: `rederived`** (independent derivation matches; anchors listed in the ledger exist; human sign-off still outstanding).

## 8. Miscut composition extension (#25)

*Implementation-context derivation; fresh-context verdict in §9. §§1--7
above validate the axis and roll terms only; there the miscut was checked
solely as a `None` no-op.*

`Slab(cut=CrystalCut(...))` now sets `recip_miscut_rad = (θ, φ)` on real
cases (`campaign/geometry.py::resolve_crystal_cut`), so the miscut term of
`_orientation_R` is now production physics.

**Claim.** With miscut polar angle $\theta$ and sample-frame azimuth $\phi$
(radians in the case, degrees at the API), define

$$
\hat{\mathbf m}=(\sin\theta\cos\phi,\;\sin\theta\sin\phi,\;\cos\theta),
\qquad
R_m = \text{minimal rotation } \hat{\mathbf z}\to\hat{\mathbf m}
    = \text{rotation by } \theta \text{ about } (-\sin\phi,\cos\phi,0).
$$

The lattice rotation applied to every reciprocal vector is

$$
R = R_m\,R_z(\psi)\,R_{\min},
$$

so $R\,\hat{\mathbf g}_{\text{axis}}=\hat{\mathbf m}$. The selected cut axis
makes the angle $\theta$ with the face normal $+\hat{\mathbf z}$, tilted toward
azimuth $\phi$. The roll $\psi$ fixes the in-plane lattice orientation before
the tilt, and $\phi$ is measured in the sample frame independently of $\psi$.

**Assumptions.** The lattice is rigid (one rotation for every reciprocal
vector) and the face is flat. The face normal, beam and detector directions are
set separately by `tilted_geometry` and are not moved by $R$. Mosaic rotations
act after $R$ in the same sample frame. $0\le\theta<90^\circ$ keeps the cut axis
on the beam-facing hemisphere.

**Units, signs and limits.**

- $R$ is dimensionless; only $\cos$/$\sin$ of the angles enter. A product of
  proper rotations is proper, so $\det R=+1$.
- $\theta=0$: $\hat{\mathbf m}=\hat{\mathbf z}$ and $R_m=I$. The code skips
  the term, so the case keeps its historical payload (`recip_miscut_rad` is
  absent when `miscut_deg == 0`).
- For small $\theta$, $R\,\hat{\mathbf g}_{\text{axis}} \approx
  \hat{\mathbf z}+\theta(\cos\phi,\sin\phi,0)$, which is positive $\theta$
  toward $+x$ at $\phi=0$. This matches the `tilted_geometry` spherical
  convention.
- A vacuum line check: HOPG (002), 30 keV, tilt $30^\circ$, observation
  $90^\circ$. Uncut, the b–g angle is $30^\circ$ and $E\approx1050.8$ eV.
  With `miscut_deg=10`, $\phi=0$ the b–g angle is $40^\circ$ and
  $E\approx929$ eV; a 20-trial `pr.simulate` peaks at 928 eV on the
  catalog-spaced grid.

**Anchors.**
`tests/campaign/test_crystal_cut.py::test_case_rotation_matches_an_independent_construction`
covers nonorthogonal HOPG with `surface_hkl=(1,0,1)`, roll and miscut: the
runtime matrix built from the case keys matches an independent Rodrigues
construction to $10^{-13}$, and the cut axis lands on $\hat{\mathbf m}$.
`::test_cut_front_cancels_independent_ray_phases` shows that bunch-front
matching on a cut target cancels independently evaluated ray phases with the
same independently rotated $\mathbf g$.

## 9. Fresh-context verdict on the miscut extension (#25)

Verifier: fresh context, Claude Sonnet 5.5, 2026-10-14. Scope: the per-target `CrystalCut` lowering and the continuous miscut term. Independence caveat: the ledger row and the `CrystalCut`/`resolve_crystal_cut`/`_orientation_R` docstrings were read first, but the `_orientation_R` body was displayed together with its docstring by an overly wide line range before the derivation below was written down. The derivation is therefore standard textbook geometry written from the stated claim, and the numerical check in 9.3 uses a separately constructed reference (own reciprocal basis, own Rodrigues routine, no implementation helper) to compensate.

### 9.1 Derivation (from the claim and signatures only)

Lowering of angles: a roll $\psi$ and a miscut $(\theta,\phi)$ in degrees lower to radians, $\psi=\mathrm{radians}(\texttt{roll\_deg})$ and $(\theta,\phi)=\mathrm{radians}(\texttt{miscut\_deg},\texttt{miscut\_azim\_deg})$. Keys are omitted when zero.

Rodrigues: for unit axis $\hat k$ and angle $\alpha$, with $K=[\hat k]_\times$,

$$
R(\hat k,\alpha)=I+\sin\alpha\,K+(1-\cos\alpha)K^2 .
$$

Minimal rotation of unit $\hat u$ onto $\hat t$: $\hat k=\hat u\times\hat t/|\hat u\times\hat t|$, $\cos\alpha=\hat u\cdot\hat t$. It is proper, and the identity when $\hat u=\hat t$.

The selected axis is $\hat g_{\rm axis}$ ($\mathbf g=h\mathbf b_1+k\mathbf b_2+l\mathbf b_3$, $\mathbf b_i=2\pi\,\mathbf a_j\times\mathbf a_k/V$) or a direct $u\mathbf a_1+v\mathbf a_2+w\mathbf a_3$. The composition is

$$
R=R_m\,R_z(\psi)\,R_{\min}(\hat g_{\rm axis}\to\hat z),\qquad
R_m=R_{\min}(\hat z\to\mathbf m),\ \ \mathbf m=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta).
$$

Consequences:

1. $R_{\min}\hat g_{\rm axis}=\hat z$, and $R_z\hat z=\hat z$, so $R\hat g_{\rm axis}=R_m\hat z=\mathbf m$. The face normal $\hat z$ stays fixed, so the lattice plane normal sits at angle $\theta$ from the face normal: a cut $\theta$ off the plane.
2. $\theta=0$ gives $R_m=I$ exactly (the aligned branch returns the identity), so the result is bit-for-bit the unmiscut rotation. If the lowering omits the key at zero, no extra multiplication occurs.
3. $\det R=+1$, as every factor is proper.
4. $R_m$ rotates about $\hat z\times\mathbf m\propto(-\sin\phi,\cos\phi,0)$, so for small $\theta$, $R_m\hat z\approx(\theta\cos\phi,\theta\sin\phi,1)$: toward $+x$ at $\phi=0$. This matches the spherical form $(\sin t\cos a,\sin t\sin a,\cos t)$ used for the tilt in the sample frame.
5. Order matters. Miscut applied before roll gives $R_z(\psi)R_mR_{\min}$, which differs (the miscut azimuth would be rotated by $\psi$). The claimed order, roll first (lattice-fixed) then miscut (face-fixed azimuth), is the physically sensible one: $\phi$ is a sample-frame azimuth, independent of roll.
6. The beam, detector and face normal come from `tilted_geometry` with the tilt measured from $\hat z$, so $R$ only reorients reciprocal vectors, $\mathbf g\mapsto R\mathbf g$. Every consumer must apply the same $R$, including front matching.

Cheap filters: units are degrees to radians once, at the lowering boundary, with $0\le\theta<90^\circ$ enforced (so $\mathbf m$ is unique and the minimal rotation is never antiparallel). Limits: $\theta\to0$ is the identity, $\psi=0$ is the plain axis alignment, and an axis already along $\hat z$ with $\psi=0$ gives $R=R_m$. Signs: right-handed roll, and $+x$ at $\phi=0$. All pass.

### 9.2 Comparison with the implementation

- `materials/crystal.py::_orientation_R` (lines 705-745): `R = Rz @ R_axis`, then `R = R_miscut @ R`, with `R_miscut = _rotation_between(z, m)` and `m = (st cos ma, st sin ma, ct)`. The miscut is skipped when `mp` is 0. `Rz` is the right-handed matrix `[[c,-s],[s,c]]`. This is identical to 9.1, including the order. For the axis, `surface_hkl` uses `reciprocal_g_vector` (rows of the reciprocal basis, $2\pi$ convention) and `beam_uvw` uses the direct vectors. The two spellings are mutually exclusive.
- `campaign/geometry.py::resolve_crystal_cut` (lines 418-452): `azimuth_rad = math.radians(roll_deg)` and `recip_miscut_rad = (radians(miscut_deg), radians(miscut_azim_deg))` only when `miscut_deg != 0`. `ResolvedCut.case_keys` emits `azimuth_rad` only if nonzero and `recip_miscut_rad` only if not `None`. `ResolvedCut.rotation` calls the shared `_orientation_R` with exactly these fields. A cut axis replaces the catalog axis, and the two spellings stay exclusive. `CrystalCut.__post_init__` enforces $0\le\theta<90^\circ$, finiteness and single scalars. Matches.
- `_TargetGeometry.lower` (geometry.py:606-670): writes `orientation.case_keys()` into each `LoweredTarget.crystal_orientation`, so the keys appear on every geometry point. The label is appended to the case name. Stack layers are built with `azimuth_rad=0.0` for the film radiator, but `CrystalCut` is rejected for substrate/stack (geometry.py:925), so the omission is unreachable.
- `campaign/sweep.py::build_cases` (lines 840-844, 1019, 1051): a cut axis plus a `Sweep.beam_uvw` override is rejected (so only one axis source). The resolved `beam_uvw`/`surface_hkl` is used for the case, and targeted (non-Gaussian) longitudinal modes are rejected with a miscut because they assume $\mathbf g\parallel\hat z$. This is a conservative guard.
- `campaign/front_matching.py::phase_matched_front_for_scene` (lines 55-58): calls the same `resolve_crystal_cut(target.cut, crystal.beam_uvw, crystal.surface_hkl).rotation(lattice)` and applies `g = orientation @ g`. It is the identical function and arguments to the case path. (Its beam-axis override semantics: a Sweep-level `beam_uvw` is not visible to the scene helper, an existing pre-#25 limitation, since the helper reads the catalog axis only.)
- Runtime consumers: `runner/case_driver.py::_cxr_for_segments_once`, `spectrum/cxr/_setup.py:270-276` (and `_batched.py:218-245`, whose table cache key includes both fields), `spectrum/reflection_rows.py:130-133`, `montecarlo/detector.py:94-100`, `spectrum/cxr_seeds.py:840-841,911-912` and `campaign/kinematic_validity.py:57,191-192` all read `azimuth_rad` and `recip_miscut_rad` from the case, or from the per-radiator override falling back to the case, and pass them positionally in the same order `(lattice, beam_uvw, azimuth, miscut, surface_hkl)` to the single `_orientation_R`. `kinematic_validity` applies `g @ rotation.T`, i.e. $R\mathbf g$ per row. No consumer reconstructs the rotation independently.
- `materials/emission.py::photon_energies` (line 222) passes only the catalog axis with azimuth 0 and no miscut. It has no cut parameter, so it is a catalog-only estimator, not a consumer of the case keys. Note: it would not reflect a `CrystalCut`. This is a scope note, not a discrepancy. `campaign/model.py:184` documents "catalog orientation", which is also stale after #25 because the code uses the target cut. Minor documentation wording only.

### 9.3 Numerical check (HOPG, nonorthogonal)

HOPG is $a=b=2.461$, $c=6.711$ Å, $\gamma=120^\circ$. The check used `surface_hkl=(1,0,1)`, roll 37 deg, miscut 2.3 deg toward 110 deg. The reference reciprocal basis, Rodrigues routine and composition were written from scratch, with scripts in the scratchpad (not in the repository). Reference $\mathbf g_{101}=(2.5531,1.4740,0.9363)$ Å$^{-1}$ equals `reciprocal_g_vector` to all printed digits.

| Check | Result |
| --- | --- |
| lowered keys vs $(\psi,\theta,\phi)$ | exact (difference 0.0) |
| $\max\lvert R_{\rm impl}-R_{\rm ref}\rvert$ | $4.0\times10^{-16}$ |
| $\det R$, $\max\lvert RR^T-I\rvert$ | $1.0$, $2.2\times10^{-16}$ |
| $R\hat g_{101}-\mathbf m$ | $\lesssim1.6\times10^{-16}$ per component |
| angle of $R\mathbf g_{101}$ to $\hat z$ | $2.300000000000175^\circ$ |
| 5 reflections, $\max\lvert R\mathbf g-R_{\rm ref}\mathbf g\rvert$ | $2.1\times10^{-15}$ |
| $\theta=0$ cut vs direct `_orientation_R(roll only)` | `np.array_equal` True, and `recip_miscut_rad is None` |
| default `CrystalCut()` | rotation `None`, no case keys |
| roll right-handed: $R(\psi)=R_z(\psi)R(0)$ | difference 0.0 |
| swapped order (miscut before roll) vs impl | differs by 0.021 (so the order is genuinely tested) |
| small $\theta=0.01$, $\phi=0$, axis $\|\hat z$ | $R\hat g/|g|=(0.0099998,0,0.99995)$, toward $+x$ |
| direct `beam_uvw=(1,0,1)` with same roll and miscut | $4.2\times10^{-16}$ from reference, $R\hat a_{\rm axis}=\mathbf m$ |

`tilted_geometry` at tilt 0.5 deg, azimuth 0 returns the beam as $(-\sin t,0,\cos t)$ in the sample frame, i.e. the sample normal leans toward $+x$ in the lab, consistent with the spherical $(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)$ convention.

### 9.4 Anchor tests

`PYRITE_MC_BACKEND=cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/campaign/test_crystal_cut.py tests/montecarlo/transport/test_surface_orientation.py`: 41 passed in 2.4 s. No heavy Monte Carlo was run.

### 9.5 Verdict

Re-derivation matches: **rederived** for the miscut composition $R=R_mR_z(\psi)R_{\min}$, the unit lowering, the zero-angle no-op, $\det R=+1$, the $+x$ at $\phi=0$ convention, and identical use of the resolved rotation in transport, spectrum, detector, kinematic-validity and front matching. No discrepancy in the claimed physics. Non-blocking notes: (i) the stale "catalog orientation" wording in `campaign/model.py:184`; (ii) `photon_energies` takes no cut; (iii) the independence caveat in the preamble. Human sign-off remains pending (#277). Suggested ledger edit: add a note that the #25 miscut term is `rederived` by this write-up (§9), keeping the row at `anchored` and not `signed-off`.
