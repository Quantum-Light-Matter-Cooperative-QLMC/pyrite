# `pixel-angular-interpolation`

**Claim.** Under `PixelScorer(reconstruction="bilinear_tile")`, pixel $p$ takes the intrinsic density $I_p(E)=\sum_k w_{pk}I_k(E)$ over at most four angular tiles, with separable clamped bilinear weights in detector-local coordinates. The knots are the tile-column means of the local $x$, and the tile-row means of the local $y$, of the detector-plane intersections of the tile representative directions. The pixel flux is then $F_p(E)=I_p(E)\,\Delta\Omega_p\,T_p(E)$.

**Code.** `src/pyrite/instrument/geometry.py::angular_tile_weights` (helper `_axis_weights`); `src/pyrite/results/model.py::SpatialResult._materialize` (`bilinear_tile` branch) and `SpatialResult._blend`. **Anchor.** `tests/instrument/test_geometry.py`; `tests/results/test_spatial_model.py`; `tests/observations/test_store.py`. **Source.** Press et al., *Numerical Recipes* 3rd ed. (2007) §3.6.1, Eq. 3.6.5 (tensor-product linear interpolation on a rectilinear grid); ray–plane intersection; tile representatives from `instrument/geometry.py::angular_tiles`. **Verifier context.** Fresh session; did not author the implementation. The derivation below was written from the ledger row, the docstrings and signatures, `angular_tiles`, `planar_detector_rays`, and `PlanarPose` only, before the bodies of `_axis_weights`, `angular_tile_weights`, `_blend`, and `_materialize` were read.

## Independent derivation

### Geometry and conventions

The source sits at the lab origin. `PlanarPose` gives the plane centre $\mathbf c$, unit normal $\hat{\mathbf n}$ (local $+z$), local $\hat{\mathbf x}$, and $\hat{\mathbf y}=\hat{\mathbf n}\times\hat{\mathbf x}$, an orthonormal right-handed frame. `planar_detector_rays` places pixel $(i,j)$ (row $i$, column $j$) of an $n_y\times n_x$ grid with pitch $(p_y,p_x)$ at

$$
\mathbf c_{ij}=\mathbf c+y_i\,\hat{\mathbf y}+x_j\,\hat{\mathbf x},
\qquad
x_j=\Bigl(j-\tfrac{n_x-1}{2}\Bigr)p_x,
\qquad
y_i=\Bigl(i-\tfrac{n_y-1}{2}\Bigr)p_y .
$$

So **row index ↔ local $y$, column index ↔ local $x$**, and both local coordinates increase with index. Every pixel centre lies in the plane, so

$$
\mathbf c_{ij}\cdot\hat{\mathbf n}=\mathbf c\cdot\hat{\mathbf n}\equiv h,
$$

and the forward-facing requirement $\hat{\mathbf u}_{ij}\cdot\hat{\mathbf n}>0$ forces $h>0$. The pixel direction is $\hat{\mathbf u}_{ij}=\mathbf c_{ij}/r_{ij}$ with $r_{ij}=\lVert\mathbf c_{ij}\rVert$, and the point-sample solid angle is $\Delta\Omega_{ij}=A\,(\hat{\mathbf u}_{ij}\cdot\hat{\mathbf n})/r_{ij}^2=A\,h/r_{ij}^3$ with $A=p_xp_y$.

### Tile representatives

`angular_tiles` splits rows into $a_y$ and columns into $a_x$ contiguous, ordered, non-empty groups (`np.array_split`), $R_0<R_1<\dots$ and $C_0<C_1<\dots$. Tile $k=(a,b)$, flattened row-major as $k=a\,a_x+b$, collects pixels $R_a\times C_b$. Its representative is the normalised solid-angle-weighted mean direction

$$
\mathbf d_k=\frac{\sum_{p\in k}\Delta\Omega_p\,\hat{\mathbf u}_p}{\bigl\lVert\sum_{p\in k}\Delta\Omega_p\,\hat{\mathbf u}_p\bigr\rVert}
\;\propto\;\sum_{p\in k}\alpha_p\,\mathbf c_p,
\qquad
\alpha_p=\frac{\Delta\Omega_p}{r_p}=\frac{A\,h}{r_p^{4}}>0 .
$$

### Knots by ray–plane intersection

The ray $s\,\mathbf d_k$, $s>0$, meets the plane $(\mathbf r-\mathbf c)\cdot\hat{\mathbf n}=0$ at

$$
s_k=\frac{\mathbf c\cdot\hat{\mathbf n}}{\mathbf d_k\cdot\hat{\mathbf n}} .
$$

Substituting $\mathbf d_k\propto\sum_p\alpha_p\mathbf c_p$ and $\mathbf c_p\cdot\hat{\mathbf n}=h$ gives $\mathbf d_k\cdot\hat{\mathbf n}\propto h\sum_p\alpha_p$. The normalisation constant cancels, so

$$
\boxed{\;s_k\,\mathbf d_k=\sum_{p\in k}\beta_p\,\mathbf c_p,
\qquad
\beta_p=\frac{\alpha_p}{\sum_{q\in k}\alpha_q}=\frac{r_p^{-4}}{\sum_{q\in k}r_q^{-4}}\in(0,1],
\qquad
\sum_{p\in k}\beta_p=1 .\;}
$$

The projected representative is a convex combination, with strictly positive weights, of the tile's own pixel centres. Its local coordinates are

$$
X_{ab}=(s_k\mathbf d_k-\mathbf c)\cdot\hat{\mathbf x}=\sum_{p\in k}\beta_p\,x_{j(p)},
\qquad
Y_{ab}=(s_k\mathbf d_k-\mathbf c)\cdot\hat{\mathbf y}=\sum_{p\in k}\beta_p\,y_{i(p)} .
$$

The knots are the row and column means

$$
\bar X_b=\frac{1}{a_y}\sum_{a=0}^{a_y-1}X_{ab},
\qquad
\bar Y_a=\frac{1}{a_x}\sum_{b=0}^{a_x-1}Y_{ab} .
$$

**Strict monotonicity.** Because the weights are a convex combination, $X_{ab}\in[\min_{j\in C_b}x_j,\max_{j\in C_b}x_j]$ for every $a$, and so is $\bar X_b$. The groups are contiguous and ordered, and $p_x>0$, so $\max_{j\in C_b}x_j<\min_{j\in C_{b+1}}x_j$. Hence

$$
\bar X_b\le x_{\max C_b}<x_{\min C_{b+1}}\le\bar X_{b+1},
$$

and the $y$ knots behave the same way. The knots are strictly increasing for any contiguous split, with no condition on pose, roll, or obliquity.

**One-pixel group.** If $C_b=\{j\}$, every pixel of every tile in column group $b$ has $x=x_j$, so $X_{ab}=x_j$ for all $a$ and $\bar X_b=x_j$ exactly. Rows behave the same way. In floating point the projection carries round-off of order $\epsilon\,\lVert\mathbf c\rVert$, which motivates a small knot snap: a pixel within a relative $10^{-9}$ of a knot interval endpoint is treated as lying on that knot.

### Clamped separable weights

For a pixel at local $(x,y)=(x_j,y_i)$ and knots $\bar X_0<\dots<\bar X_{a_x-1}$, define the lower knot index $b$ and fraction $t_x$ by

$$
b=\max\{0\le b\le a_x-2:\bar X_b\le x\}\ (\text{or }0\text{ if none}),
\qquad
t_x=\operatorname{clip}\!\left(\frac{x-\bar X_b}{\bar X_{b+1}-\bar X_b},0,1\right).
$$

For $a_x=1$ set $b=b+1=0$ and $t_x=0$. Define $(a,t_y)$ from the $\bar Y$ knots the same way. The four tile weights are

$$
w_{(a,b)}=(1-t_y)(1-t_x),\quad
w_{(a,b+1)}=(1-t_y)t_x,\quad
w_{(a+1,b)}=t_y(1-t_x),\quad
w_{(a+1,b+1)}=t_yt_x ,
$$

with flat tile index $k=a\,a_x+b$. This is Numerical Recipes Eq. 3.6.5 with $(t,u)\to(t_y,t_x)$, evaluated on the rectilinear knot grid $\{\bar Y_a\}\times\{\bar X_b\}$.

- **Convexity.** $t_x,t_y\in[0,1]$, so each weight is $\ge0$, and $\sum w=[(1-t_y)+t_y][(1-t_x)+t_x]=1$.
- **Clamping.** If $x<\bar X_0$, then $t_x=0$ and the pixel takes knot $0$. If $x>\bar X_{a_x-1}$, then $b=a_x-2$, $t_x=1$, and the pixel takes knot $a_x-1$. There is no extrapolation.
- **Continuity.** At $x=\bar X_{b+1}$ the interval $[b,b+1]$ gives $t_x=1$ and the interval $[b+1,b+2]$ gives $t_x=0$. Both put all weight on knot $b+1$, so the blend is continuous whichever interval the search picks.

### Pixel flux

The blend acts only on the intrinsic (per-steradian) tile density. The pixel's own solid angle and filter transmission multiply afterwards:

$$
F_p(E)=\Bigl[\sum_{k}w_{pk}\,I_k(E)\Bigr]\Delta\Omega_p\,T_p(E).
$$

Blending $I_k\Delta\Omega_k$ or $I_kT_k$ instead would be a convention error.

### Limiting cases

1. **`angular_shape` equals the pixel shape.** Every group is a single pixel, so $\bar X_j=x_j$ and $\bar Y_i=y_i$ exactly in exact arithmetic. Pixel $(i,j)$ sits on its own knot, $t\in\{0,1\}$ selects its own index, and $w$ is one-hot on tile $(i,j)$. This is the `nearest_tile` assignment. In floating point $1\cdot I+0\cdot I'+\dots=I$ is bit-exact for finite $I'$, given the knot snap. A non-finite neighbour spectrum would make $0\cdot\infty$ NaN.
2. **$1\times1$.** Both axes have one knot, so $t_x=t_y=0$ and all weight goes to tile $0$, again `nearest_tile`.
3. **One-tile axis.** That axis has $t=0$ everywhere. The blend is constant along it and reduces to 1-D linear interpolation along the other axis.
4. **Linear-field reproduction.** If $I_{(a,b)}=\alpha+\beta\bar X_b+\gamma\bar Y_a+\delta\bar X_b\bar Y_a$ (bilinear *in knot coordinates*), the blend reproduces $\alpha+\beta x+\gamma y+\delta xy$ exactly for $(x,y)$ in the knot hull $[\bar X_0,\bar X_{a_x-1}]\times[\bar Y_0,\bar Y_{a_y-1}]$, and the clamped value outside it. The representative points $(X_{ab},Y_{ab})$ need not lie on the knot grid when tiles have several pixels. A field linear in the representative coordinates is therefore reproduced only up to the offset $X_{ab}-\bar X_b$, which is the caveat recorded in the ledger notes.
5. **Pixel on a tile centre.** If a pixel's $x$ equals some $\bar X_b$ and its $y$ equals some $\bar Y_a$, $w$ is one-hot on tile $(a,b)$.

## Comparison with the implementation

The bodies were read only after the derivation above was written.

### Symbolic, term by term

| Derived quantity | Code | Agreement |
| --- | --- | --- |
| $s_k\mathbf d_k-\mathbf c$, with $s_k=(\mathbf c\cdot\hat{\mathbf n})/(\mathbf d_k\cdot\hat{\mathbf n})$ | `hits = directions * ((center @ normal) / (directions @ normal))[:, None] - center` | exact; the normalisation of $\mathbf d_k$ cancels, as derived |
| $X_{ab}$, $Y_{ab}$ on the $(a_y,a_x)$ tile grid, $k=a\,a_x+b$ | `(hits @ x_axis).reshape(ay, ax)`, `(hits @ y_axis).reshape(ay, ax)` | row-major reshape matches `angular_tiles`' `tile += 1` loop (rows outer, columns inner) |
| $\bar X_b$ = mean over rows $a$ | `np.mean(local_x, axis=0)` | exact |
| $\bar Y_a$ = mean over columns $b$ | `np.mean(local_y, axis=1)` | exact |
| pixel $x_j$ with $n_x$, $p_x$; pixel $y_i$ with $n_y$, $p_y$ | `(np.arange(nx) - (nx - 1) / 2.0) * pitch_x` paired with the $x$ knots; `ny`, `pitch_y` paired with the $y$ knots; `pitch_y, pitch_x = pitch_mm` | same convention as `planar_detector_rays`; no row/column swap |
| lower index $b=\max\{b\le a_x-2:\bar X_b\le x\}$, or $0$ | `np.clip(np.searchsorted(knots, coordinate, side="right") - 1, 0, knots.size - 2)` | exact; `side="right"` gives $\bar X_b\le x$ |
| $t_x=\operatorname{clip}(\cdot,0,1)$ | `np.clip((coordinate - knots[lower]) / (knots[upper] - knots[lower]), 0.0, 1.0)` | exact |
| one-knot axis, $t=0$ | `knots.size == 1` returns `lower = upper = 0`, `fraction = 0` | exact; the duplicate upper slot carries weight $0$ |
| four weights and tile indices | `(lr,lc),(lr,uc),(ur,lc),(ur,uc)` with `(1-wy)*(1-wx), (1-wy)*wx, wy*(1-wx), wy*wx`, tile `r * ax + c` | exact; NR Eq. 3.6.5 with $(t,u)\to(t_y,t_x)$ |
| $\sum_k w_{pk}I_k(E)$ | `np.einsum("pk,pke->pe", weight, factor.intrinsic_by_tile[tile])` | `tile` is $(P,4)$, so the fancy index is $(P,4,E)$ and the einsum contracts the four slots per pixel at fixed energy |
| $\times\,\Delta\Omega_p\,T_p(E)$ | `intrinsic * self.ray_map.solid_angle_sr[rows, columns, None] * transmission` | the pixel's own solid angle and transmission multiply after the blend, as required |
| $(a_y,a_x)$ recovered in `_blend` | `n_column = tile_index[0, -1] + 1`, `n_tile // n_column` | the top-right pixel of row 0 carries tile $a_x-1$ under row-major numbering, so this is correct |

**Knot snap.** `_axis_weights` sets `fraction < 1e-9` to $0$ and `fraction > 1 - 1e-9` to $1$. The tolerance is a fraction of the knot interval, so it is relative to the pitch scale. This is the step that makes limiting case 1 one-hot despite projection round-off. The snap changes any weight by at most $10^{-9}$ of the local spectral difference between adjacent tiles, which is far below the Monte Carlo noise. It is a deliberate, bounded approximation, not a discrepancy. Its margin is set by the projection round-off, about $\epsilon\,\lVert\mathbf c\rVert/p$ in units of the interval. That is about $10^{-14}$ at 100 mm with sub-mm pitch, so the margin is five orders of magnitude. The snap stops absorbing the round-off only near $\lVert\mathbf c\rVert/p\sim10^{8}$. In a 100 m, 1 µm probe, `angular_shape` equal to the pixel shape then left $7\times10^{-9}$ weight off-diagonal. That geometry is unphysical for this instrument, and the residual is still negligible, but it bounds the "bit-identical" claim.

### Numeric, independent reference

The reference script (session scratch, plain numpy plus `PlanarPose`/`PixelGrid` for the inputs only) builds the pixel centres from the pose. It computes the representative directions from their definition, and computes the knots from the closed-form convex weights $\beta_p\propto r_p^{-4}$ rather than by plane projection. It then evaluates the weights with an explicit loop interval search and no snap. The table compares it with `angular_tiles` + `angular_tile_weights`.

| Pose, pixels, pitch (mm), `angular_shape` | max $\lvert\Delta\mathbf d_k\rvert$ | max $\lvert\Delta w\rvert$ | $\min w$; max $\lvert\sum w-1\rvert$ | equals `nearest_tile` |
| --- | --- | --- | --- | --- |
| 100 mm, $60^\circ$/$20^\circ$, roll $15^\circ$; $7\times9$; $(0.5,0.75)$; $(3,4)$ | $2.2\times10^{-16}$ | $3.7\times10^{-15}$ | $0$; $1.1\times10^{-16}$ | no (expected) |
| same; $(7,9)$ | $2.2\times10^{-16}$ | $0$ | $0$; $0$ | yes, one-hot exactly |
| same; $(1,1)$ | $1.1\times10^{-16}$ | $0$ | $0$; $0$ | yes |
| same; $(1,4)$ | $1.1\times10^{-16}$ | $6.9\times10^{-15}$ | $0$; $0$ | no; constant along $y$ |
| 50 mm, $35^\circ$/$-70^\circ$, roll $-40^\circ$, offset $(3,-2)$; $16\times11$; $(0.055,0.11)$; $(5,3)$ | $2.2\times10^{-16}$ | $2.2\times10^{-14}$ | $0$; $2.2\times10^{-16}$ | no (expected) |
| explicit tilted pose, $\hat{\mathbf n}\propto(0.2,0.1,1)$; $12\times12$; $(1,1)$; $(5,5)$ | $1.1\times10^{-16}$ | $8.9\times10^{-16}$ | $0$; $1.1\times10^{-16}$ | no (expected) |

In all cases the knots are strictly increasing. The ray–plane projection of the code agrees with the closed-form $r^{-4}$ convex combination to $\le2\times10^{-14}$ in weight. For the one-pixel-tile case $\max\lvert\bar X_j-x_j\rvert/p_x=1.0\times10^{-14}$, which confirms that the knot is the pixel centre up to round-off.

**Linear-field reproduction.** A field bilinear in knot coordinates, $I=0.3+1.7\bar X-0.9\bar Y+0.25\bar X\bar Y$, plus a second energy bin $2I$, was pushed through the literal `_materialize` einsum. It matched the clamped exact field $I(\operatorname{clip}(x),\operatorname{clip}(y))$ to $\le2\times10^{-14}$ in every multi-tile case.

**Bit-identity.** With `angular_shape` equal to the pixel shape and random log-normal spectra, the literal einsum `np.einsum("pk,pke->pe", weight, spec[tile])` equals `spec[tile_index]` under `np.array_equal`. This holds because $1\cdot I+0\cdot I'=I$ exactly for finite $I'$.

**Anchors.** `tests/instrument/test_geometry.py`, `tests/results/test_spatial_model.py`, and `tests/observations/test_store.py` all pass (77 tests).

## Adjudication

- **Units.** The weights are dimensionless and the knots are in mm, the same as the pixel coordinates. $F_p$ keeps the units of `positioned-filter-attenuation`, because the blend acts only on the per-steradian intrinsic density.
- **Limits.** All five limiting cases hold, symbolically and numerically.
- **Conventions.** Row ↔ local $y$ and column ↔ local $x$ are consistent across `planar_detector_rays`, `angular_tiles`, and `angular_tile_weights`. The flat tile index is row-major throughout. The `_blend` shape inference is correct.

The independent derivation matches the implementation term for term. The $10^{-9}$ snap is a bounded approximation whose round-off margin holds for any realistic distance-to-pitch ratio. The scope caveats in the ledger notes (fixed-energy blending, clamping beyond the knot hull, exactness only at knots) agree with the derivation and are not defects.

Recommended status: **`rederived`**. Final `signed-off` is a human decision.
