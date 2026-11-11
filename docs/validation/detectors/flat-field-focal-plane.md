# `flat-field-focal-plane` — independent verification

**Code**: `src/pyrite/detectors/grating.py::{FlatFieldMount.position_mm, FlatFieldMount.exit_arm_mm, Grating.from_catalog, Grating.mount_residuals}`, with helpers `_catalog_mount` and `_focal_plane` **Ledger claim**: a ray at $\beta$ meets a flat focal plane (unit normal at $\beta_H$, perpendicular distance $r_H'$) after path $r'(\beta)=r_H'/\cos(\beta-\beta_H)$ at offset $s(\beta)=r_H'\tan(\beta-\beta_H)$ from the foot of the perpendicular, with $s\to\mp r_H'\cot\beta$ for $\beta_H\to\pm90°$; a `"difference"` record maps to the module convention by $\beta\to-\beta$ at fixed $m$ **Source**: plane–line intersection in the dispersion plane; flat-field mount geometry of Kita, Harada, Nakano & Kuroda, *Appl. Opt.* 22, 512 (1983).

## 1. Scope

- Module convention: angles from the grating normal, $\sin\beta=m\lambda/d-\sin\alpha$, so $m=0$ gives $\beta=-\alpha$.
- `FlatFieldMount(entrance_arm_mm, focal_distance_mm, focal_normal_rad, focal_plane_length_mm=None, band_nm=None, band_beta_rad=None)`: `focal_distance_mm` is $r_H'$ [mm], `focal_normal_rad` is $\beta_H$ [rad], restricted to $(-\pi/2,\pi/2)$.
- `position_mm(beta_rad)` returns $s$ [mm]; `exit_arm_mm(beta_rad)` returns $r'$ [mm].
- `Grating.from_catalog(name)` lowers a `[gratings.NAME]` record. The schema gives `"sum"` as $m\lambda=d(\sin\alpha+\sin\beta)$ and `"difference"` as $m\lambda=d(\sin\alpha-\sin\beta)$, with `band_nm` in nm and `beta_deg`, `beta_H_deg` in degrees.
- `mount_residuals()` returns the computed-minus-tabulated band-end $\beta$ in degrees, and $\lvert s(\beta_{\rm long})-s(\beta_{\rm short})\rvert/L_{\rm tab}-1$.
- `_focal_plane` returns $(s, ds/d\beta)$ for the flux Jacobian.
- Assumptions: point source at the grating centre, rays in the dispersion plane, no defocus, coma or VLS aberration.

## 2. Independent derivation (before reading the implementation bodies)

### Plane–line intersection

Put the grating centre at the origin of the dispersion plane, with the grating normal along $\hat{\mathbf y}$ and angles measured from it toward $+\hat{\mathbf x}$. A ray at angle $\beta$ has direction

$$
\hat{\mathbf u}(\beta)=(\sin\beta,\ \cos\beta).
$$

The focal plane is $\{\mathbf p:\mathbf p\cdot\hat{\mathbf n}_H=r_H'\}$ with $\hat{\mathbf n}_H=\hat{\mathbf u}(\beta_H)$. The ray $\mathbf p=t\,\hat{\mathbf u}(\beta)$ meets it where $t\,\hat{\mathbf u}(\beta)\cdot\hat{\mathbf u}(\beta_H)=t\cos(\beta-\beta_H)=r_H'$, so the exit arm is

$$
r'(\beta)=\frac{r_H'}{\cos(\beta-\beta_H)},\qquad \lvert\beta-\beta_H\rvert<\tfrac{\pi}{2}.
$$

Choose the in-plane unit tangent as the direction of increasing $\beta$, $\hat{\mathbf e}=d\hat{\mathbf u}/d\beta\rvert_{\beta_H}=(\cos\beta_H,\,-\sin\beta_H)$. The foot of the perpendicular is $r_H'\hat{\mathbf n}_H$, and $\hat{\mathbf n}_H\cdot\hat{\mathbf e}=0$, so the offset from the foot is

$$
s(\beta)=r'(\beta)\,\hat{\mathbf u}(\beta)\cdot\hat{\mathbf e}
=\frac{r_H'\,(\sin\beta\cos\beta_H-\cos\beta\sin\beta_H)}{\cos(\beta-\beta_H)}
=\boxed{r_H'\tan(\beta-\beta_H)}.
$$

Handedness: $s$ increases with $\beta$. At $\beta=\beta_H$, $s=0$ (the foot). The map is the explicit-distance flat-detector map $L\tan(\beta-\beta_{\rm ref})$ with $L=r_H'$ and $\beta_{\rm ref}=\beta_H$.

### Slope

$$
\frac{ds}{d\beta}=\frac{r_H'}{\cos^2(\beta-\beta_H)}=\frac{r'(\beta)^2}{r_H'}>0 .
$$

Chained with the grating equation, $d\beta/d\lambda=m/(d\cos\beta)$.

### Plane perpendicular to the grating surface

$\tan$ has period $\pi$, so $\tan(\beta\mp\tfrac{\pi}{2})=-\cot\beta$ for **both** signs:

$$
\lim_{\beta_H\to+90°}s=\lim_{\beta_H\to-90°}s=-r_H'\cot\beta .
$$

At $\beta_H=+90°$ the plane is $x=r_H'$ and $\hat{\mathbf e}=(0,-1)$. The ray reaches height $y=r_H'\cot\beta$ above the grating plane, and $s=-y$. At $\beta_H=-90°$ the plane is $x=-r_H'$, $\hat{\mathbf e}=(0,+1)$ and $y=-r_H'\cot\beta$ (positive for $\beta<0$), so $s=+y=-r_H'\cot\beta$. In both cases the foot lies in the grating surface plane, so $s$ is the signed distance from that plane. The unsigned height is $\pm r_H'\cot\beta$. **Neither reading gives the claimed $s\to\mp r_H'\cot\beta$.** The $\beta_H\to-90°$ branch of the claim has the wrong sign.

### Order-convention mapping

A `"difference"` record has $\sin\beta_D=\sin\alpha-m\lambda/d$. The module solves $\sin\beta_S=m\lambda/d-\sin\alpha=-\sin\beta_D$, so $\beta_S=-\beta_D$ with the same $m$ and the same $\alpha$. The `"difference"` frame is the module frame reflected about the grating normal ($x\to-x$). Every angle measured from the normal therefore changes sign, including $\beta_H$ and the tabulated band-end $\beta$, while $r$, $r_H'$, $r'$ and lengths are unchanged. Under the reflection $s_S(\beta_S)=r_H'\tan(-\beta_D+\beta_{H,D})=-s_D(\beta_D)$, so $s$ changes sign and the focal-plane length $\lvert\Delta s\rvert$ is unchanged.

### Residuals

$E\,[{\rm eV}]=hc/\lambda=12398.4198\,{\rm eV\,\mathring{A}}/(10\,\lambda[{\rm nm}])$. Then $\beta=\arcsin(m\lambda/d-\sin\alpha)$ at both band ends, the angle residual is $\beta_{\rm calc}-\beta_{\rm tab}$, and the length residual is $\lvert s(\beta_2)-s(\beta_1)\rvert/L_{\rm tab}-1$.

## 3. Diff against the implementation

```text
FlatFieldMount.position_mm:  detector_position_mm(beta_rad, self.focal_normal_rad, self.focal_distance_mm)
detector_position_mm:        distance_mm * np.tan(np.asarray(beta_rad, float) - beta_ref_rad)
FlatFieldMount.exit_arm_mm:  self.focal_distance_mm / np.cos(np.asarray(beta_rad, float) - self.focal_normal_rad)
_focal_plane (mount branch): slope = mount.focal_distance_mm / np.cos(beta - mount.focal_normal_rad) ** 2
_catalog_mount:              focal_normal_rad=sign * deg2rad(beta_H_deg); band_beta_rad=tuple(sign * deg2rad(beta))
Grating.from_catalog:        sign = 1.0 if order_convention == "sum" else -1.0; alpha, order unchanged
Grating.mount_residuals:     energy = HC_EV_ANG / (10.0 * band_nm); beta = diffraction_angle_rad(energy)
                             beta_error = rad2deg(beta - band_beta_rad)
                             length_error = abs(diff(position_mm(beta))[0]) / focal_plane_length_mm - 1
```

| Item | Independent | Code | Match |
|---|---|---|---|
| exit arm | $r_H'/\cos(\beta-\beta_H)$ | same | yes |
| offset | $r_H'\tan(\beta-\beta_H)$, increasing with $\beta$ | same | yes |
| slope | $r_H'/\cos^2(\beta-\beta_H)$ | same | yes |
| `"difference"` map | $\beta,\beta_H,\beta_{\rm band}\to-\beta$; $m,\alpha,r_H'$ kept | same | yes |
| nm $\to$ eV | $hc/(10\lambda_{\rm nm})$ with $hc$ in eV Å | same (`HC_EV_ANG = 12398.4198`) | yes |
| length residual | $\lvert\Delta s\rvert/L_{\rm tab}-1$ | same | yes |
| $\beta_H\to\pm90°$ limit | $-r_H'\cot\beta$ for both | code gives $-r_H'\cot\beta$ for both | yes |
| $\beta_H\to\pm90°$ claim text | $-r_H'\cot\beta$ for both | ledger line 37 and docstring `grating.py:171` say $\mp r_H'\cot\beta$ | **no** |

The code is correct. The mismatch is in the claim text: the docstring and the ledger row say $\mp$.

## 4. Numeric spot-checks

The script solves the plane–line intersection as a $2\times2$ linear system $[\hat{\mathbf u}(\beta),\,-\hat{\mathbf e}]\,(t,s)^{\mathsf T}=r_H'\hat{\mathbf n}_H$. It does not use the $\tan$ formula. Values are synthetic.

- 2000 random $(\beta_H,r_H',\beta)$ with $\lvert\beta_H\rvert<1.55$ and $\lvert\beta-\beta_H\rvert<1.4$: maximum relative difference from `position_mm`/`exit_arm_mm` is $6\times10^{-15}$.
- $\beta_H=+89.9999999°$, $\beta=(70°,84°)$: code $(-101.912,-29.429)$ mm, $-r_H'\cot\beta=(-101.912,-29.429)$ mm.
- $\beta_H=-89.9999999°$, $\beta=(-70°,-84°)$: code $(+101.912,+29.429)$ mm, $-r_H'\cot\beta=(+101.912,+29.429)$ mm, while the claimed $+r_H'\cot\beta=(-101.912,-29.429)$ mm. This confirms that the claim's $-90°$ branch has the wrong sign.
- Slope vs. a central difference of `position_mm` on $\beta\in[-83°,-76°]$: maximum relative difference $9\times10^{-11}$.
- Synthetic `"difference"` record (1000 /mm, $m=+1$, $\alpha=86°$, band 6–24 nm, $\beta_H=+86.5°$, $r_H'=280$ mm): the independently solved $\beta_D=(82.5525°,76.7963°)$ maps to $\beta_S=(-82.5525°,-76.7963°)$, and $\beta_H$ maps to $-86.5°$. `mount_residuals` returns zero angle error. The span from the code is $28.55832$ mm, and $\lvert\Delta s_D\rvert$ computed independently in the `"difference"` frame is the same, $28.55832$ mm.
- $E(6\,{\rm nm})=206.64033$ eV, matching CODATA $hc=1239.841984$ eV nm to $4\times10^{-9}$.
- `pyrite-dev test tests/detectors/test_grating.py`: 39 passed, 1 skipped (the private-catalog test).

## 5. Filters

- **Units**: $s$, $r'$ and $ds/d\beta$ are in mm (per rad); the angle arguments are in rad; the residuals are in degrees and are dimensionless. **Pass.**
- **Limits**: $\beta=\beta_H\Rightarrow s=0$, $r'=r_H'$. The code gives $\beta_H\to\pm90°\Rightarrow-r_H'\cot\beta$, which is correct. **The claim's $\beta_H\to-90°$ statement fails.**
- **Signs/conventions**: handedness ($s$ increasing with $\beta$), the $\beta\to-\beta$ map at fixed $m$, and the sign flip of $\beta_H$ and the band-end $\beta$ all pass.

## 6. Omissions and findings (no code change made)

1. **Claim/docstring sign** (`docs/validation/ledger-grazing-grating-spectrometer.md:37`, `src/pyrite/detectors/grating.py:171`): "$s\to\mp r_H'\cot\beta$ for $\beta_H\to\pm90°$" should read "$s\to-r_H'\cot\beta$ for $\beta_H\to\pm90°$". The anchor test (`tests/detectors/test_grating.py:458`) covers only $+90°$.
2. **No domain guard**: when $\lvert\beta-\beta_H\rvert\ge90°$, `exit_arm_mm` is negative and `position_mm` returns a finite but meaningless $\tan$ value (the ray never reaches the plane). Real mounts do not get there, but these rays are not returned as NaN.
3. **Unused tabulated exit arms**: the schema's `r_prime_mm` is validated but not lowered. `mount_residuals` does not compare it with $r'(\beta)$, although this is a free consistency check of $r_H'$ and $\beta_H$. For the synthetic test mount, $r'=(280.67,284.06)$ mm agrees with the table.

## 7. Verdict

**discrepancy**, confined to the claim text. The implementation matches the independent derivation term for term: intersection, exit arm, handedness, slope, `"difference"`→`"sum"` map including $\beta_H$ and the band-end $\beta$, nm→eV conversion and length residual. The ledger claim and the `position_mm` docstring wrongly state $s\to+r_H'\cot\beta$ for $\beta_H\to-90°$. After both texts are corrected to $s\to-r_H'\cot\beta$ for either sign, with no code change needed, the claim qualifies as `rederived`.
