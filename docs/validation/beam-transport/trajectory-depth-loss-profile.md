# Saved-segment continuous-loss depth attribution

## Claim and source

Validation id: `trajectory-depth-loss-profile`.
Code: `montecarlo/trajectory_summary.py::depth_loss_profile`.
Source: the explicit uniform continuous-loss convention in
[transport outputs](../../physics/beam-transport/transport-outputs.md#continuous-loss-depth-attribution).
This defines an analysis quantity; it introduces no stopping-power model or
dose normalization.

## Independent derivation

A fresh read-only verifier derived this expression before inspecting the
implementation. Parameterize the stored straight segment as

$$
\mathbf r(t)=\mathbf r_{\rm mid}+(t-1/2)L\mathbf v,\qquad 0\leq t\leq1.
$$

Uniform continuous loss means $dQ=\Delta E\,dt$ for
$\Delta E=E_{\rm start}-E_{\rm end}$. Let $a,b$ be the sorted endpoint depths.
For $a<b$, changing variable from $t$ to depth gives

$$
Q_j=\Delta E\frac{\max(0,\min(b,e_{j+1})-\max(a,e_j))}{b-a}.
$$

The depth ratio is dimensionless, leaving keV per bin. Disjoint bins covering
$[a,b]$ partition the segment, so their sum is $\Delta E$. Reversing $v_z$
exchanges the endpoints and preserves the result. Zero loss gives zero.
Partial coverage retains the corresponding fraction. Grouping by stored
layer preserves the segment's material identity.

For $a=b$, the loss is a point measure at the midpoint: use half-open bins,
with the final right edge inclusive. The transverse limit agrees away from
bin edges. Exactly on an internal edge, the point-bin convention differs
from the symmetric nonzero-span limit.

## Comparison and anchors

The independent verifier compared the implementation and returned `rederived`:
overlap, sign, units, layer grouping and endpoint conventions match.
`tests/montecarlo/test_trajectory_summary.py::test_depth_loss_analytic_and_transverse`
pins a 12 keV segment over depths $[-1,3]$ Å with edges $[-1,0,2,3]$ Å to
$[3,6,3]$ keV in either direction. Partial coverage $[0,2]$ receives 6 keV.
Transverse internal/final-edge cases pin the point-bin rule.
The streaming ledger test compares the profile sum to canonical
`secondary_energy_balance(...)["continuous_keV"]`.

Hard-event energy transfers and terminal residuals are outside this observable.
Only the full secondary ledger closes incident energy; no claim assigns those
extra terms to the continuous-loss depth bins. Status: `anchored`, never human
sign-off.
