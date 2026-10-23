# Continuous-source aperture scoring

Validation: `photon-continuous-source-scoring`.

## Independence and scope

Fresh-context derivation on 2026-10-22, written before implementation-body
inspection. Inputs were the ledger claim and owner signatures/docstrings,
[ADR-0020](../../adr/0020-photon-transport-source-and-scoring.md), and the
linear importance-sampling identity in
[continuous source](photon-continuous-source.md).
This verifies weighted tally algebra conditional on correct ordinary aperture
hits and inherited weights. It does not certify interaction sampling, queue
physics, charged feedback, or full physical showers.

## Independent derivation

A source weight is an expected photon count. For an absorbing physical aperture,
an ordinary detected hit contributes that weight once. Let $H$ contain those
hits; an energy bin $b$ has width $\Delta k_b$ in eV. For the original electron
population $N_e$,

$$
\boxed{S_b=\frac{1}{N_e\Delta k_b}
\sum_{h\in H}w_h\mathbf{1}(k_h\in b).}
$$

The result has units photons/(primary eV), integrated over the aperture. There
is no division by solid angle: the aperture indicator already defines its
angular acceptance. Neither proposal count nor hit count is a primary count.
Source weights already include the inverse proposal count, so an additional
division by that count would bias the result. Likewise, a second ballistic
attenuation would duplicate transport survival.

Define $D_h=1$ only when both photon generation and charged-coupling round are
zero. A newly re-emitted photon can have generation zero in a new queue but
positive round; it belongs to the redistributed category. If no coupling round
field exists, its value is zero. Set

$$
S_{b,\mathrm{direct}}=\frac{\sum_{h\in H}w_hD_h\mathbf{1}(k_h\in b)}
{N_e\Delta k_b},\qquad
S_{b,\mathrm{redistributed}}=\frac{\sum_{h\in H}w_h(1-D_h)
\mathbf{1}(k_h\in b)}{N_e\Delta k_b}.
$$

The categories are disjoint and exhaustive, so their sum is $S_b$.
Use left-closed, right-open bins except that the last bin includes its final
upper edge. A hit on any interior edge belongs to the bin to its right; a hit
on the final edge remains in the last bin. The independent outside-band count
must include both direct and redistributed hits:

$$
U=\frac{1}{N_e}\sum_{h\in H}w_h
\mathbf{1}(k_h<k_0\ \mathrm{or}\ k_h>k_B).
$$

Its units are photons/primary, with no energy-width factor. In particular,
direct hits outside the band must contribute even if their in-band category
histogram is empty.

## Cheap filters

All tallies are nonnegative. Empty hits give zero. With fixed hit weights,
doubling $N_e$ halves every tally. A collision-free vacuum produces direct
hits only; positive-round generation-zero hits are redistributed. The closure
identity is $\sum_b S_b\Delta k_b+U=\sum_{h\in H}w_h/N_e$.

## Source-to-code comparison

After writing the derivation above, inspection found that
`score_bremslib_continuum` passes unattenuated source weights to exactly one
of the photon-only or coupled history entry points, then calls `_score_exits`
once with the original primary count. It adds no hard-event histogram,
extra energy debit, extra ballistic attenuation, proposal denominator, or
solid-angle denominator. Physical-aperture and explicit charged-threshold
requirements are checked at the handoff.

`_score_exits` uses `detected_spectrum` for the total and weighted histograms
over the same edges for direct and redistributed categories. The common
denominator is the original population times each bin width. Direct requires
both generation zero and, if present, round zero. The underlying histogram
includes the final upper edge. Outside-band hits use strict comparisons with
the outer edges and their own direct weighted sum, retaining direct as well
as redistributed hits. Computing this sum separately avoids cancellation from
subtracting a large in-band count from a nearly equal all-hit count.

An independent synthetic probe used edges 1, 2, 4 eV, energies 0.5, 1, 2, 3,
4, 4.5 eV, weights 2, 3, 5, 7, 11, 13, and two original primaries.
The 2 eV hit had generation one; the 3 eV hit had generation zero but round
one. The returned total was $(1.5,5.75)$, direct $(1.5,2.75)$, redistributed
$(0,3)$ photons/(primary eV), with 20.5 photons/primary detected and 7.5
outside the band. All matched independent arithmetic exactly. A second probe
combined an in-band weight of $10^{25}$ with a direct out-of-band weight of
one; the outside tally retained exactly 0.5 photons/primary.

## Verdict

- **Filters:** units pass; limits pass; signs/conventions pass.
- **Re-derivation:** matches; no divergent primary normalization, bin-width
  factor, final-edge convention, category partition, or outside-band term found.
- **Verdict:** `rederived`.
- **Suggested ledger change:** set `photon-continuous-source-scoring` to
  `rederived`, replace the pending note with this bounded fresh-context
  comparison and link this page.
- **Human sign-off:** pending. The result remains conditional on the inherited
  history transport and does not expand its physics validation scope.

## Bin-local numerical repair (2026-10-23)

The original weighted histogram reduction could cancel a small positive bin
against a much larger weight in another bin or below the displayed band.
The total, direct and redistributed spectra now share
`photon_histories._weighted_photon_counts`: assign each in-band energy using
the internal edges and accumulate weights directly in its bin. This preserves
the final inclusive edge and the single primary/bin-width denominator.
The independently derived counterexample and fresh repair verification are
recorded in [photon history queue](../beam-transport/photon-history-queue.md).
