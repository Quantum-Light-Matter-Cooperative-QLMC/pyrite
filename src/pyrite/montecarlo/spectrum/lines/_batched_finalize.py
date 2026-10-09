"""Coherent finalisation for the batched CXR line route.

Moved verbatim out of :mod:`._batched` (re-exported there) to keep module
sizes within the line budget.
"""

import numpy as np

from ...._backend import REAL, _to_cpu, xp
from ....materials.crystal import HBARC_EV_ANG
from . import _policy
from ._formation import formation_profile
from ._per_hkl import (
    _coherent_electron_grouped_row,
    _coherent_jit_grouped_row,
    _decoherence_blend,
    _flat_energy_keep,
    _flat_view,
    _FormationLines,
    _row_decoherence_factor,
)


def _batched_coherent_finalize(
    st,
    bt,
    coh_blocks,
    coh_counts,
    coherent_fields,
    stream_segment_block,
    stream_field_mag2,
    stream_F_rows=None,
    stream_keep=None,
):
    """Reduce the coherent route's retained fields into the spectrum.

    Either finalises the streamed per-g field planes, or reduces the per-row
    line records the block loop retained. Rows are reduced independently, so
    reflections and mosaic orientations stay incoherent; where the
    inter-electron decoherence blend is active it is applied PER ROW, before
    the rows are summed.
    """
    # NVTX sub-ranges are a no-op off the profiled GPU path. Lazy import:
    # runner imports this module, so a top-level import would be circular.
    from ...runner import _nsys_pop, _nsys_push

    req = st.request
    chunk = req.chunk
    coherent = req.coherent
    Ne = st.Ne
    E_grid = st.E_grid
    spec = st.spec
    seg_elec_id = st.seg_elec_id
    cdtype = st.cdtype
    decoherence_active = st.decoherence_active or st.request.physical_electrons is not None
    N_g = bt.N_g
    seg_block = bt.seg_block
    G = bt.G
    WM = bt.WM
    wm_rows = bt.wm_rows
    _use_jit_coherent_stream = stream_segment_block is not None
    if _use_jit_coherent_stream:
        from ..coherent_stream_jit_kernel import (
            DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
            finalize_coherent_fields,
        )

    _stream_segment_block = stream_segment_block
    _stream_field_mag2 = stream_field_mag2

    # Existing per-row coherent reducer remains the compatibility fallback for
    # non-streaming coherent execution (e.g. disabling the experimental stage).
    _use_jit_coherent_reduction = (
        coherent
        and not _use_jit_coherent_stream
        and _policy._USE_JIT_COHERENT_REDUCTION
        and getattr(xp, "__name__", "") == "cupy"
        and np.dtype(REAL) == np.dtype(np.float32)
    )
    if _use_jit_coherent_reduction:
        from ..coherent_jit_kernel import (
            DEFAULT_COHERENT_KERNEL_CONFIG,
            run_coherent_reduction_kernel,
        )
    if _use_jit_coherent_stream:
        _nsys_push("cxr.lines.coherent_finalize")
        if decoherence_active:
            # The planes now hold |sum_e S_e| per row (the "flat" term);
            # snapshot its magnitude before the grouped passes reuse them.
            flat_mag2 = _stream_field_mag2()
            # sum_e |S_e|^2. Stable sorting makes each electron contiguous;
            # whole-electron blocks then keep launch count O(n_seg/seg_block)
            # instead of O(Ne). The segmented reducer assigns electron
            # groups to threads and squares each field before block reduction.
            grouped_mag2 = xp.zeros((N_g, E_grid.size), dtype=REAL)
            elec_cpu = _to_cpu(seg_elec_id)
            order = np.argsort(elec_cpu, kind="stable")
            gid_e = elec_cpu[order]
            order = order[gid_e < Ne]  # secondaries never radiate a line
            gid_e = gid_e[gid_e < Ne]
            if gid_e.size:
                e_starts = np.flatnonzero(np.concatenate(([True], gid_e[1:] != gid_e[:-1])))
                e_bounds = np.append(e_starts, gid_e.size)
                group0 = 0
                n_groups = e_bounds.size - 1
                while group0 < n_groups:
                    target = int(e_bounds[group0]) + seg_block
                    group1 = int(np.searchsorted(e_bounds, target, side="right") - 1)
                    group1 = max(group0 + 1, min(group1, n_groups))
                    b0 = int(e_bounds[group0])
                    b1 = int(e_bounds[group1])
                    sub = xp.asarray(order[b0:b1])
                    starts = xp.asarray(e_bounds[group0 : group1 + 1] - b0, dtype=xp.uint32)
                    _stream_segment_block(
                        sub,
                        int(sub.size),
                        group_starts=starts,
                        grouped_out=grouped_mag2,
                    )
                    group0 = group1
            # F PER ROW, before the incoherent row sum.
            blended = _decoherence_blend(st, stream_F_rows, grouped_mag2, flat_mag2, stream_keep)
            spec[:] += (WM.reshape(-1)[:, None] * blended).sum(axis=0)
        else:
            finalize_coherent_fields(
                coherent_fields,
                WM.reshape(-1),
                out=spec,
                n_g=N_g,
                config=DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
            )
        _nsys_pop()
    elif coherent:
        # -- 7c. per-(reflection, orientation) coherent reduction ----------
        # Sum the phased complex field over ALL of ONE row's segments, square
        # it, and add |F_s|^2 + |F_p|^2 weighted by that row's mosaic weight.
        # Rows are reduced independently, so reflections and mosaic
        # orientations stay INCOHERENT (docs/physics/materials/crystal-mosaicity.md route 2)
        # while the segment sum inside a row keeps its phase -- the defining
        # property the per-hkl coherent path had, preserved verbatim.
        # Limiting case: one row, one piece collapses to |A|^2 t_L^2 |F|^2,
        # whose energy integral is the incoherent self-term's
        # (test_coherent_emission.py, test_coherent_formation_absorption.py).
        _nsys_push("cxr.lines.accum")
        # ONE sync for every block's row lengths, then pure slicing.
        counts = _to_cpu(xp.stack(coh_counts)) if coh_blocks else np.zeros((0, N_g), int)
        starts = np.concatenate(
            [np.zeros((counts.shape[0], 1), counts.dtype), np.cumsum(counts, axis=1)], axis=1
        )
        for i_row in range(N_g):
            blocks = [
                [f[int(starts[b, i_row]) : int(starts[b, i_row + 1])] for f in flat]
                for b, flat in enumerate(coh_blocks)
                if counts[b, i_row]
            ]
            if not blocks:
                continue
            if len(blocks) == 1:
                row = blocks[0]
            else:
                row = [xp.concatenate(parts) for parts in zip(*blocks, strict=True)]
            E_r_i, aw_i, ps_i, gp_i, csr, csi, cpr, cpi = row[:8]
            L_i = row[8]  # escape distance for the in-medium phase
            elec_id_i = row[9]  # emitting electron id (decoherence_active only)
            half_dL_i, apb_i, bma_i, q_i = row[10:14]  # formation factor
            wm_i = float(wm_rows[i_row])
            # The all-electron term runs only on energies ``keep`` the
            # omission bound does not license dropping (all when inactive).
            F_row = keep = None
            if decoherence_active:
                F_row = _row_decoherence_factor(st, G[i_row])
                keep = _flat_energy_keep(st, F_row)
            flat_st = _flat_view(st, keep)
            E_flat = flat_st.E_grid
            dom_flat = flat_st.delta_omega_grid
            if _use_jit_coherent_reduction:
                per_line_i = (E_r_i, aw_i, ps_i, gp_i, csr, csi, cpr, cpi)
                formation_i = (half_dL_i, apb_i, bma_i, q_i)
                out_flat = spec if not decoherence_active else xp.zeros(E_flat.size, dtype=REAL)
                if E_flat.size:
                    run_coherent_reduction_kernel(
                        *per_line_i,
                        E_flat,
                        out=out_flat,
                        mosaic_weight=1.0 if decoherence_active else wm_i,
                        L_esc=L_i,
                        delta_omega=dom_flat,
                        half_dL=half_dL_i,
                        apb=apb_i,
                        bma=bma_i,
                        q=q_i,
                        config=DEFAULT_COHERENT_KERNEL_CONFIG,
                    )
                if decoherence_active:
                    grouped_total = _coherent_jit_grouped_row(
                        st,
                        elec_id_i,
                        per_line_i,
                        L_i,
                        formation_i,
                        xp.zeros(E_grid.size, dtype=REAL),
                    )
                    spec[:] += _decoherence_blend(st, F_row, grouped_total, out_flat, keep) * wm_i
                continue
            f_s = xp.zeros(E_flat.size, dtype=cdtype)
            f_p = xp.zeros(E_flat.size, dtype=cdtype)
            for j0 in range(0, E_r_i.size if E_flat.size else 0, chunk):
                sl = slice(j0, min(j0 + chunk, E_r_i.size))
                arg = ps_i[sl][:, None] * E_flat[None, :] - gp_i[sl][:, None]
                arg = arg - L_i[sl][:, None] * dom_flat[None, :]
                ph = xp.exp(1j * arg)
                F = formation_profile(
                    E_flat,
                    dom_flat,
                    E_r_i[sl],
                    aw_i[sl],
                    half_dL_i[sl],
                    apb_i[sl],
                    bma_i[sl],
                    q_i[sl],
                    sinc_cutoff=None,
                    xp=xp,
                )
                SP = F.astype(cdtype) * ph
                f_s += (csr[sl] + 1j * csi[sl]) @ SP
                f_p += (cpr[sl] + 1j * cpi[sl]) @ SP
            flat_total = xp.abs(f_s) ** 2 + xp.abs(f_p) ** 2
            if decoherence_active:
                grouped_total = _coherent_electron_grouped_row(
                    st,
                    elec_id_i,
                    _FormationLines((E_r_i, aw_i, half_dL_i, apb_i, bma_i, q_i)),
                    ps_i * HBARC_EV_ANG,  # ps_i is d_geom/HBARC_EV_ANG; undo the fold
                    gp_i,
                    L_i,
                    [csr + 1j * csi, cpr + 1j * cpi],
                )
                spec[:] += _decoherence_blend(st, F_row, grouped_total, flat_total, keep) * wm_i
            else:
                spec[:] += flat_total * wm_i
        _nsys_pop()
