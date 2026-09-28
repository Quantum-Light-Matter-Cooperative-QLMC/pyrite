"""Per-line yield attribution for the incoherent line route (issue #201).

A measurement instrument only: when a truncation audit carries an
``"attribute"`` config, each segment block appends its heaviest lines (by line
mass ``w pi / a_w``) with the factors that built their weight, plus each
electron's summed line mass, to ``audit["attribution"]``. Nothing here changes
a weight, a mask, or the spectrum.
"""

import numpy as np

from ...._backend import _to_cpu, xp

#: Per-line factors recorded for the heaviest lines, by ``_BatchedBlock`` name
#: or the incoherent-block local that holds them.
_BLOCK_FIELDS = (
    "vx",
    "vy",
    "vz",
    "denom",
    "gamma",
    "t_L",
    "L_esc",
    "detuning",
    "k_dot_g",
    "vdg",
    "k_mag",
    "chi_re",
    "chi_im",
    "u_re",
    "u_im",
    "mu",
)


def record_line_attribution(audit, st, bt, blk, gm_idx, E_r, a_width, weight, factors):
    """Append this block's heaviest lines and per-electron masses to ``audit``.

    ``gm_idx`` indexes the flattened ``(n_block, N_g)`` grid of surviving lines;
    ``factors`` maps extra names (``T_abs``, ``A2``, ...) to arrays that
    broadcast to that grid. With ``stop_eV`` in the config, lines are also
    ranked by their tail-bound mass above that edge -- the quantity the
    measured bandwidth search spends (``line_seeds.sincsq_upper_tail_bound``).
    """
    config = audit.get("attribute") if audit is not None else None
    if config is None:
        return
    n_g = int(bt.N_g)
    shape = blk.omega_res.shape
    mass = weight * (xp.pi / a_width)
    stop = config.get("stop_eV")
    if stop is None:
        tail = xp.zeros_like(mass)
    else:
        distance = stop - E_r
        bound = 1.0 / (xp.pi * a_width * xp.maximum(distance, xp.finfo(E_r.dtype).tiny))
        tail = mass * xp.where(distance > 0.0, xp.minimum(bound, 1.0), 1.0)
    elec = st.seg_elec_id[blk.sb][gm_idx // n_g]
    electron_mass = xp.bincount(elec, weights=mass.astype(xp.float64))
    electron_tail = xp.bincount(elec, weights=tail.astype(xp.float64), minlength=electron_mass.size)
    hit = xp.flatnonzero(electron_mass)
    fields = {name: getattr(blk, name) for name in _BLOCK_FIELDS}
    fields.update(factors)
    fields.update(gx=bt.gx, gy=bt.gy, gz=bt.gz, WM=bt.WM)
    top = int(config.get("top", 64))
    record = {
        "electron_ids": np.asarray(_to_cpu(hit)),
        "electron_mass": np.asarray(_to_cpu(electron_mass[hit]), dtype=float),
        "electron_tail": np.asarray(_to_cpu(electron_tail[hit]), dtype=float),
    }
    for rank, key in (("mass", mass), ("tail", tail)):
        pick = xp.argsort(-key)[:top] if key.size > top else xp.arange(key.size)
        flat = gm_idx[pick]
        r, c = flat // n_g, flat % n_g
        lines = {
            "segment": np.asarray(_to_cpu(blk.sb.start + r)),
            "g_row": np.asarray(_to_cpu(c)),
            "electron": np.asarray(_to_cpu(elec[pick])),
            "E_res_eV": np.asarray(_to_cpu(E_r[pick]), dtype=float),
            "width_eV": np.asarray(_to_cpu(xp.pi / a_width[pick]), dtype=float),
            "weight": np.asarray(_to_cpu(weight[pick]), dtype=float),
            "mass": np.asarray(_to_cpu(mass[pick]), dtype=float),
            "tail": np.asarray(_to_cpu(tail[pick]), dtype=float),
        }
        for name, value in fields.items():
            lines[name] = np.asarray(_to_cpu(xp.broadcast_to(value, shape)[r, c]), dtype=float)
        record[f"lines_by_{rank}"] = lines
    audit.setdefault("attribution", []).append(record)


def merge_line_attribution(records, top):
    """Global top ``top`` lines by mass and by tail, and per-electron sums."""
    empty = np.zeros(0)
    if not records:
        return {
            "lines_by_mass": {},
            "lines_by_tail": {},
            "electron_ids": np.zeros(0, int),
            "electron_mass": empty,
            "electron_tail": empty,
        }
    ids = np.concatenate([r["electron_ids"] for r in records])
    electron_ids, inverse = np.unique(ids, return_inverse=True)
    merged = {"electron_ids": electron_ids}
    for key in ("electron_mass", "electron_tail"):
        merged[key] = np.bincount(inverse, weights=np.concatenate([r[key] for r in records]))
    for rank in ("mass", "tail"):
        name = f"lines_by_{rank}"
        lines = {k: np.concatenate([r[name][k] for r in records]) for k in records[0][name]}
        order = np.argsort(-lines[rank])[:top]
        merged[name] = {k: v[order] for k, v in lines.items()}
    return merged
