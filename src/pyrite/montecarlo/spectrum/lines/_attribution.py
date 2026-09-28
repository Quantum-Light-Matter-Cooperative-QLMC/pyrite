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
    broadcast to that grid.
    """
    config = audit.get("attribute") if audit is not None else None
    if config is None:
        return
    n_g = int(bt.N_g)
    shape = blk.omega_res.shape
    mass = weight * (xp.pi / a_width)
    row = gm_idx // n_g
    elec = st.seg_elec_id[blk.sb][row]
    electron_mass = xp.bincount(elec, weights=mass.astype(xp.float64))
    hit = xp.flatnonzero(electron_mass)
    top = int(config.get("top", 64))
    pick = xp.argsort(-mass)[:top] if mass.size > top else xp.arange(mass.size)
    flat = gm_idx[pick]
    r, c = flat // n_g, flat % n_g
    fields = {name: getattr(blk, name) for name in _BLOCK_FIELDS}
    fields.update(factors)
    fields.update(gx=bt.gx, gy=bt.gy, gz=bt.gz, WM=bt.WM)
    record = {
        "segment": np.asarray(_to_cpu(blk.sb.start + r)),
        "g_row": np.asarray(_to_cpu(c)),
        "electron": np.asarray(_to_cpu(elec[pick])),
        "E_res_eV": np.asarray(_to_cpu(E_r[pick]), dtype=float),
        "width_eV": np.asarray(_to_cpu(xp.pi / a_width[pick]), dtype=float),
        "weight": np.asarray(_to_cpu(weight[pick]), dtype=float),
        "mass": np.asarray(_to_cpu(mass[pick]), dtype=float),
    }
    for name, value in fields.items():
        record[name] = np.asarray(_to_cpu(xp.broadcast_to(value, shape)[r, c]), dtype=float)
    record["electron_ids"] = np.asarray(_to_cpu(hit))
    record["electron_mass"] = np.asarray(_to_cpu(electron_mass[hit]), dtype=float)
    audit.setdefault("attribution", []).append(record)


def merge_line_attribution(records, top):
    """Global heaviest ``top`` lines and per-electron line mass from block records."""
    if not records:
        return {"lines": {}, "electron_ids": np.zeros(0, int), "electron_mass": np.zeros(0)}
    ids = np.concatenate([r["electron_ids"] for r in records])
    masses = np.concatenate([r["electron_mass"] for r in records])
    electron_ids, inverse = np.unique(ids, return_inverse=True)
    electron_mass = np.bincount(inverse, weights=masses)
    line_keys = [k for k in records[0] if k not in ("electron_ids", "electron_mass")]
    lines = {k: np.concatenate([r[k] for r in records]) for k in line_keys}
    order = np.argsort(-lines["mass"])[:top]
    return {
        "lines": {k: v[order] for k, v in lines.items()},
        "electron_ids": electron_ids,
        "electron_mass": electron_mass,
    }
