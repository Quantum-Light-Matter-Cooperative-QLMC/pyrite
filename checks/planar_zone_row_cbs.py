"""Native PXR/CBS on planar zone rows against the in-cut free→free block (#1055).

For each requested plane incidence ``psi`` this driver builds one planar
channeling case (Si(110) by default: plane ``(1,-1,0)``; the slab is tilted 3°
from the catalog cut's surface axis, ⟨110⟩ for silicon and ⟨001⟩ for diamond,
far outside any axial critical angle; diamond uses (0,1,-1)) with the tilt azimuth solved so the
vacuum angle to the planes is ``psi``), transports it with the production runner and scores,
on the beam axis and on the same line grid:

* ``native_zone``: the production native line kernel (``_cxr_for_segments``,
  incoherent) on the zone rows only, split into the channeling plane's own
  reciprocal row (``g`` parallel to the plane normal) and every other row.
  Per-electron normalized; the standard error comes from electron chunks.
* ``spec_channeling``: the production planar source (bound→bound + capture),
  :func:`case_channeling_spectrum` exactly as the runner calls it.
* ``free_free``: the same call with the diagnostic ``final_states="tracked"``
  catalog; its ``free_free_spectrum`` is the in-cut free→free block (lowering
  currents between tracked above-barrier states, ``E < barrier + margin``).
  It is an expected value on the same censored paths, so it has no sampling
  error of its own.

The planar potential holds only the plane's own row, so only own-row native
emission can overlap the density-matrix source; other rows are reported for
completeness. Band integrals are photons/sr per incident line electron.

    PYRITE_MC_BACKEND=cpu uv run python checks/planar_zone_row_cbs.py \
        --psi-mrad 0 2.2 --electrons 2000 --out scratch/zone-cbs.json

Heavy settings (production basis, many electrons, several cases) go through
``pyrite remote``.

Validation: cr-planar-free-free-block
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import pyrite as pr
from pyrite.campaign.lowering import build_case
from pyrite.config import FixedSampling
from pyrite.detectors import EnergyBins, PointDetector
from pyrite.montecarlo import runner
from pyrite.montecarlo.channeling.handoff import (
    CHANNELING_MODEL,
    case_plane_incidence,
    zone_solver,
)
from pyrite.montecarlo.runner import channeling_emission

TILT_DEG = 3.0
# {110} planes that contain each catalog cut's surface axis: [110] for silicon
# (surface 440), [100] for diamond (surface 400).
PLANES = {"silicon": (1, -1, 0), "diamond": (0, 1, -1)}


def _case(args, crystal, azimuth_deg):
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=args.electrons, brem_histories=1),
        energy_bins=EnergyBins(
            cxr=np.linspace(args.e_min_keV * 1e3, args.e_max_keV * 1e3, args.n_energy)
        ),
        channeling_basis_size=args.basis,
        channeling_bloch_cells=args.cells,
        channeling_dz_ang=args.dz,
    )
    physics = pr.Physics(
        channeling_model="planar", channeling_plane=PLANES[crystal], radiative_model="uncoupled"
    )
    beam = pr.Beam(
        energy_keV=args.energy_MeV * 1e3,
        transverse=pr.beams.Spot(),
        longitudinal=pr.beams.PointBunch(),
    )
    target = pr.geometry.Slab(
        crystal,
        thickness_ang=args.thickness_um * 1e4,
        tilt_deg=TILT_DEG,
        tilt_azim_deg=azimuth_deg,
        footprint=pr.geometry.Footprint(5.0, 5.0),
    )
    return build_case(pr.Scene(beam, target, PointDetector()), numerics, physics=physics)


def _azimuth_for(args, crystal, psi_rad):
    """Tilt azimuth [deg] whose vacuum plane angle is ``psi_rad``.

    Scans for the azimuth where the tilted beam lies in the planes (the
    catalog cut differs per crystal), then solves on the rising side of it.
    """

    def angle(azimuth):
        return case_plane_incidence(_case(args, crystal, azimuth)).geometric_rad

    grid = np.arange(0.5, 360.0, 2.0)  # avoids the banned 90° azimuth
    values = np.array([angle(a) for a in grid])
    rising = np.flatnonzero((values[:-1] < 0.0) & (values[1:] >= 0.0))
    if not rising.size:
        raise RuntimeError(f"tilted beam never lies in the {PLANES[crystal]} planes of {crystal}")
    lo = grid[rising[0]]
    zero = brentq(angle, lo, lo + 2.0, xtol=1e-12)
    if psi_rad == 0.0:
        return zero
    step = 0.5
    hi = zero
    while angle(hi) < psi_rad:
        hi += step
    return brentq(lambda a: angle(a) - psi_rad, zero, hi, xtol=1e-12)


class _Override(Mapping):
    def __init__(self, base, **extra):
        self._base, self._extra = base, extra

    def __getitem__(self, key):
        return self._extra[key] if key in self._extra else self._base[key]

    def __iter__(self):
        return iter({*self._base, *self._extra})

    def __len__(self):
        return len({*self._base, *self._extra})


def _own_row(hkl, normal_hkl):
    hkl = np.asarray(hkl, dtype=float)
    normal = np.asarray(normal_hkl, dtype=float)
    return bool(np.allclose(np.cross(hkl, normal), 0.0))


def _subset(segs, keep, first_id):
    """Rows ``keep`` with electron ids shifted so the chunk starts at 0.

    Every row-shaped array (aliases included) is sliced. The line kernel bins
    rows by ``electron_id < Ne``; a chunk scored with ``Ne = chunk size`` must
    carry ids ``0 .. size - 1``.
    """
    n = int(np.asarray(segs["L_ang"]).shape[0])
    out = {
        key: (
            np.asarray(value)[keep]
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n
            else value
        )
        for key, value in segs.items()
    }
    for key in ("electron_id", "elec_id"):
        if key in out:
            out[key] = out[key] - int(first_id)
    return out


def _native(case, tp, segs, hkls, count):
    if not hkls:
        return np.zeros(np.asarray(tp["E_grid"]).size)
    # The planar case carries a channeling record; native_line_rows only acts
    # on axial cases, so the rows passed here are the rows scored.
    value = runner._cxr_for_segments(
        segs,
        np.asarray(tp["E_grid"], dtype=float),
        _Override(case, hkl_list=hkls),
        tp["n_hat"],
        case.get("abs_layers"),
        tp.get("groove"),
        coherent=False,
        Ne=count,
        table_cache={},
    )
    return np.asarray(value, dtype=float)


def _reset_solver(case):
    """Put the cached zone solver back on its ``gamma0`` basis (#1090).

    Transport's stopping-aware zone solve leaves the shared solver on the
    exit-face basis; the first CR score would then pair a stale catalog with a
    ``gamma0`` density. Both arms here are scored on the consistent basis.
    """
    record = case["channeling"]
    _profile, solver = zone_solver(
        case["crystal"],
        tuple(record["plane_hkl"]),
        float(record["temperature_K"]),
        float(case["E0_keV"]) * 1e3,
        int(record["basis_size"]),
        int(record["bloch_cells"]),
        float(record["margin_eV"]),
    )
    solver.basis, solver._scale = solver._basis(solver.gamma0), 1.0


def _spec_channeling(case, tp):
    """Production source and the tracked-final (free→free) arm on the same call."""
    _reset_solver(case)
    captured = {}
    original = channeling_emission.channeling_spectrum

    def tracked(*a, **kw):
        result = original(*a, final_states="tracked", **kw)
        captured["result"] = result
        return result

    production = channeling_emission.case_channeling_spectrum(case, tp)
    channeling_emission.channeling_spectrum = tracked
    try:
        channeling_emission.case_channeling_spectrum(case, tp)
    finally:
        channeling_emission.channeling_spectrum = original
    result = captured["result"]
    bound_arm = result.spectrum - result.free_free_spectrum
    return np.asarray(production), np.asarray(result.free_free_spectrum), bound_arm


def _own_line_eV(case, psi_rad):
    """Forward (quantum, classical) energies of the own-row first-order line [eV].

    Nearly-free ``k -> k - g`` with ``k = p psi / hbar``:
    quantum ``hbar^2 (2 k g - g^2) / (2 gamma m)``, classical ``hbar v k g / p``
    (``= hbar c beta g psi``), both over ``1 - beta``. ``None`` at ``psi = 0``.
    """
    from pyrite.materials.channeling import planar_profile
    from pyrite.materials.crystal import HBAR2_2M_EV_ANG2, HBARC_EV_ANG

    if abs(psi_rad) < 1e-9:
        return None
    record = case["channeling"]
    g = planar_profile(
        case["crystal"], tuple(record["plane_hkl"]), float(record["temperature_K"])
    ).g_inv_ang
    gamma = 1.0 + float(case["E0_keV"]) / 510.99895
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    k = gamma * beta * 510998.95 / HBARC_EV_ANG * abs(psi_rad)
    one_minus_beta = 1.0 / (gamma**2 * (1.0 + beta))
    quantum = HBAR2_2M_EV_ANG2 * (2.0 * k * g - g**2) / gamma / one_minus_beta
    classical = HBARC_EV_ANG * beta * g * abs(psi_rad) / one_minus_beta
    return float(quantum), float(classical)


def _band(E, spectrum, lo, hi):
    mask = (E >= lo) & (E <= hi)
    return float(np.trapezoid(spectrum[..., mask], E[mask], axis=-1).sum()) if mask.any() else 0.0


def run_one(args, crystal, psi_mrad):
    psi = psi_mrad * 1e-3
    azimuth = _azimuth_for(args, crystal, psi)
    case = _case(args, crystal, azimuth)
    incidence = case_plane_incidence(case)
    started = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tp = runner._transport_case(case, transport_core="per-electron")
    segs = tp["segs"]
    count = int(tp["Ne_lines"])
    E = np.asarray(tp["E_grid"], dtype=float)
    zone = np.asarray(segs["channeling_zone"], dtype=bool)
    ids = np.asarray(segs["electron_id"])
    hkls = [tuple(int(v) for v in h) for h in case["hkl_list"]]
    normal_hkl = PLANES[crystal]
    own = [h for h in hkls if _own_row(h, normal_hkl)]
    other = [h for h in hkls if not _own_row(h, normal_hkl)]
    chunks = np.array_split(np.arange(count), args.chunks)
    per_chunk = {"own": [], "other": []}
    for members in chunks:
        keep = zone & np.isin(ids, members)
        sub = _subset(segs, keep, members[0])
        per_chunk["own"].append(_native(case, tp, sub, own, members.size))
        per_chunk["other"].append(_native(case, tp, sub, other, members.size))
    weights = np.array([m.size for m in chunks], dtype=float) / count
    native = {k: np.tensordot(weights, np.array(v), axes=1) for k, v in per_chunk.items()}
    production, free_free, bound_arm = _spec_channeling(case, tp)
    scale = float(np.max(np.abs(production + free_free)))
    defect = float(np.max(np.abs(bound_arm - production))) / scale if scale else 0.0
    if defect > 1e-10:
        raise AssertionError(f"tracked catalog changed the bound/capture part ({defect:.2e})")
    info = segs["channeling"]
    fates = {k: int(v) for k, v in (info.get("fate_counts") or {}).items()}
    bands = {"whole": (E[0], E[-1])}
    for name, (lo, hi) in (args.bands or {}).items():
        bands[name] = (lo * 1e3, hi * 1e3)
    own_line = _own_line_eV(case, incidence.effective_rad)
    if own_line is not None:
        quantum, classical = own_line
        bands["own_line"] = (0.85 * quantum, 1.15 * classical)
    out_bands = {}
    for name, (lo, hi) in bands.items():
        chunk_vals = {k: np.array([_band(E, s, lo, hi) for s in v]) for k, v in per_chunk.items()}
        ff = _band(E, free_free, lo, hi)
        row = {"band_eV": [lo, hi], "free_free_per_sr": ff}
        row["spec_channeling_per_sr"] = _band(E, production, lo, hi)
        for k, vals in chunk_vals.items():
            mean = float(weights @ vals)
            sem = float(
                np.sqrt(np.sum(weights**2 * (vals - mean) ** 2) * len(vals) / (len(vals) - 1))
            )
            row[f"native_zone_{k}_per_sr"] = mean
            row[f"native_zone_{k}_sem"] = sem
            if ff > 0:
                row[f"ratio_{k}_to_free_free"] = mean / ff
                row[f"ratio_{k}_to_free_free_sem"] = sem / ff
        out_bands[name] = row
    return {
        "crystal": crystal,
        "plane_hkl": list(normal_hkl),
        "psi_mrad_target": psi_mrad,
        "incidence": {
            "geometric_mrad": incidence.geometric_rad * 1e3,
            "internal_mrad": incidence.internal_rad * 1e3,
            "effective_mrad": incidence.effective_rad * 1e3,
            "tilt_deg": TILT_DEG,
            "tilt_azim_deg": azimuth,
        },
        "own_line_eV": _own_line_eV(case, incidence.effective_rad),
        "own_row_hkl": own,
        "n_other_rows": len(other),
        "electrons": count,
        "zone_rows": int(zone.sum()),
        "mean_zone_length_ang": float(np.asarray(segs["L_ang"])[zone].mean())
        if zone.any()
        else 0.0,
        "fates": fates,
        "bands": out_bands,
        "photon_energy_eV": E.tolist(),
        "spectra_per_eV_sr": {
            "spec_channeling": production.ravel().tolist(),
            "free_free": free_free.ravel().tolist(),
            "native_zone_own": native["own"].ravel().tolist(),
            "native_zone_other": native["other"].ravel().tolist(),
        },
        "seconds": time.time() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crystal", choices=sorted(PLANES), default="silicon")
    parser.add_argument("--energy-MeV", type=float, default=17.0)
    parser.add_argument("--thickness-um", type=float, default=20.0)
    parser.add_argument("--psi-mrad", type=float, nargs="+", default=[0.0, 2.2])
    parser.add_argument("--electrons", type=int, default=2000)
    parser.add_argument("--chunks", type=int, default=20)
    parser.add_argument("--basis", type=int, default=16)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--dz", type=float, default=250.0)
    parser.add_argument("--e-min-keV", type=float, default=5.0)
    parser.add_argument("--e-max-keV", type=float, default=60.0)
    parser.add_argument("--n-energy", type=int, default=551)
    parser.add_argument(
        "--band",
        action="append",
        default=[],
        metavar="NAME:LO:HI",
        help="extra band in keV, e.g. cbs:25:40",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    args.bands = {
        b.split(":")[0]: (float(b.split(":")[1]), float(b.split(":")[2])) for b in args.band
    }
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:  # no git on the compute node: the synced checkout's SHA file
        sha = ""
    if not sha and Path("SHA").is_file():
        sha = Path("SHA").read_text().strip()
    record = {
        "status": "research evidence for #1055: native PXR/CBS on planar zone rows vs the "
        "in-cut free->free block of the density-matrix source",
        "sha": sha,
        "channeling_model": CHANNELING_MODEL,
        "driver": "checks/planar_zone_row_cbs.py",
        "settings": {k: v for k, v in vars(args).items() if k not in {"out", "band"}},
        "cases": [run_one(args, args.crystal, psi) for psi in args.psi_mrad],
    }
    with open(args.out, "w") as handle:
        json.dump(record, handle, indent=1)
    for case in record["cases"]:
        for name, row in case["bands"].items():
            print(
                f"{case['crystal']} psi={case['incidence']['geometric_mrad']:.3f} mrad "
                f"{name}: own={row['native_zone_own_per_sr']:.3e}±{row['native_zone_own_sem']:.1e} "
                f"other={row['native_zone_other_per_sr']:.3e} ff={row['free_free_per_sr']:.3e} "
                f"CR={row['spec_channeling_per_sr']:.3e} "
                f"own/ff={row.get('ratio_own_to_free_free', float('nan')):.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
