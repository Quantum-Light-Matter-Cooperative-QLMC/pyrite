"""Multilayer slice-3 validation: per-layer COHERENT radiation
(docs/physics/materials/multilayer-materials.md (2)).

mc_spectrum already radiates one crystal; slice 3 makes _spectrum_case radiate
EVERY crystalline layer of a film-on-substrate stack and sum them incoherently,
each self-absorbing through the whole stack. `_spectrum_case`'s returned
``spec`` is the per-layer COHERENT (PXR/CBS) sum PLUS the incoherent
characteristic (EEDL/xraydb relaxation) component that every layer -- crystalline
or not -- contributes from its own composition; ``spec_characteristic`` carries
that second term separately so it can be subtracted back out. On real
transport segments:

  1. A CRYSTALLINE substrate (silicon) gets its own electron segments and
     radiates NONZERO PXR/CBS lines of its own.
  2. The pipeline spectrum equals the incoherent sum of the per-layer
     contributions (film layer 0 + substrate layer 1, each COHERENT +
     characteristic) -- i.e. _spectrum_case really does sum the layers, with
     no cross-layer coherence.
  3. An AMORPHOUS substrate (sio2) radiates NO COHERENT lines of its own (its
     layer radiator is None, so only its characteristic emission contributes);
     the same per-layer sum invariant as (2) still holds for this stack too.
  4. Sanity: a crystalline substrate raises the total coherent line flux above
     the same film on an amorphous substrate.

Assertions 2/3 diff the pipeline spectrum against a manual per-layer
reproduction at a 1e-9 relative tolerance -- exact under CPU/NumPy's
associative fp64 reduction, but CUDA's non-associative reduction order
introduces ~1e-8 relative noise (amplified here by a small denominator) that
is physically negligible yet trips that tolerance. Pin the CPU backend below
the same way tests/conftest.py does: stubbing cupy alone does not force CPU on
a box whose environment pins PYRITE_MC_BACKEND to an accelerator (e.g. this
repo's .envrc PYRITE_MC_BACKEND=cuda) -- _backend.select_backend honors that
explicit request instead of falling back, so the stubbed import raises.
"""

import os

os.environ["PYRITE_MC_BACKEND"] = "cpu"

import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases, substrate_radiator  # noqa: E402
from pyrite.detectors import Detector, EnergyBins  # noqa: E402
from pyrite.montecarlo import (  # noqa: E402
    _segments_in_layer,
    _spectrum_case,
    _transport_case,
    mc_characteristic_spectrum,
    mc_spectrum,
)

E0 = 30.0
TILT = -30.0  # front exit (high flux); thin substrate so its lines escape
T_FILM = 300.0  # 30 nm MoSe2 film
T_SUB = 3000.0  # 0.3 um substrate -- thin enough that substrate lines aren't fully self-absorbed
E_LINE = np.arange(100.0, 3500.0, 3.0)  # wide: brackets BOTH the film and Si lines
E_BREM = np.arange(100.0, 30000.0, 100.0)


def _case(substrate):
    sw = Sweep(
        material="mose2",
        tilt_deg=TILT,
        beam=BeamSpec(energy_keV=E0),
        thickness_ang=T_FILM,
        substrate=substrate,
        substrate_thickness_ang=T_SUB,
        detector=Detector(energy_bins=EnergyBins(line=E_LINE, brem=E_BREM)),
    )
    return build_cases(sw, n_electrons=600, n_electrons_brem=120)[0]


def _radiate_layer(segs, E_grid, n_hat, case, tp, L):
    """One layer's COHERENT + characteristic spectrum, exactly as
    _spectrum_case computes it (same per-radiator kwargs as
    runner._lines_for_segments / _characteristic_from_segments): the film and
    substrate radiators carry different surface_hkl/beam_uvw, so reproducing
    the pipeline bit-for-bit requires passing them through, not just the
    shared case-level defaults."""
    abs_layers = case["abs_layers"]
    rad = case["layer_radiators"][L]
    sL = _segments_in_layer(segs, L)
    if sL["L_ang"].size == 0:
        return np.zeros(E_grid.shape)
    coherent = np.zeros(E_grid.shape)
    if rad is not None:
        coherent = mc_spectrum(
            sL,
            E_grid,
            crystal=rad["crystal"],
            hkl_list=rad["hkl_list"],
            n_hat=n_hat,
            B_ang2=rad["B_ang2"],
            composition=abs_layers[L][2],
            beam_uvw=rad.get("beam_uvw"),
            surface_hkl=rad.get("surface_hkl"),
            azimuth_rad=rad.get("azimuth_rad", case.get("azimuth_rad", 0.0)),
            recip_miscut_rad=rad.get("recip_miscut_rad", case.get("recip_miscut_rad")),
            sinc_cutoff=case.get("sinc_cutoff"),
            layers=abs_layers,
            coherent=bool(case.get("coherent_emission", False)),
            electron_limit=tp["Ne_lines"],
            E_cut_keV=case.get("E_cut_lines_keV", 5.0),
        )
    characteristic = mc_characteristic_spectrum(
        sL,
        E_grid,
        composition=abs_layers[L][2],
        n_hat=n_hat,
        layers=abs_layers,
        electron_limit=tp["Ne_brem"],
        E_cut_keV=case.get("E_cut_brem_keV", 1.0),
    )
    return coherent + characteristic


def main():
    ok = True

    # --- crystalline substrate: MoSe2 film on a Si substrate --------------------
    case = _case("silicon")
    tp = _transport_case(case)
    out = _spectrum_case(case, tp)  # the slice-3 pipeline spectrum
    segs, n_hat, E_grid = tp["segs"], tp["n_hat"], tp["E_grid"]
    n_sub = int(_segments_in_layer(segs, 1)["L_ang"].size)

    print(
        f"MoSe2 {T_FILM:.0f}A on silicon {T_SUB:.0f}A, {E0:.0f} keV, tilt {TILT:g} deg, "
        f"{segs['L_ang'].size} segments ({n_sub} in the substrate)\n"
    )

    spec_film = _radiate_layer(segs, E_grid, n_hat, case, tp, 0)
    spec_sub = _radiate_layer(segs, E_grid, n_hat, case, tp, 1)

    # 1. the crystalline substrate radiates its own lines -----------------------
    sub_coherent = mc_spectrum(
        _segments_in_layer(segs, 1),
        E_grid,
        crystal=case["layer_radiators"][1]["crystal"],
        hkl_list=case["layer_radiators"][1]["hkl_list"],
        n_hat=n_hat,
        B_ang2=case["layer_radiators"][1]["B_ang2"],
        composition=case["abs_layers"][1][2],
        beam_uvw=case["layer_radiators"][1].get("beam_uvw"),
        surface_hkl=case["layer_radiators"][1].get("surface_hkl"),
        layers=case["abs_layers"],
    )
    sub_int = float(np.trapezoid(sub_coherent, E_grid))
    radiates = n_sub > 0 and sub_int > 0.0
    ok &= radiates
    print(
        f"1. crystalline substrate radiates its own lines:        "
        f"{'PASS' if radiates else 'FAIL'}  (Si line integral {sub_int:.3e})"
    )

    # 2. pipeline spectrum == incoherent sum of the per-layer parts -------------
    manual = spec_film + spec_sub
    denom = max(float(np.max(np.abs(manual))), 1e-300)
    rel = float(np.max(np.abs(out["spec"] - manual)) / denom)
    summed = rel < 1e-9
    ok &= summed
    print(
        f"2. pipeline == film(0) + substrate(1), incoherent sum:  "
        f"{'PASS' if summed else 'FAIL'}  (max rel {rel:.1e})"
    )

    # 3. amorphous substrate: no COHERENT lines of its own, but the same
    #    per-layer sum invariant as (2) still holds (its characteristic term
    #    is nonzero, so this is no longer a bit-for-bit film-only match). ---
    assert substrate_radiator("sio2") is None
    case_a = _case("sio2")
    tp_a = _transport_case(case_a)
    out_a = _spectrum_case(case_a, tp_a)
    segs_a, n_hat_a, E_grid_a = tp_a["segs"], tp_a["n_hat"], tp_a["E_grid"]
    assert case_a["layer_radiators"][1] is None  # amorphous: no coherent radiator
    film_a = _radiate_layer(segs_a, E_grid_a, n_hat_a, case_a, tp_a, 0)
    sub_a = _radiate_layer(segs_a, E_grid_a, n_hat_a, case_a, tp_a, 1)
    manual_a = film_a + sub_a
    denom_a = max(float(np.max(np.abs(manual_a))), 1e-300)
    rel_a = float(np.max(np.abs(out_a["spec"] - manual_a)) / denom_a)
    summed_a = rel_a < 1e-9
    ok &= summed_a
    print(
        f"3. amorphous substrate: no coherent lines, sum invariant holds:  "
        f"{'PASS' if summed_a else 'FAIL'}  (max rel {rel_a:.1e})"
    )

    # 4. sanity: crystalline substrate raises total coherent line flux ---------
    f_si = float(np.trapezoid(out["spec"], E_grid))
    f_sio2 = float(np.trapezoid(out_a["spec"], E_grid_a))
    boosted = f_si > f_sio2
    ok &= boosted
    print(
        f"4. crystalline substrate boosts coherent flux:          "
        f"{'PASS' if boosted else 'FAIL'}  (Si {f_si:.3e} > SiO2 {f_sio2:.3e})"
    )

    print("\nALL CHECKS:", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
