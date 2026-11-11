"""Multilayer slice-3 validation: per-layer PXR/CBS and characteristic emission
(docs/physics/materials/multilayer-materials.md (2)).

`_spectrum_case` scores each crystalline layer of a film-on-substrate stack
with self-absorption through the whole stack. It returns the
incoherently summed PXR/CBS density in ``spec`` and the characteristic
(EEDL/xraydb relaxation) density in ``spec_characteristic``. Both are checked
against independent per-layer sums on the same transport segments:

  1. A CRYSTALLINE substrate (silicon) gets its own electron segments and
     radiates NONZERO PXR/CBS lines of its own.
  2. Each pipeline component equals the sum of that component from the film
     and substrate.
  3. An AMORPHOUS substrate (sio2) radiates NO COHERENT lines of its own (its
     layer radiator is None, so only its characteristic emission contributes);
     the same per-layer sum invariant as (2) still holds for this stack too.
  4. Sanity: a crystalline substrate raises the PXR/CBS line flux above
     the same film on an amorphous substrate.

Assertions 2 and 3 compare each component at a 1e-9 relative tolerance.
The check pins CPU scoring because CUDA reduction-order differences can exceed
that tolerance.
"""

import os

os.environ["PYRITE_MC_BACKEND"] = "cpu"

import sys  # noqa: E402

import numpy as np  # noqa: E402

from pyrite.campaign.geometry import DEFAULT_FOOTPRINT  # noqa: E402
from pyrite.campaign.sweep import BeamSpec, Stack, Sweep, build_cases  # noqa: E402
from pyrite.detectors import EnergyBins  # noqa: E402
from pyrite.montecarlo import (  # noqa: E402
    _segments_in_layer,
    _spectrum_case,
    _transport_case,
    mc_characteristic_spectrum,
    mc_cxr_spectrum,
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
        beam=BeamSpec(energy_keV=E0),
        target=Stack.on_substrate(
            "mose2", T_FILM, substrate, T_SUB, tilt_deg=TILT, footprint=DEFAULT_FOOTPRINT
        ),
        energy_bins=EnergyBins(cxr=E_LINE, brem=E_BREM),
    )
    # Use the same stopping model in both stacks. The amorphous substrate
    # has no catalog crystal for shell-resolved stopping.
    return build_cases(
        sw,
        n_electrons=600,
        n_electrons_brem=120,
        inelastic_model="continuous",
        radiative_model="uncoupled",
    )[0]


def _radiate_layer(segs, E_grid, n_hat, case, tp, L):
    """One layer's separate PXR/CBS and characteristic spectra, as
    _spectrum_case computes it (same per-radiator kwargs as
    runner._cxr_for_segments / _characteristic_from_segments): the film and
    substrate radiators carry different surface_hkl/beam_uvw, so reproducing
    the pipeline bit-for-bit requires passing them through, not just the
    shared case-level defaults."""
    abs_layers = case["abs_layers"]
    rad = case["layer_radiators"][L]
    sL = _segments_in_layer(segs, L)
    if sL["L_ang"].size == 0:
        return np.zeros(E_grid.shape), np.zeros(E_grid.shape)
    cxr = np.zeros(E_grid.shape)
    if rad is not None:
        cxr = mc_cxr_spectrum(
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
            coherent=False,
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
    return cxr, characteristic


def _component_errors(output, film, substrate):
    """Maximum error relative to each manual component's peak."""
    errors = []
    for key, film_part, substrate_part in zip(
        ("spec", "spec_characteristic"), film, substrate, strict=True
    ):
        expected = film_part + substrate_part
        peak = max(float(np.max(np.abs(expected))), 1e-300)
        errors.append(float(np.max(np.abs(output[key] - expected)) / peak))
    return errors


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
    sub_int = float(np.trapezoid(spec_sub[0], E_grid))
    radiates = n_sub > 0 and sub_int > 0.0
    ok &= radiates
    print(
        f"1. crystalline substrate radiates its own lines:        "
        f"{'PASS' if radiates else 'FAIL'}  (Si line integral {sub_int:.3e})"
    )

    # 2. pipeline spectrum == incoherent sum of the per-layer parts -------------
    rel_cxr, rel_characteristic = _component_errors(out, spec_film, spec_sub)
    summed = rel_cxr < 1e-9 and rel_characteristic < 1e-9
    ok &= summed
    print(
        f"2. pipeline == film(0) + substrate(1), incoherent sum:  "
        f"{'PASS' if summed else 'FAIL'}  (PXR/CBS {rel_cxr:.1e}, "
        f"characteristic {rel_characteristic:.1e})"
    )

    # 3. amorphous substrate: no COHERENT lines of its own, but the same
    #    per-layer sum invariant as (2) still holds. ---------------------------
    case_a = _case("sio2")
    tp_a = _transport_case(case_a)
    out_a = _spectrum_case(case_a, tp_a)
    segs_a, n_hat_a, E_grid_a = tp_a["segs"], tp_a["n_hat"], tp_a["E_grid"]
    assert case_a["layer_radiators"][1] is None  # amorphous: no coherent radiator
    film_a = _radiate_layer(segs_a, E_grid_a, n_hat_a, case_a, tp_a, 0)
    sub_a = _radiate_layer(segs_a, E_grid_a, n_hat_a, case_a, tp_a, 1)
    rel_cxr_a, rel_characteristic_a = _component_errors(out_a, film_a, sub_a)
    summed_a = rel_cxr_a < 1e-9 and rel_characteristic_a < 1e-9
    ok &= summed_a
    print(
        f"3. amorphous substrate: no coherent lines, sum invariant holds:  "
        f"{'PASS' if summed_a else 'FAIL'}  (PXR/CBS {rel_cxr_a:.1e}, "
        f"characteristic {rel_characteristic_a:.1e})"
    )

    # 4. sanity: crystalline substrate raises total coherent line flux ---------
    f_si = float(np.trapezoid(out["spec"], E_grid))
    f_sio2 = float(np.trapezoid(out_a["spec"], E_grid_a))
    boosted = f_si > f_sio2
    ok &= boosted
    print(
        f"4. crystalline substrate boosts PXR/CBS flux:           "
        f"{'PASS' if boosted else 'FAIL'}  (Si {f_si:.3e} > SiO2 {f_sio2:.3e})"
    )

    print("\nALL CHECKS:", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
