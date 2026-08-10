"""results.select_thickness: pin a thickness across a multi-beam overlay without
dropping a beam energy the penetration watchdog stopped computing early.

Synthetic store shaped like a watchdog-gated sweep: the low beam (30 keV) is
computed only at thin slabs, the high beam (300 keV) at every slab. No
Monte-Carlo run needed.
"""

import numpy as np

from pyrite.results import records, select_thickness, thicknesses_by_energy


def _record(E0, thickness_ang):
    return {"case": {"E0_keV": E0, "thickness_ang": thickness_ang}, "spec": np.zeros(4)}


def _store():
    # 30 keV dies early: computed at 1e4, 5e4 only. 300 keV: 1e4, 5e4, 1e6.
    layout = {30.0: [1e4, 5e4], 300.0: [1e4, 5e4, 1e6]}
    store = {}
    for E0, thicknesses in layout.items():
        for t in thicknesses:
            store[f"cfg_{E0:g}_{t:g}"] = {E0: _record(E0, t)}
    return store


def _energies_and_fallbacks(view):
    present, fallback = set(), set()
    for r in records(view):
        c = r["case"]
        present.add(c["E0_keV"])
        if "thickness_fallback" in c:
            fallback.add(c["E0_keV"])
    return present, fallback


def test_inventory_lists_computed_slabs_per_energy():
    inv = thicknesses_by_energy(_store())
    assert inv == {30.0: [1e4, 5e4], 300.0: [1e4, 5e4, 1e6]}


def test_exact_thickness_keeps_both_energies_without_fallback():
    present, fallback = _energies_and_fallbacks(select_thickness(_store(), 1e4))
    assert present == {30.0, 300.0}
    assert fallback == set()


def test_bulk_thickness_substitutes_low_energy_with_marker():
    view = select_thickness(_store(), 1e6)
    present, fallback = _energies_and_fallbacks(view)
    # 300 keV computed at 1e6; 30 keV falls back to its thickest slab (5e4).
    assert present == {30.0, 300.0}
    assert fallback == {30.0}
    low = next(r for r in records(view) if r["case"]["E0_keV"] == 30.0)
    assert low["case"]["thickness_fallback"] == (1e6, 5e4)
    assert low["case"]["thickness_ang"] == 5e4


def test_fallback_copies_case_and_never_mutates_input():
    store = _store()
    select_thickness(store, 1e6)
    for by_E in store.values():
        for r in by_E.values():
            assert "thickness_fallback" not in r["case"]


def test_mid_thickness_uncomputed_by_low_energy_still_falls_back():
    # 30 keV never computed 5e4? it did (5e4 in its layout) -> exact, no fallback.
    # Ask for a thickness only the high beam computed but below its max: 30 keV
    # has no record there, so it falls back to its own thickest slab (5e4).
    view = select_thickness(_store(), 1e6)
    inv = thicknesses_by_energy(_store())
    assert inv[30.0][-1] == 5e4
    low = next(r for r in records(view) if r["case"]["E0_keV"] == 30.0)
    assert low["case"]["thickness_ang"] == inv[30.0][-1]
