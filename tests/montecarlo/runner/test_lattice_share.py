"""Which continua lose their lattice-coherent share, and for which reflections (#1250).

Validation: brem-lattice-coherent-fraction
"""

import warnings

import pytest

from pyrite.montecarlo.runner import lattice_share as ls
from pyrite.montecarlo.runner.reflection_cutoff import iter_reflection_shells

SI = dict(crystal="silicon", B_ang2=0.46, hkl_list=[(1, 1, 1), (2, 2, 0)])


def _case(**extra):
    return dict(SI, bremsstrahlung_model="bremslib", **extra)


def test_crystalline_bremslib_cases_are_corrected_and_others_say_why():
    assert ls.skip_reason(_case(), None) is None
    assert ls.lattice_share_applies(_case())
    assert ls.skip_reason(_case(**{ls.LATTICE_SHARE_CASE_KEY: False}), None) == "disabled"
    # The isotropic EEDL continuum is corrected too (#1261).
    assert ls.skip_reason(dict(SI, bremsstrahlung_model="eedl"), None) is None
    assert ls.lattice_share_applies(dict(SI, bremsstrahlung_model="eedl"))
    assert ls.skip_reason(_case(channeling={"model": "planar"}), None) == "channeling zone"
    assert ls.skip_reason(_case(crystal=None), None) == "amorphous layer"
    assert ls.skip_reason(_case(hkl_list=[]), None) == "no scored reflections"
    assert not ls.lattice_share_applies(_case(hkl_list=[]))


def test_layered_stacks_correct_only_their_crystalline_layers():
    case = dict(bremsstrahlung_model="bremslib", layer_radiators=[None, dict(SI)])
    assert ls.skip_reason(case, 0) == "amorphous layer"
    assert ls.skip_reason(case, 1) is None
    assert ls.scored_reflections(case, 1) == SI["hkl_list"]
    # A one-layer stack is scored as a single view (layer None).
    single = dict(bremsstrahlung_model="bremslib", layer_radiators=[dict(SI)])
    assert ls.scored_reflections(single, None) == SI["hkl_list"]


def test_measured_selection_subtracts_exactly_the_selected_shells():
    width, cutoff = 1.0, 4.0
    expected = [
        hkl
        for _, members in iter_reflection_shells("silicon", 0.46, cutoff, width)
        for hkl in members
    ]
    selected = _case(hkl_list=None, reflection_selection=dict(shell_width_invang=width))
    assert ls.continuum_case(selected, None) is selected
    record = dict(radiators=[dict(layer=None, cutoff_invang=cutoff, reflections=len(expected))])
    case = ls.continuum_case(selected, record)
    assert ls.SCORED_CUTOFFS_CASE_KEY not in selected
    assert ls.lattice_share_applies(case)
    assert ls.scored_reflections(case, None) == expected
    case[ls.SCORED_CUTOFFS_CASE_KEY] = {None: (cutoff, len(expected) + 1)}
    with pytest.raises(RuntimeError, match="would not match the lines"):
        ls.scored_reflections(case, None)


def test_eedl_crystal_gets_theta_integrated_shares_without_warning():
    from pyrite.montecarlo.spectrum.brem_lattice_share import (
        BREM_LATTICE_SHARE_ISOTROPIC_MODEL,
        BREM_LATTICE_SHARE_MODEL,
        IsotropicLatticeShare,
    )

    case = dict(SI, bremsstrahlung_model="eedl", E_cut_lines_keV=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shares, record = ls.layer_isotropic_share(case, None, [("Si", 0.05)])
    assert isinstance(shares["Si"], IsotropicLatticeShare)
    assert shares["Si"].T_keV[0] == pytest.approx(5.0)
    assert record["corrected"] and record["continuum"] == "isotropic"
    assert record["reflections"] == len(SI["hkl_list"])
    out = ls.lattice_share_record(dict(case, composition=[("Si", 0.05)]), 1)
    assert out["model"] == BREM_LATTICE_SHARE_ISOTROPIC_MODEL != BREM_LATTICE_SHARE_MODEL
    assert ls.layer_isotropic_share(dict(case, crystal=None), None, [("Si", 0.05)]) == (
        None,
        dict(layer=None, corrected=False, reason="amorphous layer"),
    )


def test_untabled_element_of_a_bremslib_run_warns(monkeypatch):
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

    tables = load_bremsstrahlung_tables(["Si"])
    case = _case(hkl_list=[(1, 1, 1), (-1, -1, -1)], E_cut_lines_keV=7.5)
    with pytest.warns(ls.LatticeShareUncorrectedWarning, match="O"):
        _, record = ls.layer_emission_tables(case, None, [("Si", 0.05), ("O", 0.01)], tables)
    assert record["elements"][1] == dict(element="O", corrected=False, reason="no BremsLib table")


def test_uncorrected_layers_pass_their_tables_through():
    tables = {"Si": object()}
    out, record = ls.layer_emission_tables(_case(hkl_list=[]), None, [("Si", 0.05)], tables)
    assert out is tables
    assert record == dict(layer=None, corrected=False, reason="no scored reflections")


def test_selection_case_without_its_record_subtracts_the_scored_families():
    # Streamed and directional routes never run the selector; their lines are
    # the build-time families, and so is the subtracted set.
    case = _case(reflection_selection=dict(shell_width_invang=1.0, automatic=True))
    assert ls.scored_reflections(case, None) == SI["hkl_list"]
    assert ls.scored_reflections(dict(case, hkl_list=None), None) == []


def test_share_only_tables_hold_what_the_correction_removes(monkeypatch):
    # The selector's S_k: base - corrected for exactly those reflections, and
    # shells add (the scorer is linear in the table; no node clamps here).
    import numpy as np

    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

    tables = load_bremsstrahlung_tables(["Si"])
    composition = [("Si", 0.05)]
    first, second = [(1, 1, 1), (-1, -1, -1)], [(2, 2, 0), (-2, -2, 0)]
    case = _case(hkl_list=first + second, E_cut_lines_keV=5.0)
    corrected, record = ls.layer_emission_tables(case, None, composition, tables)
    assert record["corrected"] and record["line_cutoff_keV"] == 5.0
    parts = [ls.share_only_tables(case, None, composition, tables, h) for h in (first, second)]
    whole = tables["Si"].scaled_ddcs_mb_sr - corrected["Si"].scaled_ddcs_mb_sr
    np.testing.assert_allclose(
        parts[0]["Si"].scaled_ddcs_mb_sr + parts[1]["Si"].scaled_ddcs_mb_sr,
        whole,
        rtol=1e-9,
        atol=1e-12 * np.abs(whole).max(),
    )
    assert (
        ls.share_only_tables(
            dict(SI, bremsstrahlung_model="eedl"), None, composition, tables, first
        )
        is None
    )
    assert ls.share_only_tables(case, None, composition, tables, []) is None


def test_layer_tables_keep_every_node_below_the_line_cutoff():
    # No CBS line is scored for electrons below E_cut_lines_keV, so the
    # continuum keeps its full DDCS there (v2 floor); above it, it loses share.
    import numpy as np

    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

    tables = load_bremsstrahlung_tables(["Si"])
    cutoff = 20.0
    case = _case(E_cut_lines_keV=cutoff)
    corrected, record = ls.layer_emission_tables(case, None, [("Si", 0.05)], tables)
    assert record["line_cutoff_keV"] == cutoff
    T = np.asarray(tables["Si"].incident_energy_keV)
    base, new = tables["Si"].scaled_ddcs_mb_sr, corrected["Si"].scaled_ddcs_mb_sr
    np.testing.assert_array_equal(new[T < cutoff], base[T < cutoff])
    above = (T >= cutoff) & (T <= 1000.0)
    assert np.all((base[above] - new[above]).reshape(above.sum(), -1).sum(axis=1) > 0)
