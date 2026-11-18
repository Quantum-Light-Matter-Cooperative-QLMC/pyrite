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
    assert ls.skip_reason(dict(SI, bremsstrahlung_model="eedl"), None).startswith("no BremsLib")
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


def test_eedl_crystal_warns_and_bremslib_or_amorphous_do_not():
    with pytest.warns(ls.LatticeShareUncorrectedWarning):
        ls.warn_if_eedl_double_counts(dict(SI, bremsstrahlung_model="eedl"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ls.warn_if_eedl_double_counts(_case())
        ls.warn_if_eedl_double_counts(dict(SI, crystal=None, bremsstrahlung_model="eedl"))


def test_uncorrected_layers_pass_their_tables_through():
    tables = {"Si": object()}
    out, record = ls.layer_emission_tables(_case(hkl_list=[]), None, [("Si", 0.05)], tables)
    assert out is tables
    assert record == dict(layer=None, corrected=False, reason="no scored reflections")


def test_selected_case_without_its_record_refuses_rather_than_skips():
    case = _case(hkl_list=None, reflection_selection=dict(shell_width_invang=1.0))
    with pytest.raises(RuntimeError, match="without its selection record"):
        ls.scored_reflections(case, None)
