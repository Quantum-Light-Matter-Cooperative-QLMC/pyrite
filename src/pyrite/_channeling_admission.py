"""Data-only admission shared by configuration and production consumers."""

from collections.abc import Iterable, Mapping
from typing import Any, NoReturn


class AxialAdmissionError(ValueError):
    """An axial production run lacks its required physical admission contract."""


def refuse_unverified_axial_run() -> NoReturn:
    """Fail before production work until the physical axial contract exists."""
    raise AxialAdmissionError(
        "channeling_model='axial' cannot run yet: physical matching surface, "
        "incident-normalized photon source and case-specific depth budget "
        "are unvalidated (#623); #624 only provides preliminary configuration"
    )


def refuse_axial_channeling(case: Mapping[str, Any]) -> None:
    """Apply axial admission to raw compatibility mappings as well as cases."""
    record = case.get("channeling")
    if (
        (isinstance(record, Mapping) and record.get("model") == "axial")
        or case.get("channeling_model") == "axial"
        or case.get("channeling_axial") is not None
    ):
        refuse_unverified_axial_run()


def refuse_axial_channeling_cases(cases: Iterable[Mapping[str, Any]]) -> None:
    """Admit a whole batch before any case can read caches, write, or execute."""
    for case in cases:
        refuse_axial_channeling(case)
