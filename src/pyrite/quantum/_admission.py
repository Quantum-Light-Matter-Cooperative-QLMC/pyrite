"""Data-only admission shared by configuration and production consumers."""

from collections.abc import Iterable, Mapping
from typing import Any, NoReturn


class AxialAdmissionError(ValueError):
    """An axial production run lacks its required physical admission contract."""


def refuse_unverified_axial_run() -> NoReturn:
    """Fail before production work for an axial request outside the envelope."""
    raise AxialAdmissionError(
        "channeling_model='axial' cannot run this configuration: only the "
        "repository-calibrated fibre cases are admitted (channeling_axial "
        "version 2; docs/research/physics/axial-first-envelope-886.md, #994). "
        "The version 1 recipe of #624 is configuration only and its physical "
        "contract (#623) is unvalidated"
    )


def refuse_unverified_axial_settings(model: object, selector: object) -> None:
    """Refuse run settings that request axial channeling without the fibre selector.

    Only ``channeling_axial`` version 2 may proceed; each resolved case is then
    admitted or refused against the repository lookup (#994).
    """
    if model == "axial" and not (isinstance(selector, Mapping) and selector.get("version") == 2):
        refuse_unverified_axial_run()


def is_axial_channeling(case: Mapping[str, Any]) -> bool:
    """Whether a case or raw compatibility mapping requests axial channeling."""
    record = case.get("channeling")
    return (
        (isinstance(record, Mapping) and record.get("model") == "axial")
        or case.get("channeling_model") == "axial"
        or case.get("channeling_axial") is not None
    )


def refuse_axial_channeling(case: Mapping[str, Any], consumer: str | None = None) -> None:
    """Refuse every axial case in a consumer outside the admitted envelope.

    Admitted fibre cases run through the standard case runner only
    (``montecarlo.case.admit_axial_channeling``). Replay, recompute, repair,
    trajectory and block consumers call this and refuse before work or writes.
    """
    if not is_axial_channeling(case):
        return
    record = case.get("channeling")
    if isinstance(record, Mapping) and record.get("case_key") is not None:
        raise AxialAdmissionError(
            f"{consumer or 'this consumer'} does not support axial fibre cases; "
            "they run through the standard case runner only (#994)"
        )
    refuse_unverified_axial_run()


def refuse_axial_channeling_cases(cases: Iterable[Mapping[str, Any]]) -> None:
    """Refuse a whole batch before any case can read caches, write, or execute."""
    for case in cases:
        refuse_axial_channeling(case)
