"""Versioned identity of the cascade RNG addressing (#923).

Secondary cascades (``secondary_threshold_eV``, with any pair conversion or
positron transport) draw every descendant stream from lineage keys
(``montecarlo/transport/lineage.py``). Cases record the scheme as
``cascade_rng_model``; noncascade cases never carry it, so their payloads and
digests are unchanged.

A cascade case without the marker was recorded under the former dense-track
addressing. That realization cannot be reproduced, so such records are refused
rather than replayed under the new scheme.
"""

CASCADE_RNG_MODEL = "lineage-addressed-v1"


def check_cascade_rng_model(case) -> None:
    """Refuse a cascade case whose RNG marker is missing or unknown."""
    marker = case.get("cascade_rng_model")
    if case.get("secondary_threshold_eV") is None:
        if marker is not None:
            raise ValueError("cascade_rng_model requires secondary_threshold_eV")
        return
    if marker is None:
        raise ValueError(
            "this secondary-cascade case has no cascade_rng_model: it was recorded "
            "under the dense-track cascade RNG replaced in #923, whose fixed-seed "
            "realization cannot be replayed. Re-run it from its campaign, which "
            f"records cascade_rng_model={CASCADE_RNG_MODEL!r}."
        )
    if marker != CASCADE_RNG_MODEL:
        raise ValueError(
            f"unsupported cascade_rng_model {marker!r}; this version runs {CASCADE_RNG_MODEL!r}"
        )
