"""One-way handoff from the planar channeling zone to native transport (#506).

ADR-0018 D4-A: the zone owns each electron from crystal entry until it
dechannels; the dechanneled electron then continues as an ordinary launch
state of the native cores, behind an ``EVENT_DECHANNEL`` row. Selection,
gates and identity live in ``campaign.channeling_cases`` and
:class:`~pyrite.montecarlo.case.Case`; this module owns the physics-generation
marker the identity carries.

Import-light at module scope: consumers that only refuse channeling cases
import :data:`CHANNELING_MODEL` without loading the solvers.
"""

from __future__ import annotations

#: Physics-generation marker of the planar zone and its handoff. It joins
#: ``case_content_key`` and run provenance only for cases with a ``channeling``
#: record, so channeling-off identities are unchanged.
CHANNELING_MODEL = "planar-density-matrix-handoff-v1"

__all__ = ["CHANNELING_MODEL"]
