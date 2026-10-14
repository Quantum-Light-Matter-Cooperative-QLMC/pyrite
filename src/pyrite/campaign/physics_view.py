"""Campaign spelling of shared resolved-case provenance (ADR-0017).

The runtime helper lives below campaign so transport/replay can record and
check it without importing a campaign driver. Canonical cache hashes remain
owned by ``campaign.profiles`` and are unchanged by this descriptive view.
"""

from .._physics_provenance import resolved_physics_view as resolved_physics_view
