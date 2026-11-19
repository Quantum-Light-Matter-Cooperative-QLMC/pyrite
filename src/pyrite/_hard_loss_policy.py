"""Versioned identity of the shell hard-loss CDF inversion (#916).

Shell soft/hard inelastic transport samples each hard collision's energy loss
by inverting the channel's restricted loss CDF
(``montecarlo/transport/hard_inelastic._sample_hard_transfer_eV`` and its
CUDA twin). Safeguarded Newton iteration replaced bisection to adjacent
doubles in #916. Both reach the same root to round-off, but the sampled losses
differ in their last bits, so fixed-seed shell trajectories are not
bit-identical across the change. Shell cases' dataset identity, content key and
provenance carry :data:`HARD_LOSS_SAMPLING_MODEL`. Continuous-stopping cases
never carry it, so their digests are unchanged.
"""

HARD_LOSS_SAMPLING_MODEL = "safeguarded-newton-v1"


def hard_loss_markers(inelastic_model) -> dict[str, str]:
    """``{"hard_loss_sampling_model": ...}`` for a shell case, else empty."""
    if inelastic_model != "shell-soft-hard":
        return {}
    return {"hard_loss_sampling_model": HARD_LOSS_SAMPLING_MODEL}
