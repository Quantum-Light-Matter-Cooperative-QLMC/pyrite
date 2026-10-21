"""Stored-array population and ensemble metric views for channeling apps."""

import numpy as np


def population_figure(depths, populations, survival, sample, *, conditional=False, labels=None):
    """Plot incident or survivor-conditioned populations without solver callbacks."""
    import matplotlib.pyplot as plt

    values = np.asarray(populations)
    represented = values.sum(axis=1)
    if conditional:
        values = np.divide(
            values, survival[:, None], out=np.zeros_like(values), where=survival[:, None] > 0
        )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    index = np.arange(values.shape[1])
    axes[0].bar(index, values[sample])
    axes[0].set(
        title=f"Instantaneous populations · {depths[sample] / 1e4:.4g} µm",
        xlabel="State / level index",
        ylabel="Per survivor" if conditional else "Per incident electron",
    )
    if labels is not None and len(labels) <= 16:
        axes[0].set_xticks(index, labels, rotation=45, ha="right")
    image = axes[1].pcolormesh(index, depths / 1e4, values, shading="nearest")
    axes[1].set(
        title="State populations versus depth", xlabel="State / level index", ylabel="Depth (µm)"
    )
    fig.colorbar(image, ax=axes[1])
    for data, label in (
        (survival, "surviving"),
        (represented, "represented"),
        (survival - represented, "survival minus displayed weights (signed)"),
    ):
        axes[2].plot(depths / 1e4, data, label=label)
    axes[2].set(xlabel="Depth (µm)", ylabel="Per incident electron", title="Projection accounting")
    axes[2].legend(fontsize=8)
    return fig


def metric_figure(depths, metrics):
    """Draw empirical purity, squared coherence, and exact conditional H moments."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    z = depths / 1e4
    axes[0].errorbar(z, metrics.purity, yerr=metrics.purity_probe_sem, label="purity", capsize=2)
    axes[0].plot(z, metrics.coherence_squared, label="squared off-diagonal norm")
    axes[0].set(title=f"Empirical ensemble · {metrics.basis}", ylabel="Dimensionless")
    axes[0].legend(fontsize=8)
    axes[1].plot(z, metrics.energy_mean_eV)
    axes[1].set(title="Conditional transverse energy", ylabel="Mean H⊥ (eV)")
    axes[2].plot(z, metrics.energy_std_eV)
    axes[2].set(title="Conditional transverse-energy spread", ylabel="Standard deviation (eV)")
    for ax in axes:
        ax.set_xlabel("Depth (µm)")
    return fig


def axial_accounting_figure(result):
    """Recorded grid survival, numerical loss, jumps and local OAM moments."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    z = result.depths_ang / 1e4
    axes[0].plot(z, result.surviving, label="grid survival")
    if result.numerical_sink is not None:
        axes[0].plot(z, result.numerical_sink, label="numerical grid sink")
        axes[0].plot(
            z, 1 - result.surviving - result.numerical_sink, label="unresolved balance residual"
        )
    axes[0].set(
        ylabel="Per incident electron",
        title="Diagnostic accounting · no physical escape instrument",
    )
    axes[0].legend(fontsize=8)
    if result.jump_history:
        for name, counts in result.jump_history.items():
            axes[1].plot(z, counts / result.config.n_electrons, label=name)
        axes[1].legend()
    axes[1].set(ylabel="Cumulative jumps per incident electron", title="Quantum jumps")
    if result.oam_populations is not None:
        populations = result.oam_populations
        weight = populations.sum(axis=1)
        denom = np.where(weight > 0, weight, np.nan)
        mean = (populations * result.oam_ells).sum(axis=1) / denom
        spread = np.sqrt(
            np.maximum((populations * result.oam_ells**2).sum(axis=1) / denom - mean**2, 0)
        )
        axes[2].plot(z, mean, label="mean")
        axes[2].plot(z, spread, label="standard deviation")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "Enable local OAM recording", ha="center")
    axes[2].set(ylabel="Lz / ħ", title="Conditioned on represented disk-local ℓ modes")
    for ax in axes:
        ax.set_xlabel("Depth (µm)")
    return fig
