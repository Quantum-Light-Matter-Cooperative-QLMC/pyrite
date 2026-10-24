"""Plotly figures from already computed radiation diagnostic arrays."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _host(values):
    """NumPy copy of a NumPy or CuPy array (CUDA field-device results)."""
    return np.asarray(values.get() if hasattr(values, "get") else values)


def optical_source_figure(band, field):
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Collected spectrum", "Single-electron |Ex|²", "Single-electron arg Ex"),
    )
    fig.add_trace(
        go.Scatter(
            x=_host(band.energy_eV),
            y=_host(band.collected_photons_per_eV_per_pulse),
            name="photons/(eV pulse)",
        ),
        row=1,
        col=1,
    )
    electric = _host(field.electric_xy_v_s_m)[0]
    axis = (np.arange(electric.shape[0]) - electric.shape[0] / 2 + 0.5) * field.dx_m * 1e6
    fig.add_trace(go.Heatmap(x=axis, y=axis, z=abs(electric) ** 2, showscale=False), row=1, col=2)
    fig.add_trace(
        go.Heatmap(x=axis, y=axis, z=np.angle(electric), zmin=-np.pi, zmax=np.pi, showscale=False),
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text="Photon energy (eV)", row=1, col=1)
    fig.update_yaxes(title_text="photons/(eV pulse)", row=1, col=1)
    for col in (2, 3):
        fig.update_xaxes(title_text="x (µm)", row=1, col=col)
        fig.update_yaxes(title_text="y (µm)", row=1, col=col)
    return fig


def optical_scan_figure(scan, fit):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        subplot_titles=("Two-port voltage", "Residuals", "Conditional duration profile"),
    )
    for port in range(2):
        fig.add_trace(
            go.Scatter(
                x=scan.mirror_displacement_m * 1e3,
                y=scan.voltage_v_ports[:, port],
                mode="markers",
                name=f"Port {port + 1} data",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=scan.mirror_displacement_m * 1e3,
                y=fit.predicted_voltage_v_ports[:, port],
                name=f"Port {port + 1} fit",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=scan.mirror_displacement_m * 1e3,
                y=fit.residual_v_ports[:, port],
                name=f"Port {port + 1} residual",
            ),
            row=2,
            col=1,
        )
    fig.update_xaxes(title_text="Mirror displacement (mm); OPD = 2 × displacement", row=2, col=1)
    fig.update_yaxes(title_text="V", row=1, col=1)
    fig.update_yaxes(title_text="V", row=2, col=1)
    profile = fit.metadata["diagnostic_profile"]
    fig.add_trace(
        go.Scatter(x=profile["sigma_fs"], y=profile["delta_chi2"], name="Profile Δχ²"), row=3, col=1
    )
    fig.add_hline(y=3.841, row=3, col=1, line_dash="dash")
    fig.update_xaxes(title_text="Gaussian RMS duration (fs)", type="log", row=3, col=1)
    fig.update_yaxes(title_text="Δχ²", row=3, col=1)
    fig.update_layout(height=800)
    return fig


def dtr_reference_figure(columns):
    fig = go.Figure()
    for key in (
        "tr",
        "pxr",
        "dtr",
        "line",
        "pxr_dtr_cross",
        "tr_line_cross",
        "total",
        "two_beam_transition",
        "two_beam_line",
        "two_beam_cross",
        "two_beam_total",
        "two_beam_minus_born_total",
    ):
        if key in columns:
            fig.add_trace(go.Scatter(x=columns["energy_eV"], y=columns[key], name=key))
    fig.update_layout(
        title="Experimental DTR reference · unverified",
        xaxis_title="Photon energy (eV)",
        yaxis_title="photons/(eV sr electron)",
        yaxis_type="linear",
    )
    return fig


def saved_optical_figure(artifact, kind):
    fig = go.Figure()
    if kind == "Fit":
        for port in range(2):
            fig.add_trace(
                go.Scatter(
                    y=artifact.predicted_voltage_v_ports[:, port],
                    name=f"Port {port + 1} predicted V",
                )
            )
            fig.add_trace(
                go.Scatter(y=artifact.residual_v_ports[:, port], name=f"Port {port + 1} residual V")
            )
        fig.update_layout(
            xaxis_title="Saved scan index",
            yaxis_title="V",
            title=f"Conditional RMS {artifact.rms_duration_fs:g} fs; 95% CI {artifact.confidence_interval_95_fs}; multimodal={artifact.is_multimodal}",
        )
    elif kind == "Interferogram":
        for port in range(2):
            fig.add_trace(
                go.Scatter(
                    x=artifact.mirror_displacement_m * 1e3,
                    y=artifact.reported_voltage_v_ports[:, port],
                    name=f"Port {port + 1}",
                )
            )
        fig.update_layout(xaxis_title="Mirror displacement (mm)", yaxis_title="V")
    else:
        values = np.asarray(artifact.yield_per_electron).reshape(len(artifact.energy_eV), -1)
        for direction in range(values.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=artifact.energy_eV, y=values[:, direction], name=f"Direction {direction + 1}"
                )
            )
        fig.update_layout(xaxis_title="Photon energy (eV)", yaxis_title="photons/(eV sr electron)")
    return fig
