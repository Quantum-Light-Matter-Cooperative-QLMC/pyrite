"""Synthetic BremsLib tables in the ``pyrite.xsgen.bremslib`` stored layout.

CI holds neither the BremsLib library nor its released tables, so the
direction-resolved spectrum is tested against an analytic stand-in: a
Sommerfeld-like ``sin^2(theta) / (1 - beta cos(theta))^4`` shape plus an
isotropic floor, deliberately left unnormalized so staging has to normalize it
against the SDCS.
"""

import numpy as np

K_OVER_T1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.0])
SYNTHETIC_T1_MEV = np.array([1.0e-2, 2.0e-2, 5.0e-2, 1.0e-1, 2.0e-1])


def synthetic_scaled_sdcs(t1_MeV, reduced):
    """Scaled SDCS ``chi`` in mb: smooth, positive, falling with ``k/T1``."""
    return 10.0 * (1.0 - 0.5 * np.asarray(reduced)) * np.asarray(t1_MeV) ** -0.1


def synthetic_shape(t1_MeV, theta_rad):
    """Unnormalized angular shape, forward-peaking as ``T1`` rises."""
    gamma = 1.0 + np.asarray(t1_MeV) / 0.51099895
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    theta = np.asarray(theta_rad)
    return np.sin(theta) ** 2 / (1.0 - beta * np.cos(theta)) ** 4 + 0.2


def synthetic_bremslib_arrays(t1_MeV=SYNTHETIC_T1_MEV, theta_step_deg=1.0):
    """Return one element's arrays as :func:`pyrite.xsgen.bremslib.build_table` stores them."""
    t1 = np.asarray(t1_MeV, dtype=float)
    theta_deg = np.arange(0.0, 180.0 + 0.5 * theta_step_deg, theta_step_deg)
    n_nodes = t1.size * K_OVER_T1.size
    sdcs = synthetic_scaled_sdcs(t1[:, None], K_OVER_T1[None, :])
    ddcs = []
    for row in range(t1.size):
        for column in range(K_OVER_T1.size):
            # The 3.7 x (1 + k/T1) scale is arbitrary: staging must remove it.
            ddcs.append(
                3.7 * (1.0 + K_OVER_T1[column]) * synthetic_shape(t1[row], np.radians(theta_deg))
            )
    top = np.minimum(K_OVER_T1[-1], 1.0 - 50.0e-6 / t1)
    node_k = (np.tile(K_OVER_T1, t1.size).reshape(t1.size, -1) * t1[:, None]).copy()
    node_k[:, -1] = top * t1
    return {
        "t1_MeV": t1,
        "k_over_t1": K_OVER_T1.copy(),
        "sdcs_mb": sdcs,
        "sdcs_point_finite": np.ones_like(sdcs, dtype=np.float32),
        "node_t1_index": np.repeat(np.arange(t1.size), K_OVER_T1.size).astype(np.int32),
        "node_k_index": np.tile(np.arange(K_OVER_T1.size), t1.size).astype(np.int32),
        "node_k_MeV": node_k.reshape(-1),
        "node_offset": (np.arange(n_nodes + 1) * theta_deg.size).astype(np.int64),
        "node_angular_integral_mb": sdcs.reshape(-1),
        "node_vendor_integral_mb": sdcs.reshape(-1),
        "node_finite_nucleus": np.zeros(n_nodes, dtype=bool),
        "theta_deg": np.tile(theta_deg, n_nodes),
        "ddcs_mb_sr": np.concatenate(ddcs).astype(np.float32),
        "ddcs_point_finite": np.ones(n_nodes * theta_deg.size, dtype=np.float32),
    }


def extend_synthetic_arrays(arrays, extension_t1_MeV, *, tip_columns=2):
    """Append extension rows as the library holds them above 30 MeV.

    Each new row keeps the synthetic SDCS for its leading columns, NaN for the
    last ``tip_columns``, and only its ``k = 0`` DDCS node.
    """
    t1_ext = np.asarray(extension_t1_MeV, dtype=float)
    complete = arrays["t1_MeV"].size
    theta_deg = arrays["theta_deg"][: int(arrays["node_offset"][1])]
    sdcs_ext = synthetic_scaled_sdcs(t1_ext[:, None], K_OVER_T1[None, :])
    sdcs_ext[:, K_OVER_T1.size - tip_columns :] = np.nan
    ddcs_ext = [
        (3.7 * synthetic_shape(t1, np.radians(theta_deg))).astype(np.float32) for t1 in t1_ext
    ]
    offset_end = int(arrays["node_offset"][-1])
    return {
        **arrays,
        "t1_MeV": np.r_[arrays["t1_MeV"], t1_ext],
        "complete_rows": np.asarray(complete, dtype=np.int64),
        "sdcs_mb": np.vstack([arrays["sdcs_mb"], sdcs_ext]),
        "sdcs_point_finite": np.vstack(
            [arrays["sdcs_point_finite"], np.ones_like(sdcs_ext, dtype=np.float32)]
        ),
        "node_t1_index": np.r_[arrays["node_t1_index"], complete + np.arange(t1_ext.size)].astype(
            np.int32
        ),
        "node_k_index": np.r_[arrays["node_k_index"], np.zeros(t1_ext.size)].astype(np.int32),
        "node_k_MeV": np.r_[arrays["node_k_MeV"], np.zeros(t1_ext.size)],
        "node_offset": np.r_[
            arrays["node_offset"], offset_end + theta_deg.size * np.arange(1, t1_ext.size + 1)
        ].astype(np.int64),
        "node_angular_integral_mb": np.r_[arrays["node_angular_integral_mb"], sdcs_ext[:, 0]],
        "node_vendor_integral_mb": np.r_[arrays["node_vendor_integral_mb"], sdcs_ext[:, 0]],
        "node_finite_nucleus": np.r_[arrays["node_finite_nucleus"], np.zeros(t1_ext.size, bool)],
        "theta_deg": np.r_[arrays["theta_deg"], np.tile(theta_deg, t1_ext.size)],
        "ddcs_mb_sr": np.r_[arrays["ddcs_mb_sr"], *ddcs_ext],
        "ddcs_point_finite": np.r_[
            arrays["ddcs_point_finite"], np.ones(theta_deg.size * t1_ext.size, np.float32)
        ],
    }
