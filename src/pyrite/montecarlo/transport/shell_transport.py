"""Host tables for the opt-in shell soft/hard inelastic transport mode.

``simulate_trajectories(inelastic_model="shell-soft-hard")`` replaces the
continuous collision stopping of each layer by the PENELOPE-like mixed
scheme of :mod:`.shell_partition`: soft losses (``W <= W_c``) stay continuous
and hard losses (``W > W_c``) become discrete events. This module turns the
stopping-closed partition into the per-layer energy tables the CPU cores
read; :mod:`.hard_inelastic` holds the matching Numba kernels.

Closure is exact at the SBETHE nodes by construction. At each node the soft
fraction ``f_s = sigma_s^(1)/(sigma_s^(1)+sigma_h^(1))`` multiplies the
transport's own corrected ``stp.dat`` stopping ``S`` to give the soft
stopping table, and each hard channel's rate is ``S sigma_h,k^(0) /
(sigma_s^(1)+sigma_h^(1))``, so soft plus mean hard loss per path equals
``S`` whatever density the transport table uses.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np

from .hard_inelastic import INELASTIC_MODELS, POSITRON_BRANCH_OFFSET
from .shell_gos import ShellGOSMoments
from .shell_partition import catalog_shell_partition
from .shell_rates import catalog_shell_oscillators

__all__ = [
    "INELASTIC_MODELS",
    "HardChannel",
    "ShellInelasticTables",
    "build_shell_inelastic_tables",
    "hard_event_energy_accounting",
    "validate_shell_cutoff",
]

BRANCH_NAMES = ("distant_longitudinal", "distant_transverse", "close")


@dataclass(frozen=True, slots=True)
class HardChannel:
    """One tabulated hard channel: oscillator, branch and its energies [eV]."""

    oscillator_index: int
    atomic_number: int
    shell_label: str
    branch: str
    ionization_eV: float
    resonance_eV: float
    inner: bool


@dataclass(frozen=True, slots=True)
class ShellInelasticTables:
    """Per-layer soft stopping and hard-event tables for the CPU cores.

    ``soft_stopping_tables`` are prepared SBETHE-style ``(log E_keV, log
    S_soft keV/Angstrom)`` pairs on the transport table's own nodes;
    ``log_energy_keV`` repeats those nodes for the hard and straggling tables.
    Rates are 1/Angstrom, ``soft_straggling_keV2_per_ang`` is
    ``Omega_s^2`` in keV^2/Angstrom. Channel arrays are padded to the widest
    layer; ``n_channels`` bounds every loop.
    """

    cutoff_eV: float
    materials: tuple[str, ...]
    soft_stopping_tables: tuple[tuple[np.ndarray, np.ndarray], ...]
    n_grid: np.ndarray
    log_energy_keV: np.ndarray
    hard_rate_per_ang: np.ndarray
    soft_straggling_keV2_per_ang: np.ndarray
    n_channels: np.ndarray
    channel_rate_per_ang: np.ndarray
    channel_ionization_eV: np.ndarray
    channel_resonance_eV: np.ndarray
    channel_branch: np.ndarray
    channel_code: np.ndarray
    channels: tuple[tuple[HardChannel, ...], ...]
    projectile: str = "electron"

    def core_args(self, hard_keys: np.ndarray) -> tuple:
        """The ``inelastic`` argument tuple the CPU cores unpack."""
        return (
            hard_keys,
            float(self.cutoff_eV),
            self.n_grid,
            self.log_energy_keV,
            self.hard_rate_per_ang,
            self.soft_straggling_keV2_per_ang,
            self.n_channels,
            self.channel_rate_per_ang,
            self.channel_ionization_eV,
            self.channel_resonance_eV,
            self.channel_branch,
            self.channel_code,
        )

    def result_fields(self, hard_W_keV: np.ndarray, hard_channel: np.ndarray) -> dict:
        """Per-row hard-event fields and metadata added to the transport result.

        ``hard_W_keV`` is the transfer on each row that ended in a hard
        collision (or a CUTOFF row whose collision absorbed the primary), 0
        elsewhere; ``hard_channel`` is ``3*oscillator + branch``, -1 without one.
        """
        return {
            "hard_W_keV": hard_W_keV,
            "hard_channel": hard_channel,
            "inelastic": self.metadata(),
        }

    def metadata(self) -> dict:
        """JSON-safe description carried on the transport result."""
        species = {} if self.projectile == "electron" else {"projectile": self.projectile}
        return {
            "model": "shell-soft-hard",
            **species,
            "cutoff_eV": float(self.cutoff_eV),
            "materials": list(self.materials),
            "channels": [
                [
                    {
                        "code": int(self.channel_code[layer, k]),
                        "atomic_number": ch.atomic_number,
                        "shell": ch.shell_label,
                        "branch": ch.branch,
                        "ionization_eV": ch.ionization_eV,
                        "resonance_eV": ch.resonance_eV,
                        "inner": ch.inner,
                    }
                    for k, ch in enumerate(layer_channels)
                ]
                for layer, layer_channels in enumerate(self.channels)
            ],
        }


def _conduction_resonance_eV(key: str) -> float:
    material = catalog_shell_oscillators(key)
    bands = [o.resonance_energy_eV for o in material.oscillators if o.atomic_number == 0]
    if len(bands) != 1:
        raise ValueError(f"{key!r}: shell model has no single conduction-band oscillator")
    return float(bands[0])


def validate_shell_cutoff(materials: Sequence[str], cutoff_eV: float) -> None:
    """Reject a cutoff that is not strictly above every layer's ``W_cb``.

    The closed model's total IMFP is 26-35% below independent full Penn
    values, and 76-85% of its total rate is the conduction-band distant loss
    at exactly ``W_cb``. That discrepancy is benign only when this loss is
    soft, i.e. ``W_c > W_cb``, where it enters transport solely through the
    stopping-closed soft moments (see
    ``docs/validation/beam-transport/penelope-shell-rate-closure.md``).

    Validation: shell-soft-hard-transport
    """
    if isinstance(cutoff_eV, bool) or not np.isfinite(cutoff_eV) or cutoff_eV <= 0.0:
        raise ValueError("inelastic_cutoff_eV must be finite and positive")
    for key in materials:
        w_cb = _conduction_resonance_eV(key)
        if not cutoff_eV > w_cb:
            raise ValueError(
                f"inelastic_cutoff_eV={cutoff_eV:g} eV must exceed the {key!r} "
                f"conduction-band resonance W_cb={w_cb:g} eV: below it the "
                "unvalidated total IMFP would drive hard events"
            )


@cache
def _node_moments(key: str, energy_eV: float, cutoff_eV: float, projectile: str = "electron"):
    """Soft/hard partition summary at one stopping node, cached per process."""
    part = catalog_shell_partition(key, energy_eV, cutoff_eV, projectile=projectile)
    hard: ShellGOSMoments = part.hard
    channels = np.stack([hard.distant_longitudinal, hard.distant_transverse, hard.close], axis=1)[
        :, :, 0
    ]
    return (
        float(part.soft.total[1]),
        float(part.soft.total[2]),
        float(hard.total[1]),
        channels.copy(),
        part.closure.inner.copy(),
    )


def _node_range(log_energy: np.ndarray, E_min_keV: float, E_max_keV: float) -> tuple[int, int]:
    n = log_energy.size
    lo = int(np.searchsorted(log_energy, np.log(E_min_keV), side="right")) - 1
    hi = int(np.searchsorted(log_energy, np.log(E_max_keV), side="left"))
    lo = min(max(lo, 0), n - 2)
    hi = max(min(hi, n - 1), lo + 1)
    return lo, hi


def build_shell_inelastic_tables(
    materials: str | Sequence[str],
    cutoff_eV: float,
    stopping_tables: Sequence[tuple[np.ndarray, np.ndarray]],
    E_min_keV: float,
    E_max_keV: float,
    *,
    projectile: str = "electron",
) -> ShellInelasticTables:
    """Tabulate the shell soft/hard partition on each layer's stopping nodes.

    ``stopping_tables`` are the prepared per-layer SBETHE tables the
    transport would otherwise run with; ``materials`` name the catalog key of
    each layer's shell model. Only the nodes bracketing
    ``[E_min_keV, E_max_keV]`` are evaluated.

    Source: PENELOPE-2024 Eqs. 3.124 and 4.44-4.47 via
    :func:`~.shell_partition.catalog_shell_partition`. Assumptions: the
    layer's transport table and the catalog table share their energy nodes
    (checked); rates and ``Omega_s^2`` are linear in ``log E`` between nodes,
    the soft stopping log-log like the full table. Limits: ``W_c`` above every
    channel endpoint gives ``f_s = 1`` exactly, i.e. the unchanged continuous
    table and no hard events.

    ``projectile="positron"`` closes the Bhabha partition on the positron
    SBETHE nodes (``stopping_tables`` must be the positron tables) and tags
    ``channel_branch`` with ``POSITRON_BRANCH_OFFSET`` for the kernels;
    ``channel_code`` keeps the species-independent ``3*oscillator + branch``.

    Validation: shell-soft-hard-transport, bhabha-close
    """
    from ...xsgen.sbethe.catalog import resolve_catalog_table

    if isinstance(materials, str):
        materials = (materials,)
    materials = tuple(str(key) for key in materials)
    if len(materials) != len(stopping_tables):
        raise ValueError("inelastic_materials must name one catalog key per layer")
    cutoff_eV = float(cutoff_eV)
    validate_shell_cutoff(materials, cutoff_eV)
    n_layers = len(materials)
    soft_tables = []
    layer_rows = []
    for key, (log_e, log_s) in zip(materials, stopping_tables, strict=True):
        energy_eV = np.asarray(
            resolve_catalog_table(key, projectile=projectile).arrays()["stopping_energy_eV"],
            float,
        )
        if energy_eV.shape != log_e.shape or not np.allclose(
            np.log(energy_eV * 1e-3), log_e, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"layer stopping table does not share the {key!r} catalog energy nodes"
            )
        lo, hi = _node_range(log_e, E_min_keV, E_max_keV)
        nodes = range(lo, hi + 1)
        summary = [_node_moments(key, float(energy_eV[i]), cutoff_eV, projectile) for i in nodes]
        soft1 = np.array([s[0] for s in summary])
        soft2 = np.array([s[1] for s in summary])
        hard1 = np.array([s[2] for s in summary])
        channels = np.stack([s[3] for s in summary], axis=-1)  # (n_osc, 3, n_nodes)
        total1 = soft1 + hard1
        stopping_eV_per_ang = np.exp(log_s[lo : hi + 1]) * 1e3
        soft_fraction = soft1 / total1
        if not np.all(soft_fraction > 0.0):
            raise ValueError(f"{key!r}: soft stopping vanishes at W_c={cutoff_eV:g} eV")
        soft_log_s = log_s[lo : hi + 1] + np.log(soft_fraction)
        soft_tables.append((log_e[lo : hi + 1].copy(), soft_log_s))
        per_ang = stopping_eV_per_ang / total1
        channel_rates = channels * per_ang  # 1/Angstrom
        omega2 = soft2 * per_ang * 1e-6  # keV^2/Angstrom
        material = catalog_shell_oscillators(key)
        inner = summary[0][4]
        active = [
            (i, b)
            for i in range(channel_rates.shape[0])
            for b in range(3)
            if np.any(channel_rates[i, b] > 0.0)
        ]
        layer_channels = tuple(
            HardChannel(
                i,
                material.oscillators[i].atomic_number,
                material.oscillators[i].label,
                BRANCH_NAMES[b],
                float(material.oscillators[i].ionization_energy_eV),
                float(material.oscillators[i].resonance_energy_eV),
                bool(inner[i]),
            )
            for i, b in active
        )
        rates = np.array([channel_rates[i, b] for i, b in active]).reshape(
            len(active), log_e[lo : hi + 1].size
        )
        codes = [3 * i + b for i, b in active]
        layer_rows.append((log_e[lo : hi + 1], rates, omega2, layer_channels, codes))

    n_grid = np.array([row[0].size for row in layer_rows], dtype=np.int32)
    width = int(n_grid.max())
    n_channels = np.array([len(row[3]) for row in layer_rows], dtype=np.int32)
    max_ch = max(1, int(n_channels.max()))
    log_energy = np.zeros((n_layers, width))
    hard_rate = np.zeros((n_layers, width))
    omega2_all = np.zeros((n_layers, width))
    channel_rate = np.zeros((n_layers, max_ch, width))
    ionization = np.zeros((n_layers, max_ch))
    resonance = np.zeros((n_layers, max_ch))
    branch = np.zeros((n_layers, max_ch), dtype=np.int64)
    code = np.full((n_layers, max_ch), -1, dtype=np.int16)
    for layer, (log_e, rates, omega2, layer_channels, codes) in enumerate(layer_rows):
        n = log_e.size
        log_energy[layer, :n] = log_e
        # Pad by repeating the last node so a clamped read stays in range.
        log_energy[layer, n:] = log_e[-1]
        omega2_all[layer, :n] = omega2
        m = len(layer_channels)
        if m:
            channel_rate[layer, :m, :n] = rates
            hard_rate[layer, :n] = rates.sum(axis=0)
        for k, ch in enumerate(layer_channels):
            ionization[layer, k] = ch.ionization_eV
            resonance[layer, k] = ch.resonance_eV
            branch[layer, k] = BRANCH_NAMES.index(ch.branch) + (
                POSITRON_BRANCH_OFFSET if projectile == "positron" else 0
            )
            code[layer, k] = codes[k]
    return ShellInelasticTables(
        cutoff_eV,
        materials,
        tuple(soft_tables),
        n_grid,
        log_energy,
        hard_rate,
        omega2_all,
        n_channels,
        channel_rate,
        ionization,
        resonance,
        branch,
        code,
        tuple(row[3] for row in layer_rows),
        projectile,
    )


def hard_event_energy_accounting(result: dict, *, production_threshold_eV: float = 0.0) -> dict:
    """Split every recorded hard transfer into deposit, secondary and binding.

    Follows ``shell_sampling.sample_shell_hard_loss``: an EEDL-substituted
    inner shell reserves its binding ``U`` for relaxation (vacancy handoff
    to #94) and emits ``W - U``; an outer shell or the conduction band emits
    the free-electron proxy ``W``. A secondary below
    ``production_threshold_eV`` is deposited locally. Secondaries are not
    transported (#94). Returns per-hard-row arrays in keV plus the vacancy
    flag; ``deposit + secondary + binding == W`` row by row.

    Validation: shell-soft-hard-transport
    """
    meta = result.get("inelastic")
    if meta is None or meta.get("model") != "shell-soft-hard":
        raise ValueError("result was not transported with inelastic_model='shell-soft-hard'")
    code = np.asarray(result["hard_channel"])
    rows = np.flatnonzero(code >= 0)
    layer = np.asarray(result["layer"])[rows]
    transfer = np.asarray(result["hard_W_keV"])[rows]
    lookup = [{c["code"]: c for c in channels} for channels in meta["channels"]]
    binding = np.zeros(rows.size)
    vacancy = np.zeros(rows.size, dtype=bool)
    for j, (layer_j, code_j) in enumerate(zip(layer, code[rows], strict=True)):
        channel = lookup[int(layer_j)][int(code_j)]
        if channel["inner"]:
            vacancy[j] = True
            binding[j] = channel["ionization_eV"] * 1e-3
    kinetic = np.where(vacancy, np.maximum(0.0, transfer - binding), transfer)
    binding = np.where(vacancy, transfer - kinetic, 0.0)
    emitted = (kinetic > 0.0) & (kinetic * 1e3 >= production_threshold_eV)
    return {
        "row": rows,
        "transfer_keV": transfer,
        "secondary_keV": np.where(emitted, kinetic, 0.0),
        "deposit_keV": np.where(emitted, 0.0, kinetic),
        "binding_keV": binding,
        "vacancy": vacancy,
    }
