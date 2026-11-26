"""Shared ``--FIELD`` options for profile and preset Numerics/Physics setters.

One declaration serves ``pyrite profile numerics|physics set`` and
``pyrite numerics|physics create|set``; :func:`field_params` copies the
options a given owner accepts. The API partition
(``pyrite._numerics.PHYSICS_KEYS``) decides whether a field is Physics or
Numerics.
"""

from copy import copy

import click
from click.shell_completion import CompletionItem

from pyrite._numerics import CHANNELING_KEYS, PHYSICS_KEYS
from pyrite._off_settings import DEPRECATED_OFF, NONE
from pyrite.cli._deprecations import canonical_option, removed_option, warn_off_value
from pyrite.cli._unit_params import unit_float
from pyrite.cli.commands import _profile_shared

_ENERGY_MODEL_VALUES = ("frozen", "midpoint")
_INELASTIC_MODEL_VALUES = ("auto", "continuous", "shell-soft-hard")
_ELASTIC_MODEL_VALUES = ("mott", "elsepa")
_BREMSSTRAHLUNG_MODEL_VALUES = ("auto", "eedl", "bremslib")
_RADIATIVE_MODEL_VALUES = ("auto", "uncoupled", "bremslib-soft-hard")
_PAIR_PRODUCTION_MODEL_VALUES = ("penelope-2024",)
_TRANSITION_RADIATION_VALUES = ("history", "bunch", "two-beam")
_ATOMIC_ELECTRON_DEFLECTION_VALUES = ("kawrakow", "none")
_SHELL_IONIZATION_MODEL_VALUES = ("eedl", "bote-salvat")
_CXR_SOURCE_VALUES = ("mc", "none")
_MOSAIC_ROUTE_VALUES = ("analytic", "mc", "mc-sampled")


class NoneOr(click.ParamType):
    """``none`` (turn the setting off) or a value of ``base`` (#1242).

    Non-boolean optional settings follow the "off follows type" convention
    (docs/repo-design/profile-settings.md): ``none`` is stored, so it overrides an
    attached preset. ``off`` is a deprecated alias that warns on stderr.
    """

    def __init__(self, base: click.ParamType, metavar: str | None = None) -> None:
        self.base = base
        self.name = f"none|{base.name}"
        self._metavar = metavar

    def get_metavar(self, param: click.Parameter, ctx: click.Context) -> str | None:
        if self._metavar is not None:
            return f"none|{self._metavar}"
        choices = getattr(self.base, "choices", None)
        return None if choices is None else f"[none|{'|'.join(map(str, choices))}]"

    def convert(self, value, param, ctx):
        if value == NONE:
            return NONE
        if value == DEPRECATED_OFF:
            warn_off_value(f"--{(param.name if param else '').replace('_', '-')}")
            return NONE
        try:
            return self.base.convert(value, param, ctx)
        except click.BadParameter as exc:
            self.fail(f"{exc.message}; or 'none' to turn it off", param, ctx)

    def shell_complete(self, ctx, param, incomplete):
        items = [CompletionItem(NONE)] if NONE.startswith(incomplete) else []
        return items + self.base.shell_complete(ctx, param, incomplete)


#: Option spelling -> catalog/API field name.
FIELD_NAMES = {
    "cxr-histories": "n_electrons",
    "brem-histories": "n_electrons_brem",
    "reflection-families": "n_families",
    "maximum-reflections": "max_reflections",
    "mosaic-nodes": "mosaic_nodes",
    "mosaic-route": "mosaic_route",
    "line-yield-tolerance": "line_yield_tolerance",
    "straggling": "straggling",
    "energy-model": "energy_model",
    "maximum-fractional-energy-loss": "max_dE_frac",
    "inelastic-model": "inelastic_model",
    "inelastic-cutoff-ev": "inelastic_cutoff_eV",
    "secondary-threshold-ev": "secondary_threshold_eV",
    "elastic-model": "elastic_model",
    "transport-mode": "transport_mode",
    "mixed-c1": "mixed_c1",
    "mixed-c2": "mixed_c2",
    "mixed-maximum-step-um": "mixed_max_step_um",
    "bremsstrahlung-model": "bremsstrahlung_model",
    "radiative-model": "radiative_model",
    "radiative-cutoff-ev": "radiative_cutoff_eV",
    "pair-production-model": "pair_production_model",
    "positron-transport": "positron_transport",
    "atomic-electron-deflection": "atomic_electron_deflection",
    "channeling-model": "channeling_model",
    "channeling-axial": "channeling_axial",
    "channeling-admission": "channeling_admission",
    "transition-radiation": "transition_radiation",
    "shell-ionization-model": "shell_ionization_model",
    "cxr-source": "cxr_source",
    "channeling-plane": "channeling_plane",
    "channeling-basis-size": "channeling_basis_size",
    "channeling-bloch-cells": "channeling_bloch_cells",
    "channeling-dz-ang": "channeling_dz_ang",
    "channeling-zone-max-ang": "channeling_zone_max_ang",
    "math": "math",
}


def field_names(*, physics: bool, channeling: bool = True) -> dict[str, str]:
    """Option spelling -> field for the Physics or the Numerics owner.

    ``channeling=False`` drops the ``[channeling]`` table fields, which only
    profiles carry (#1017); named presets have no channeling section.
    """
    return {
        flag: key
        for flag, key in FIELD_NAMES.items()
        if (key in PHYSICS_KEYS) == physics and (channeling or key not in CHANNELING_KEYS)
    }


#: Channeling option spellings, in declaration order.
CHANNELING_FLAGS = tuple(flag for flag, key in FIELD_NAMES.items() if key in CHANNELING_KEYS)


class ChannelingSectionCommand(click.Command):
    """Command whose ``--channeling-*`` options render under their own heading.

    The flags write the profile's ``[channeling]`` table (#1017), so help lists
    them apart from the Physics/Numerics fields.
    """

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Render ordinary options, then the channeling-table options."""
        flags = {f"--{flag}" for flag in CHANNELING_FLAGS}
        rows: list[tuple[str, str]] = []
        channeling: list[tuple[str, str]] = []
        for param in self.get_params(ctx):
            record = param.get_help_record(ctx)
            if record is not None:
                target = channeling if flags.intersection(getattr(param, "opts", ())) else rows
                target.append(record)
        if rows:
            with formatter.section("Options"):
                formatter.write_dl(rows)
        if channeling:
            with formatter.section("Channeling options (profile [channeling] table)"):
                formatter.write_dl(channeling)


@removed_option("--cxr-trials", "--cxr-histories")
@removed_option("--cxr-electrons", "--cxr-histories")
@removed_option("--brem-trials", "--brem-histories")
@removed_option("--brem-electrons", "--brem-histories")
@canonical_option(
    "--cxr-histories",
    "cxr_histories",
    type=click.IntRange(min=1),
    metavar="N",
    help=(
        "Fixed Monte Carlo electron histories used to estimate the CXR spectrum; setting it "
        "selects fixed counts. Adaptive sampling (the default) is restored by "
        "'pyrite profile numerics adaptive-sampling enable PROFILE'."
    ),
)
@canonical_option(
    "--brem-histories",
    "brem_histories",
    type=click.IntRange(min=1),
    metavar="N",
    help=(
        "Fixed Monte Carlo electron histories used to estimate bremsstrahlung; setting it "
        "selects fixed counts. Adaptive sampling (the default) is restored by "
        "'pyrite profile numerics adaptive-sampling enable PROFILE'."
    ),
)
@click.option(
    "--reflection-families",
    type=click.IntRange(min=1),
    metavar="N",
    help="Ranked reflection families to resolve.",
)
@click.option(
    "--maximum-reflections",
    type=click.IntRange(min=1),
    metavar="N",
    help="Cap resolved reflections after family expansion.",
)
@click.option(
    "--mosaic-nodes",
    type=click.IntRange(min=1),
    metavar="N",
    help="Gauss-Hermite nodes per mosaic tilt axis.",
)
@click.option(
    "--mosaic-route",
    type=click.Choice(_MOSAIC_ROUTE_VALUES),
    help="Mosaic broadening route.",
)
@click.option(
    "--line-yield-tolerance",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True, max_open=True),
    metavar="FRACTION",
    help="Opt-in measured reflection cutoff: stop |g| shells once the estimated "
    "omitted incoherent line yield is at most FRACTION of the kept yield.",
)
@click.option(
    "--straggling/--no-straggling",
    default=None,
    help="Collision energy-loss straggling: restricted soft loss in shell mode (default on), Urban fluctuations with continuous stopping (choose explicitly).",
)
@click.option(
    "--energy-model", type=click.Choice(_ENERGY_MODEL_VALUES), help="Transport clock model."
)
@click.option(
    "--maximum-fractional-energy-loss",
    type=click.FloatRange(min=0.0),
    metavar="FRACTION",
    help="Cap one row's fractional mean energy loss; positive values require midpoint.",
)
@click.option(
    "--inelastic-model",
    type=click.Choice(_INELASTIC_MODEL_VALUES),
    help=(
        "Collision energy-loss scheme: shell-soft-hard (default; midpoint, ungrooved, "
        "shell data for every layer), continuous stopping (explicit approximation), or auto "
        "(legacy conditional shell selection with warned continuous fallback)."
    ),
)
@click.option(
    "--inelastic-cutoff-ev",
    type=unit_float(click.FloatRange(min=0.0, min_open=True), "eV", "energy"),
    metavar="EV",
    help=(
        "Soft/hard energy-loss cutoff W_c in eV (default 50 for shell-soft-hard or auto); must exceed each "
        "material's conduction-band resonance (Si 16.7, SiO2 22, MoS2 23 eV)."
    ),
)
@click.option(
    "--secondary-threshold-ev",
    type=NoneOr(unit_float(click.FloatRange(min=0.0, min_open=True), "eV", "energy"), "EV"),
    help=(
        "Transport hard-collision secondaries above this energy in eV (production cut and "
        "tracking cutoff; requires shell-soft-hard and SBETHE coverage, >= 1000 eV). "
        "none turns secondary transport off (needs --pair-production-model none)."
    ),
)
@click.option(
    "--elastic-model",
    type=click.Choice(_ELASTIC_MODEL_VALUES),
    help=(
        "Elastic scattering: elsepa (default) full differential cross sections (needs "
        "'pyrite tables fetch elsepa'; bypasses the transport LUT), or the historical "
        "mott screened-Rutherford angles."
    ),
)
@click.option(
    "--bremsstrahlung-model",
    type=click.Choice(_BREMSSTRAHLUNG_MODEL_VALUES),
    help=(
        "Continuum bremsstrahlung: bremslib (default) requires the released BremsLib tables "
        "('pyrite tables fetch bremslib') and uses their angular model; eedl is the fetched "
        "EEDL continuum with an isotropic photon angle (pair with --radiative-model "
        "uncoupled); auto is the legacy table-checked BremsLib selection."
    ),
)
@click.option(
    "--radiative-model",
    type=click.Choice(_RADIATIVE_MODEL_VALUES),
    help=(
        "Radiative energy loss: bremslib-soft-hard (default) uses soft loss plus sampled "
        "hard photons and requires midpoint energy, BremsLib and an ungrooved target; "
        "uncoupled (explicit approximation) scores after collision-only transport; auto "
        "(legacy) couples where supported. Unsupported scenes must pass uncoupled."
    ),
)
@click.option(
    "--radiative-cutoff-ev",
    type=unit_float(click.FloatRange(min=0.0, min_open=True), "eV", "energy"),
    metavar="EV",
    help=(
        "Hard-photon cutoff k_c in eV for bremslib-soft-hard or auto (default 1000); must not "
        "exceed the continuum electron cutoff (1000 eV by default)."
    ),
)
@click.option(
    "--pair-production-model",
    type=NoneOr(click.Choice(_PAIR_PRODUCTION_MODEL_VALUES)),
    help=(
        "Convert coupled hard photons above 1.022 MeV to e-/e+ pairs (opt-in). Requires "
        "--secondary-threshold-ev and coupled BremsLib transport; pair electrons are "
        "transported as secondaries, positrons are recorded unless --positron-transport. "
        "none turns pair production off."
    ),
)
@click.option(
    "--positron-transport/--no-positron-transport",
    default=None,
    help=(
        "Transport pair positrons above the secondary threshold (opt-in; requires "
        "--pair-production-model). Positrons annihilate in flight and at rest; each "
        "annihilation photon escapes or is absorbed at its first interaction."
    ),
)
@click.option(
    "--atomic-electron-deflection",
    type=click.Choice(_ATOMIC_ELECTRON_DEFLECTION_VALUES),
    help=(
        "Angular deflection by atomic electrons: kawrakow (default) scales each element's "
        "elastic rate by 1 + xi/Z (xi = 1 under continuous stopping; the Moller share "
        "above W_c is removed under shell-soft-hard, which needs elsepa); none keeps "
        "elastic-only deflection."
    ),
)
@_profile_shared.mixed_transport_options
@click.option(
    "--shell-ionization-model",
    type=click.Choice(_SHELL_IONIZATION_MODEL_VALUES),
    help=(
        "Characteristic-line primary vacancies: eedl (default) subshell cross sections, or "
        "the opt-in bote-salvat K/L/M fit (needs 'pyrite tables fetch bote-salvat'; EEDL "
        "for higher shells)."
    ),
)
@click.option(
    "--cxr-source",
    type=click.Choice(_CXR_SOURCE_VALUES),
    help=(
        "Characteristic-line (CXR) scoring: mc (default) or none, which skips reflection "
        "selection, the line grid and every line pass for continuum-only studies. none "
        "refuses coherent emission, temporal profiles, line roulette, channeling and "
        "transition radiation."
    ),
)
@click.option(
    "--channeling-model",
    type=NoneOr(click.Choice(("planar",))),
    help="Planar quantum entrance zone (default none); requires plane, midpoint, uncoupled "
    "radiation and incoherent emission. none turns it off (also reset --channeling-plane "
    "with 'profile physics reset PROFILE channeling-plane channeling-admission').",
)
@click.option(
    "--transition-radiation",
    type=NoneOr(click.Choice(_TRANSITION_RADIATION_VALUES)),
    help="Transition-radiation component (default none): history (per-history coherent "
    "sum), bunch (adds CTR cross-history terms) or experimental two-beam PXR/DTR/CBS; "
    "requires midpoint energy and no channeling. none turns it off.",
)
@click.option(
    "--channeling-plane",
    type=(int, int, int),
    metavar="H K L",
    help="Nonzero Miller indices of the planar channeling family; requires planar.",
)
@click.option(
    "--channeling-admission",
    type=click.Choice(("admitted", "accept-inaccurate")),
    help="admitted (default) runs only channeling cases in the measured admission "
    "tables; accept-inaccurate runs any other case, acknowledging possibly very "
    "inaccurate results, marked unadmitted with no error budget.",
)
@click.option(
    "--channeling-basis-size",
    type=click.IntRange(min=1),
    help="Transverse plane-wave band |h| <= H (table basis_size; default 16).",
)
@click.option(
    "--channeling-bloch-cells",
    type=click.IntRange(min=1),
    help="Bloch-grid supercell size (table bloch_cells; default 16).",
)
@click.option(
    "--channeling-dz-ang",
    type=unit_float(click.FloatRange(min=0, min_open=True), "angstrom", "length"),
    help="Zone depth step in angstrom (table step_ang; default 250).",
)
@click.option(
    "--channeling-zone-max-ang",
    type=unit_float(click.FloatRange(min=0, min_open=True), "angstrom", "length"),
    help="Zone length cap in angstrom (table zone_max_ang; default first layer thickness).",
)
@click.option(
    "--math",
    type=click.Choice(("fast", "reproducible")),
    help=(
        "Floating-point math mode (#1468): reproducible keeps bit-for-bit results where they "
        "are pinned; fast lets adopted kernels use fast-math flags, approximate "
        "transcendentals and FMA contraction (not bit-for-bit across hosts or devices). "
        "Unset defers to PYRITE_MATH, then the numerics.math config key, then reproducible. "
        "fast joins case identity, so its checkpoints never mix with reproducible ones."
    ),
)
def _options_holder(**_values):
    """Parameter holder; never invoked."""


_PARAMS = list(reversed(getattr(_options_holder, "__click_params__", [])))


def field_params(
    *, physics: bool, removed: bool = True, channeling: bool = True
) -> list[click.Parameter]:
    """Fresh copies of the owner's field options (plus removed-spelling guards)."""
    names = field_names(physics=physics, channeling=channeling)
    return [
        copy(param)
        for param in _PARAMS
        if names.get((param.name or "").replace("_", "-")) is not None
        or (removed and not physics and not param.expose_value)
    ]
