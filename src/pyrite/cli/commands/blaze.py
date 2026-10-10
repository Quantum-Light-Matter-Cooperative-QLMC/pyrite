"""Click wiring for ``pyrite material blaze``."""

from pathlib import Path

import click

from ...console import output as _cli_core
from ...console.outputs import output_default, output_label
from ...runs import blaze as _blaze
from .. import _completion as _cli_completion
from .._deprecations import canonical_option


class _BlazeCommand(click.Command):
    """Preserve one-or-more values after selected options."""

    _variadic = frozenset({"--energy", "--spacing", "--polar", "--angles"})
    _option_names = frozenset(
        {
            "--energy",
            "--spacing",
            "--polar",
            "--angles",
            "--workers",
            "--checkpoint-dir",
            "--max-minutes",
            "--progress-file",
            "--no-progress",
            "--json",
            "--help",
        }
    )

    def parse_args(self, ctx, args):
        normalized = []
        index = 0
        while index < len(args):
            token = args[index]
            option, separator, first = token.partition("=")
            if option not in self._variadic:
                normalized.append(token)
                index += 1
                continue
            values = [first] if separator else []
            index += 1
            while index < len(args):
                candidate = args[index]
                try:
                    float(candidate)
                    numeric = True
                except ValueError:
                    numeric = False
                if candidate in self._option_names or (candidate.startswith("-") and not numeric):
                    break
                values.append(candidate)
                index += 1
            if not values:
                normalized.append(option)
            else:
                for value in values:
                    normalized.extend((option, value))
        return super().parse_args(ctx, normalized)


_EMISSION_ANGLE = click.FloatRange(min=0.0, max=90.0, min_open=True, max_open=True)


@click.command(
    "blaze",
    cls=_BlazeCommand,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Run one material's blazed-crystal MC sweep and write its checkpoint.\n\n"
        "Writes checkpoints/<material>_blazed/, separate from flat-face scan "
        "checkpoints. Repeat --energy/--spacing/--polar for multiple values."
    ),
)
@click.argument(
    "material",
    type=_cli_completion.MATERIAL,
    shell_complete=_cli_completion.complete_material,
)
@click.option(
    "--energy",
    "energies",
    type=_cli_core.POSITIVE_FLOAT,
    multiple=True,
    required=True,
    metavar="E",
    help="Beam energies in keV (one or more).",
)
@click.option(
    "--spacing",
    "spacings",
    type=_cli_core.POSITIVE_FLOAT,
    multiple=True,
    required=True,
    metavar="S",
    help="Groove spacing(s) in meters (one, or one per energy).",
)
@canonical_option(
    "--polar",
    "angles",
    type=_EMISSION_ANGLE,
    multiple=True,
    default=None,
    metavar="A",
    help="Polar tilt values in degrees.",
)
@click.option(
    "--workers",
    type=_cli_core.NONNEGATIVE_INT,
    default=None,
    help="run_cases max_workers (default auto; 0 = serial).",
)
@click.option(
    "--checkpoint-dir",
    default=output_default("checkpoints"),
    show_default=output_label("checkpoints"),
    metavar="DIR",
    help="Read and write blazed component checkpoints in DIR.",
)
@click.option(
    "--max-minutes",
    type=_cli_core.POSITIVE_FLOAT,
    default=None,
    metavar="MINUTES",
    help="Soft wall-clock budget in minutes; exit 75 if resumable work remains.",
)
@click.option("--progress-file", type=click.Path(path_type=Path), default=None, hidden=True)
@click.option("--no-progress", is_flag=True, hidden=True)
@_cli_core.output_option
def command(
    material,
    energies,
    spacings,
    angles,
    workers,
    checkpoint_dir,
    max_minutes,
    progress_file,
    no_progress,
    json_output,
):
    handler = _blaze._run_json if json_output else _blaze.run
    return _cli_core.invoke_legacy(
        handler,
        material=material,
        energy=list(energies),
        spacing=list(spacings),
        angles=list(angles) if angles else None,
        workers=workers,
        checkpoint_dir=checkpoint_dir,
        max_minutes=max_minutes,
        progress_file=progress_file,
        no_progress=no_progress,
        json_output=json_output,
    )
