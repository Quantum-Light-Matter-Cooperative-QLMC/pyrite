"""Opt-in pair-conversion validation shared by Physics and profile settings (#275, #276)."""

#: Opt-in pair conversion of coupled hard photons (issue #275); mirrors
#: ``montecarlo.transport.pair_production.PAIR_PRODUCTION_MODELS`` (a test
#: keeps the two in step) without importing the transport package here.
PAIR_PRODUCTION_MODELS = ("penelope-2024",)


def validate_pair_production_numerics(
    model: object,
    secondary_threshold_eV: object,
    radiative_model: object,
    bremsstrahlung_model: object,
    positron_transport: object = False,
) -> None:
    """Validate opt-in pair conversion: it needs secondaries and coupled photons.

    ``positron_transport`` (#276) needs pair conversion.
    """
    if not isinstance(positron_transport, bool):
        raise TypeError("positron_transport must be a bool")
    if positron_transport and model is None:
        raise ValueError("positron_transport requires pair_production_model")
    if model is None:
        return
    if model not in PAIR_PRODUCTION_MODELS:
        raise ValueError(
            f"pair_production_model must be one of {', '.join(PAIR_PRODUCTION_MODELS)} or None"
        )
    if secondary_threshold_eV is None:
        raise ValueError("pair_production_model requires secondary_threshold_eV")
    if radiative_model == "uncoupled" or bremsstrahlung_model == "eedl":
        raise ValueError("pair_production_model requires coupled BremsLib radiative transport")
