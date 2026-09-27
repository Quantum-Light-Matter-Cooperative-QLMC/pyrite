import math
from collections.abc import Mapping

from .models import AxisPair, AxisSpec, Domain


def _resolved_domain(
    *,
    auto: bool,
    lower: float | None,
    upper: float | None,
    logarithmic: bool,
    positive_floor: float,
    axis_name: str,
) -> tuple[Domain, tuple[str, ...]]:
    if auto:
        return None, ()

    warnings: list[str] = []
    try:
        resolved_lower = float(str(lower).replace(",", ""))
        resolved_upper = float(str(upper).replace(",", ""))
    except (TypeError, ValueError):
        return None, (f"{axis_name}: invalid manual limits; using automatic domain.",)
    if not math.isfinite(resolved_lower) or not math.isfinite(resolved_upper):
        return None, (f"{axis_name}: non-finite manual limits; using automatic domain.",)

    if logarithmic and resolved_lower <= 0:
        if resolved_upper <= 0:
            return None, (
                f"{axis_name}: log limits must include positive values; using automatic domain.",
            )
        # Scale the fallback to the requested upper limit, but retain a strict floor.
        resolved_lower = min(max(positive_floor, abs(resolved_upper) * 1e-9), resolved_upper / 10.0)
        warnings.append(
            f"{axis_name}: lower limit was non-positive in log mode; using {resolved_lower:g}."
        )

    if resolved_upper <= resolved_lower:
        return None, (
            *warnings,
            f"{axis_name}: upper limit must exceed lower limit; using automatic domain.",
        )

    return (resolved_lower, resolved_upper), tuple(warnings)


def resolve_axis_spec(
    values: Mapping[str, object],
    *,
    include_y_domain: bool = False,
    x_positive_floor: float = 1e-12,
    y_positive_floor: float = 1e-30,
) -> AxisSpec:
    x_log = bool(values.get("xlog", False))
    y_log = bool(values.get("ylog", False))
    auto = bool(values.get("auto", False))

    x_domain, x_warnings = _resolved_domain(
        auto=auto,
        lower=values.get("xmin"),
        upper=values.get("xmax"),
        logarithmic=x_log,
        positive_floor=x_positive_floor,
        axis_name="x-axis",
    )

    if include_y_domain and (values.get("ymin"), values.get("ymax")) not in (
        (0, 0),
        ("0", "0"),
    ):
        y_domain, y_warnings = _resolved_domain(
            auto=auto,
            lower=values.get("ymin"),
            upper=values.get("ymax"),
            logarithmic=y_log,
            positive_floor=y_positive_floor,
            axis_name="y-axis",
        )
    else:
        y_domain, y_warnings = None, ()

    return AxisSpec(
        x_domain=x_domain,
        y_domain=y_domain,
        x_type="log" if x_log else "linear",
        y_type="log" if y_log else "linear",
        warnings=(*x_warnings, *y_warnings),
    )


def resolve_axis_pair(
    control_values: Mapping[str, object],
    *,
    include_y_domain: bool = False,
) -> AxisPair:
    return AxisPair(
        narrow=resolve_axis_spec(
            control_values["narrow"],
            include_y_domain=include_y_domain,
        ),
        broad=resolve_axis_spec(
            control_values["broad"],
            include_y_domain=include_y_domain,
        ),
    )
