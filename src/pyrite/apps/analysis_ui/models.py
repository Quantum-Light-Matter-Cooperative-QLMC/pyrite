from dataclasses import dataclass, replace
from typing import Any, Literal

ScaleType = Literal["linear", "log"]
Domain = tuple[float, float] | None


@dataclass(frozen=True)
class AxisSpec:
    """Validated axis configuration passed to plotting functions."""

    x_domain: Domain
    y_domain: Domain
    x_type: ScaleType
    y_type: ScaleType
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AxisPair:
    narrow: AxisSpec
    broad: AxisSpec


@dataclass(frozen=True)
class AnalysisContext:
    """The currently loaded checkpoint and its selected emission view."""

    selected_material: str | None
    selected_face: str | None
    checkpoint_stem: str | None
    settings: Any
    checkpoint_results: Any
    results: Any
    cases: Any
    load_error: str | None = None
    notice: str | None = None
    emission: str = "incoherent"

    @property
    def has_data(self) -> bool:
        return bool(self.checkpoint_results)

    @property
    def show_both_emissions(self) -> bool:
        return self.emission == "both"

    def with_results(self, results: Any, *, emission: str | None = None) -> AnalysisContext:
        return replace(self, results=results, emission=emission or self.emission)

    def title_for_face(self, chart: Any) -> Any:
        if chart is None or self.selected_face != "blazed" or not hasattr(chart, "title"):
            return chart
        title = chart.title
        if isinstance(title, str):
            return chart.properties(title=f"{title} (blazed)")
        try:
            import altair as alt

            params = title.to_dict(validate=False)
            text = params.get("text")
            if isinstance(text, str):
                params["text"] = f"{text} (blazed)"
                return chart.properties(title=alt.TitleParams(**params))
        except AttributeError, TypeError, ValueError:
            pass
        return chart


@dataclass(frozen=True)
class DimensionComparisonSpec:
    """One Spectra-view "Vary" choice.

    ``varying_key`` is the case field that varies across curves; ``pinned``
    names the sidebar slice dimensions (``energy``/``tilt``/``azimuth``) held
    fixed. Thickness is always pinned.
    """

    varying_key: str
    varying_label: str
    varying_plural: str
    unit: str
    pinned: tuple[str, ...]
    description: str
