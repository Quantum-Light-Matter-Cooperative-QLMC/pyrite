"""Warning category for a physical model substituted after missing reference data.

Consumers may promote this category to an error with Python's warning filters.
The presence of this warning means the requested and executed physics differ;
selecting EEDL or another model *explicitly* is not a fallback.
"""


class PhysicsFallbackWarning(RuntimeWarning):
    """A requested physics model was replaced by a lower-fidelity surrogate."""
