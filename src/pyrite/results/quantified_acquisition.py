"""Compatibility alias for the instrument-owned acquisition quantity view."""

# get_type_hints resolves class annotations in its historical __module__ namespace.
from typing import ClassVar as ClassVar
from typing import Literal as Literal

import numpy as np  # noqa: F401 - retained annotation namespace

from ..instrument.quantified_acquisition import QuantifiedAcquisitionBatch

__all__ = ["QuantifiedAcquisitionBatch"]
