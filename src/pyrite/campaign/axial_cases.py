"""Campaign-side names of the axial admission resolver (#994).

The resolver lives with the physics core
(:mod:`pyrite.montecarlo.channeling.axial_case`) so case validation and the
runner can use it without importing campaign code.
"""

from ..montecarlo.channeling.axial_case import ALIGNED_TILT_DEG as ALIGNED_TILT_DEG
from ..montecarlo.channeling.axial_case import axial_beam_angles as axial_beam_angles
from ..montecarlo.channeling.axial_case import axial_case_query as axial_case_query
from ..montecarlo.channeling.axial_case import axial_channeling_record as axial_channeling_record
