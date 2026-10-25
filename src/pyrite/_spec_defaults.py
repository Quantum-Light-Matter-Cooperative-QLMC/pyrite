"""Internal omission marker shared by independent simulation specifications."""

from enum import Enum


class _Unset(Enum):
    VALUE = "not supplied"


UNSET = _Unset.VALUE
