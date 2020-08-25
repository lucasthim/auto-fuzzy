"""
The :mod:`autofis.base` module includes the base functions for an automatic fuzzy inference system for
classification and regression tasks.
"""

from .aggregation import Aggregation
from .association import Association
from .decision import Decision
from .formulation import Formulation
from .fuzzification import Fuzzification
from .filter import *
from .utils import *

__all__ = ["Aggregation", "Association", "Decision", "Formulation", "Fuzzification"]