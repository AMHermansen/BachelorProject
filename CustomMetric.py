import numpy as np
from astropy import units as u

from einsteinpy import constant
from einsteinpy.metric import BaseMetric
from einsteinpy.utils import CoordinateError

_c = constant.c.value


class SchwarzchildPertubation(BaseMetric):
    """
    Class for defining
    """
