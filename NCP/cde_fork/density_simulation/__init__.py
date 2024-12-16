import inspect
import sys

import numpy as np

from .ArmaJump import ArmaJump
from .BaseConditionalDensitySimulation import BaseConditionalDensitySimulation
from .EconDensity import EconDensity
from .GMM import GaussianMixture
from .JumpDiffusionModel import JumpDiffusionModel
from .LinearGaussian import LinearGaussian
from .LinearStudentT import LinearStudentT
from .SkewNormal import SkewNormal

# def get_probabilistic_models_list():
#   clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
#   return np.asarray(clsmembers)

