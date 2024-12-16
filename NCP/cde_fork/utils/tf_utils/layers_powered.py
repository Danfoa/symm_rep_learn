#
# Code from rllab https://github.com/rll/rllab/tree/master/sandbox
#


import itertools

import NCP.cde_fork.utils.tf_utils.layers as L
from NCP.cde_fork.utils.tf_utils.parameterized import Parameterized


class LayersPowered(Parameterized):

    def __init__(self, output_layers, input_layers=None):
        self._output_layers = output_layers
        self._input_layers = input_layers
        Parameterized.__init__(self)

    def get_params_internal(self, **tags):
        layers = L.get_all_layers(self._output_layers, treat_as_input=self._input_layers)
        params = itertools.chain.from_iterable(l.get_params(**tags) for l in layers)
        return L.unique(params)
