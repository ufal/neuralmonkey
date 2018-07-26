"""Variable regularizers.

This module contains classes that can be used as a variable regularizers
during training. All implementation should be derived from the BaseRegularizer
class.

"""
from typing import List

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.logging import log


class BaseRegularizer:
    """Base class for the regularizers."""

    def __init__(self,
                 name: str,
                 weight: float) -> None:
        check_argument_types()

        self._name = name
        self._weight = weight

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight

    def value(self, variables) -> float:
        raise NotImplementedError("Abstract method")


class L1Regularizer(BaseRegularizer):

    def __init__(self,
                 name: str = "train_l1",
                 weight: float = 1.0e-8) -> None:
        BaseRegularizer.__init__(self, name, weight)

    def value(self, variables: List[tf.Tensor]) -> float:
        return sum(tf.reduce_sum(abs(v)) for v in variables)


class L2Regularizer(BaseRegularizer):

    def __init__(self,
                 name: str = "train_l2",
                 weight: float = 1.0e-8) -> None:
        BaseRegularizer.__init__(self, name, weight)

    def value(self, variables: List[tf.Tensor]) -> float:
        return sum(tf.reduce_sum(v ** 2) for v in variables)


class EWCRegularizer(BaseRegularizer):
    """Regularizer based on the Elastic Weight Consolidation.

    TODO description
    """

    def __init__(self,
                 name: str = "train_ewc",
                 weight: float = 0.,
                 gradients_file: str = None,
                 variables_file: str = None) -> None:
        check_argument_types()

        BaseRegularizer.__init__(self, name, weight)

        if gradients_file is None:
            raise ValueError("Missing gradients_file")
        if variables_file is None:
            raise ValueError("Missing variables_file")

        log("Loading initial variables for EWC from {}".format(variables_file))
        self.init_vars = tf.contrib.framework.load_checkpoint(variables_file)
        log("EWC initial variables loaded")

        log("Loading gradient estimates from {}".format(gradients_file))
        self.gradients = np.load(gradients_file)
        log("Gradient estimates loaded")

    def value(self, variables: List[tf.Tensor]) -> float:
        ewc_value = 0.0
        for var in variables:
            var_name = var.name.split(":")[0]
            init_var = self.init_vars.get_tensor(var_name)
            gradient = self.gradients[var_name]
            ewc_value += tf.reduce_sum(tf.multiply(
                tf.square(gradient), tf.square(var - init_var)))

        return ewc_value


L1 = L1Regularizer()
L2 = L2Regularizer()
