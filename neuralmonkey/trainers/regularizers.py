"""Variable regularizers.

This module contains classes that can be used as a variable regularizers
during training. All implementation should be derived from the Regularizer
class.
"""
from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.logging import log


class Regularizer(metaclass=ABCMeta):
    """Base class for regularizers.

    Regularizer objects are used to introduce additional loss terms to
    the trainer, thus constraining the model variable during training. These
    loss terms have an adjustable weight allowing to set the "importance"
    of the term.
    """

    def __init__(self,
                 name: str,
                 weight: float) -> None:
        """Create the regularizer.

        Arguments:
            name: Regularizer name.
            weight: Weight of the regularization term (usually expressed
                 as "lambda" in the literature).
        """
        self._name = name
        self._weight = weight

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight

    @abstractmethod
    def value(self, variables: List[tf.Tensor]) -> tf.Tensor:
        """Compute the unweighted value of the regularization loss term.

        Arguments:
            variables: List of the regularizable model variables.
        """
        raise NotImplementedError("Abstract method")


class L1Regularizer(Regularizer):
    """L1 regularizer."""

    def __init__(self,
                 name: str,
                 weight: float) -> None:
        """Create the regularizer.

        Arguments:
            name: Regularizer name.
            weight: Weight of the regularization term.
        """
        Regularizer.__init__(self, name, weight)

    def value(self, variables: List[tf.Tensor]) -> tf.Tensor:
        return sum(tf.reduce_sum(abs(v)) for v in variables)


class L2Regularizer(Regularizer):
    """L2 regularizer."""

    def __init__(self,
                 name: str,
                 weight: float) -> None:
        """Create the regularizer.

        Arguments:
            name: Regularizer name.
            weight: Weight of the regularization term.
        """
        Regularizer.__init__(self, name, weight)

    def value(self, variables: List[tf.Tensor]) -> tf.Tensor:
        return sum(tf.reduce_sum(v ** 2) for v in variables)


class EWCRegularizer(Regularizer):
    """Regularizer based on the Elastic Weight Consolidation.

    Implements Elastic Weight Consolidation from the "Overcoming catastrophic
    forgetting in neural networks" paper.
    The regularizer applies a separate regularization weight to each trainable
    variable based on its importance for the previously learned task.

    https://arxiv.org/pdf/1612.00796.pdf
    """

    def __init__(self,
                 name: str,
                 weight: float,
                 gradients_file: str,
                 variables_file: str) -> None:
        """Create the regularizer.

        Arguments:
            name: Regularizer name.
            weight: Weight of the regularization term.
            gradients_file: File containing the gradient estimates
                from the previous task.
            variables_files: File containing the variables learned
                on the previous task.
        """
        check_argument_types()
        Regularizer.__init__(self, name, weight)

        log("Loading initial variables for EWC from {}."
            .format(variables_file))
        self.init_vars = tf.contrib.framework.load_checkpoint(variables_file)
        log("EWC initial variables loaded.")

        log("Loading gradient estimates from {}.".format(gradients_file))
        self.gradients = np.load(gradients_file)
        log("Gradient estimates loaded.")

    def value(self, variables: List[tf.Tensor]) -> tf.Tensor:
        ewc_value = tf.constant(0.0)
        for var in variables:
            init_var_name = var.name.split(":")[0]
            if (var.name in self.gradients.files
                    and self.init_vars.has_tensor(init_var_name)):
                init_var = tf.constant(
                    self.init_vars.get_tensor(init_var_name),
                    name="{}_init_value".format(init_var_name))
                grad_squared = tf.constant(
                    np.square(self.gradients[var.name]),
                    name="{}_ewc_weight".format(init_var_name))
                ewc_value += tf.reduce_sum(tf.multiply(
                    grad_squared, tf.square(var - init_var)))

        return ewc_value
