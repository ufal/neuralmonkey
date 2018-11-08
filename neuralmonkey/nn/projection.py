"""Module which implements various types of projections."""
from typing import List, Callable, Set
import tensorflow as tf
from typeguard import check_argument_types
from neuralmonkey.nn.utils import dropout
from neuralmonkey.decorators import tensor
from neuralmonkey.model.stateful import TemporalStateful
from neuralmonkey.model.model_part import ModelPart, InitializerSpecs


class TemporalStatefulProjection(ModelPart, TemporalStateful):

    def __init__(self,
                 name: str,
                 inputs: List[TemporalStateful],
                 dim: int,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        ModelPart.__init__(self, name, reuse, save_checkpoint, load_checkpoint,
                           initializers)

        self.inputs = inputs
        self.dim = dim

        if not inputs:
            raise ValueError("At least one input is required")

    @tensor
    def assertions(self) -> tf.Tensor:
        first_mask = self.inputs[0].temporal_mask
        return [tf.assert_equal(inp.temporal_mask, first_mask)
                for inp in self.inputs[1:]]

    @tensor
    def temporal_states(self) -> tf.Tensor:
        with tf.control_dependencies(self.assertions):
            cat = tf.concat([i.temporal_states for i in self.inputs], axis=2)
            return tf.layers.dense(cat, self.dim)

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        with tf.control_dependencies(self.assertions):
            return self.inputs[0].temporal_mask

    def get_dependencies(self) -> Set[ModelPart]:
        to_return = ModelPart.get_dependencies(self)
        to_return = to_return.union(
            *(inp.get_dependencies() for inp in self.inputs))

        return to_return


def maxout(inputs: tf.Tensor,
           size: int,
           scope: str = "MaxoutProjection") -> tf.Tensor:
    """Apply a maxout operation.

    Implementation of Maxout layer (Goodfellow et al., 2013).

    http://arxiv.org/pdf/1302.4389.pdf

    z = Wx + b
    y_i = max(z_{2i-1}, z_{2i})

    Arguments:
        inputs: A tensor or list of tensors. It should be 2D tensors with
                equal length in the first dimension (batch size)
        size: The size of dimension 1 of the output tensor.
        scope: The name of the scope used for the variables

    Returns:
        A tensor of shape batch x size
    """
    with tf.variable_scope(scope):
        projected = tf.layers.dense(inputs, size * 2, name=scope)
        maxout_input = tf.reshape(projected, [-1, 1, 2, size])
        maxpooled = tf.nn.max_pool(
            maxout_input, [1, 1, 2, 1], [1, 1, 2, 1], "SAME")

        reshaped = tf.reshape(maxpooled, [-1, size])
        return reshaped


def multilayer_projection(
        input_: tf.Tensor,
        layer_sizes: List[int],
        train_mode: tf.Tensor,
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
        dropout_keep_prob: float = 1.0,
        scope: str = "mlp") -> tf.Tensor:
    mlp_input = input_

    with tf.variable_scope(scope):
        for i, size in enumerate(layer_sizes):
            mlp_input = tf.layers.dense(
                mlp_input,
                size,
                activation=activation,
                name="mlp_layer_{}".format(i))

            mlp_input = dropout(mlp_input, dropout_keep_prob, train_mode)

    return mlp_input


def glu(input_: tf.Tensor,
        gating_fn: Callable[[tf.Tensor], tf.Tensor] = tf.sigmoid) -> tf.Tensor:
    """Apply a Gated Linear Unit.

    Gated Linear Unit - Dauphin et al. (2016).

    http://arxiv.org/abs/1612.08083
    """
    dimensions = input_.get_shape().as_list()

    if dimensions[-1] % 2 != 0:
        raise ValueError("Input size should be an even number")

    lin, nonlin = tf.split(input_, 2, axis=len(dimensions) - 1)

    return lin * gating_fn(nonlin)
