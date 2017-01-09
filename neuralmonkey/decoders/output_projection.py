"""
This module contains different variants of projection functions
for RNN outputs.
"""
# tests: lint, mypy
import tensorflow as tf

from neuralmonkey.nn.projection import maxout
from neuralmonkey.nn.mlp import multilayer_perceptron


def no_deep_output(prev_state, prev_output, ctx_tensors):
    """Compute RNN output out of the previous state and output, and the
    context tensors returned from attention mechanisms.

    This function corresponds to the equations for computation the
    t_tilde in the Bahdanau et al. (2015) paper, on page 14,
    **before** the linear projection.

    Arguments:
        prev_state: Previous decoder RNN state. (Denoted s_i-1)
        prev_output: Embedded output of the previous step. (y_i-1)
        ctx_tensors: Context tensors computed by the attentions. (c_i)

    Returns:
        This function returns the concatenation of all its inputs.
    """
    return tf.concat(1, [prev_state, prev_output] + ctx_tensors)


def maxout_output(maxout_size):
    """Compute RNN output out of the previous state and output, and the
    context tensors returned from attention mechanisms, as described
    in the article

    This function corresponds to the equations for computation the
    t_tilde in the Bahdanau et al. (2015) paper, on page 14,
    with the maxout projection, before the last linear projection.

    Arguments:
        maxout_size: The size of the hidden maxout layer in the deep output

    Returns:
        Returns the maxout projection of the concatenated inputs
    """
    def _projection(prev_state, prev_output, ctx_tensors):
        return maxout([prev_state, prev_output] + ctx_tensors,
                      maxout_size)

    return _projection


# TODO fix this so it does not need dropout_plc. But it needs to be
# able to do dropout!! So does maxout, BTW..
def mlp_output(layer_sizes, dropout_plc=None, activation=tf.tanh):
    """Compute RNN deep output using the multilayer perceptron
    with a specified activation function.
    (Pascanu et al., 2013 [https://arxiv.org/pdf/1312.6026v5.pdf])

    Arguments:
        layer_sizes: A list of sizes of the hiddel layers of the MLP
        dropout_plc: Dropout placeholder. TODO this is not going to
            work with current configuration
        activation: The activation function to use in each layer.
    """
    def _projection(prev_state, prev_output, ctx_tensors):
        mlp_input = tf.concat(1, [prev_state, prev_output] + ctx_tensors)

        return multilayer_perceptron(mlp_input, layer_sizes,
                                     activation=activation,
                                     dropout_plc=dropout_plc,
                                     scope="deep_output_mlp")

    return _projection
