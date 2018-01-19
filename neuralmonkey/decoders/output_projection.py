"""Output Projection Module.

This module contains different variants of projection functions of decoder
outputs into the logit function inputs.

Output projections are specified in the configuration file. Each output
projection function has a unified type ``OutputProjection``, which is a
callable that takes four arguments and returns a tensor:

1. ``prev_state`` -- the hidden state of the decoder.
2. ``prev_output`` -- embedding of the previously decoded word (or train input)
3. ``ctx_tensots`` -- a list of context vectors (for each attention object)

To enable further parameterization of output projection functions, one can
use higher-order functions.
"""
from typing import Union, Tuple, List, Callable
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.nn.projection import multilayer_projection, maxout
from neuralmonkey.nn.utils import dropout
from neuralmonkey.tf_utils import get_initializer


# pylint: disable=invalid-name
OutputProjection = Callable[
    [tf.Tensor, tf.Tensor, List[tf.Tensor], tf.Tensor], tf.Tensor]

OutputProjectionSpec = Union[Tuple[OutputProjection, int],
                             OutputProjection]
# pylint: enable=invalid-name


def _legacy_linear(output_size: int) -> Tuple[OutputProjection, int]:
    """Apply a legacy linear projection.

    This was the default projection before commit 9a09553.

    For backward compatibility, set the output_size parameter
    to decoder's rnn_size param.
    """
    check_argument_types()

    # pylint: disable=unused-argument
    def _projection(prev_state, prev_output, ctx_tensors, train_mode):
        state_with_ctx = tf.concat([prev_state] + ctx_tensors, 1)
        return tf.layers.dense(state_with_ctx, output_size,
                               name="AttnOutputProjection")
    # pylint: enable=unused-argument

    return _projection, output_size


def _legacy_relu(output_size: int) -> Tuple[OutputProjection, int]:
    """Apply a legacy relu projection.

    This was the default projection after commit 9a09553.

    For backward compatibility, set the output_size parameter
    to decoder's rnn_size param.
    """
    check_argument_types()

    # pylint: disable=unused-argument
    def _projection(prev_state, prev_output, ctx_tensors, train_mode):
        state_with_ctx = tf.concat([prev_state] + ctx_tensors, 1)
        return tf.layers.dense(state_with_ctx, output_size,
                               activation=tf.nn.relu,
                               name="AttnOutputProjection")
    # pylint: enable=unused-argument

    return _projection, output_size


def nematus_output(
        output_size: int,
        activation_fn: Callable[[tf.Tensor], tf.Tensor] = tf.tanh,
        dropout_keep_prob: float = 1.0) -> Tuple[OutputProjection, int]:
    """Apply nonlinear one-hidden-layer deep output.

    Implementation consistent with Nematus.
    Can be used instead of (and is in theory equivalent to) nonlinear_output.

    Projects the RNN state, embedding of the previously outputted word, and
    concatenation of all context vectors into a shared vector space, sums them
    up and apply a hyperbolic tangent activation function.
    """
    check_argument_types()

    def _projection(prev_state, prev_output, ctx_tensors, train_mode):
        prev_state = dropout(prev_state, dropout_keep_prob, train_mode)
        prev_output = dropout(prev_output, dropout_keep_prob, train_mode)
        ctx_concat = tf.concat(ctx_tensors, 1)
        ctx = dropout(ctx_concat, dropout_keep_prob, train_mode)

        logit_rnn = tf.layers.dense(
            prev_state, output_size,
            kernel_initializer=get_initializer(
                "rnn_state/kernel", tf.glorot_normal_initializer()),
            name="rnn_state")

        logit_emb = tf.layers.dense(
            prev_output, output_size,
            kernel_initializer=get_initializer(
                "prev_out/kernel", tf.glorot_normal_initializer()),
            name="prev_out")

        logit_ctx = tf.layers.dense(
            ctx, output_size,
            kernel_initializer=get_initializer(
                "context/kernel", tf.glorot_normal_initializer()),
            name="context")

        return activation_fn(logit_rnn + logit_emb + logit_ctx)

    return _projection, output_size


def nonlinear_output(
        output_size: int,
        activation_fn: Callable[[tf.Tensor], tf.Tensor] = tf.tanh
) -> Tuple[OutputProjection, int]:
    check_argument_types()

    # pylint: disable=unused-argument
    def _projection(prev_state, prev_output, ctx_tensors, train_mode):
        state_out_ctx = tf.concat([prev_state, prev_output] + ctx_tensors, 1)
        return tf.layers.dense(state_out_ctx, output_size,
                               activation=activation_fn)
    # pylint: enable=unused-argument

    return _projection, output_size


def maxout_output(maxout_size: int) -> Tuple[OutputProjection, int]:
    """Apply maxout.

    Compute RNN output out of the previous state and output, and the
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
    check_argument_types()

    def _projection(prev_state, prev_output, ctx_tensors, _):
        state_out_ctx = tf.concat([prev_state, prev_output] + ctx_tensors, 1)
        return maxout(state_out_ctx, maxout_size)

    return _projection, maxout_size


def mlp_output(layer_sizes: List[int],
               activation: Callable[[tf.Tensor], tf.Tensor] = tf.tanh,
               dropout_keep_prob: float = 1.0) -> Tuple[OutputProjection, int]:
    """Apply a multilayer perceptron.

    Compute RNN deep output using the multilayer perceptron
    with a specified activation function.
    (Pascanu et al., 2013 [https://arxiv.org/pdf/1312.6026v5.pdf])

    Arguments:
        layer_sizes: A list of sizes of the hiddel layers of the MLP
        dropout_keep_prob: the dropout keep probability
        activation: The activation function to use in each layer.
    """
    check_argument_types()

    def _projection(prev_state, prev_output, ctx_tensors, train_mode):
        mlp_input = tf.concat([prev_state, prev_output] + ctx_tensors, 1)

        return multilayer_projection(mlp_input, layer_sizes,
                                     activation=activation,
                                     dropout_keep_prob=dropout_keep_prob,
                                     train_mode=train_mode,
                                     scope="deep_output_mlp")

    return _projection, layer_sizes[-1]
