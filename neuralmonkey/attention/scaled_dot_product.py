"""The scaled dot-product attention mechanism defined in Vaswani et al. (2017).

The attention energies are computed as dot products between the query vector
and the key vector. The query vector is scaled down by the square root of its
dimensionality. This attention function has no trainable parameters.

See arxiv.org/abs/1706.03762
"""
import math
from typing import Tuple, Callable, Union

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.nn.utils import dropout
from neuralmonkey.model.model_part import InitializerSpecs, ModelPart
from neuralmonkey.attention.base_attention import (
    BaseAttention, Attendable, get_attention_states, get_attention_mask)
from neuralmonkey.attention.namedtuples import MultiHeadLoopState


def split_for_heads(x: tf.Tensor, n_heads: int, head_dim: int) -> tf.Tensor:
    """Split a tensor for multi-head attention.

    Split last dimension of 3D vector of shape ``(batch, time, dim)`` and
    return a 4D vector with shape ``(batch, n_heads, time, dim/n_heads)``.

    Arguments:
        x: input Tensor of shape ``(batch, time, dim)``.
        n_heads: Number of attention heads.
        head_dim: Dimension of the attention heads.

    Returns:
        A 4D Tensor of shape ``(batch, n_heads, time, head_dim/n_heads)``
    """
    x_shape = tf.shape(x)
    x_4d = tf.reshape(tf.expand_dims(x, 2),
                      [x_shape[0], x_shape[1], n_heads, head_dim])

    return tf.transpose(x_4d, perm=[0, 2, 1, 3])


def mask_energies(energies_4d: tf.Tensor,
                  mask: tf.Tensor,
                  mask_value=-1e9) -> tf.Tensor:
    """Apply mask to the attention energies before passing to softmax.

    Arguments:
        energies_4d: Energies of shape ``(batch, n_heads, time(q), time(k))``.
        mask: Float Tensor of zeros and ones of shape ``(batch, time(k))``,
            specifies valid positions in the energies tensor.
        mask_value: Value used to mask energies. Default taken value
            from tensor2tensor.

    Returns:
        Energies (logits) of valid positions. Same shape as ``energies_4d``.

    NOTE:
        We do not use ``mask_value=-np.inf`` to avoid potential underflow.
    """
    mask_4d = tf.expand_dims(tf.expand_dims(mask, 1), 1)
    energies_all = energies_4d * mask_4d

    # Energies are log probabilities, so setting the invalid energies to
    # negative infinity (aka -1e9 for compatibility with tensor2tensor) yields
    # probability of zero to the padded positions.
    return energies_all + (1.0 - mask_4d) * mask_value


def mask_context(energies: tf.Tensor,
                 left: bool = False,
                 mask_value: float = -1e9) -> tf.Tensor:
    """Mask energies of keys using lower triangular matrix.

    Mask simulates autoregressive decoding, such that it prevents
    the attention to look at what has not yet been decoded.
    Mask is not necessary during training when true output values
    are used instead of the decoded ones.

    Arguments:
        energies: A tensor to mask.
        left: By default, mask future states (right of the given position).
            Setting this to True will mask past states (left of pos).
        mask_value: Value used to mask energies.

    Returns:
        Masked energies tensor.
    """
    if not left:
        triangular_mask = tf.matrix_band_part(tf.ones_like(energies), -1, 0)
    else:
        triangular_mask = tf.matrix_band_part(tf.ones_like(energies), 0, -1)

    mask_area = tf.equal(triangular_mask, 1)

    # Note that for compatibility with tensor2tensor, we use -1e9 for negative
    # infinity.
    masked_value = tf.fill(tf.shape(energies), mask_value)
    return tf.where(mask_area, energies, masked_value)


# pylint: disable=too-many-locals
# TODO split this to more functions
def attention(
        queries: tf.Tensor,
        keys: tf.Tensor,
        values: tf.Tensor,
        keys_mask: tf.Tensor,
        num_heads: int,
        dropout_callback: Callable[[tf.Tensor], tf.Tensor],
        mask_left_context: bool = False,
        mask_right_context: bool = False,
        use_bias: bool = False) -> tf.Tensor:
    """Run multi-head scaled dot-product attention.

    See arxiv.org/abs/1706.03762

    When performing multi-head attention, the queries, keys and values
    vectors are first split to sets of smaller vectors, one for each attention
    head. Next, they are transformed using a linear layer and a separate
    attention (from a corresponding head) is applied on each set of
    the transformed triple of query, key and value. The resulting contexts
    from each head are then concatenated and a linear layer is applied
    on this concatenated output. The following can be summed by following
    equations::

        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_o
        head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)

    The scaled dot-product attention is a simple dot-product between
    the query and a transposed key vector. The result is then scaled
    using square root of the vector dimensions and a softmax layer is applied.
    Finally, the output of the softmax layer is multiplied by the value vector.
    See the following equation::

        Attention(Q, K, V) = softmax(Q * K^T / √(d_k)) * V

    Arguments:
        queries: Input queries of shape ``(batch, time(q), k_channels)``.
        keys: Input keys of shape ``(batch, time(k), k_channels)``.
        values: Input values of shape ``(batch, time(k), v_channels)``.
        keys_mask: A float Tensor for masking sequences in keys.
        num_heads: Number of attention heads.
        dropout_callback: Callable function implementing dropout.
        use_bias: If True, enable bias in the attention head projections
            (for all queries, keys and values).
        mask_left_context: Flag whether to mask past energies (left context).
        mask_right_context: Flag whether to mask future energies (right ctx.)

    Returns:
        Contexts of shape ``(batch, time(q), v_channels)`` and
        weights of shape ``(batch, time(q), time(k))``.
    """
    if num_heads <= 0:
        raise ValueError("Number of heads must be greater than zero.")

    queries_dim = queries.shape.as_list()[-1]
    keys_shape = keys.shape.as_list()
    values_shape = values.shape.as_list()

    # Query and keys should match in the last dimension
    if queries_dim != keys_shape[-1]:
        raise ValueError(
            "Queries and keys do not match in the last dimension."
            " Queries: {}, Keys: {}".format(queries_dim, keys_shape[-1]))

    if keys_shape[1] != values_shape[1]:
        raise ValueError(
            "Keys and values 'time' dimension does not match. "
            "Keys: {}, Values: {}".format(keys_shape[1], values_shape[1]))

    # Last dimension must be divisible by num_heads
    if queries_dim % num_heads != 0:
        raise ValueError(
            "Last dimension of the query ({}) should be divisible by the "
            "number of heads ({})".format(queries_dim, num_heads))

    head_dim = int(queries_dim / num_heads)

    # For multi-head attention, queries, keys and values are linearly projected
    if num_heads > 1:
        queries = tf.layers.dense(
            queries, queries_dim, use_bias=use_bias, name="query_proj")
        keys = tf.layers.dense(
            keys, queries_dim, use_bias=use_bias, name="keys_proj")
        values = tf.layers.dense(
            values, queries_dim, use_bias=use_bias, name="vals_proj")

    # Scale first:
    queries_scaled = queries / math.sqrt(head_dim)

    # Reshape the k_channels dimension to the number of heads
    queries = split_for_heads(queries_scaled, num_heads, head_dim)
    keys = split_for_heads(keys, num_heads, head_dim)
    values = split_for_heads(values, num_heads, head_dim)

    # For dot-product, we use matrix multiplication
    # shape: batch, head, time(q), time(k) (k_channels is the matmul axis)
    energies = tf.matmul(queries, keys, transpose_b=True)

    # To protect the attention from looking ahead of time, we must replace the
    # energies of future keys with negative infinity
    if mask_right_context:
        energies = mask_context(energies)

    if mask_left_context:
        energies = mask_context(energies, left=True)

    # To exclude the padded positions (those after the end of sentence),
    # we mask the attention energies given this mask.
    if keys_mask is not None:
        energies = mask_energies(energies, keys_mask)
    energies = tf.identity(energies, "energies")

    # Softmax along the last axis
    # shape: batch, head, time(q), time(k)
    weights = tf.nn.softmax(energies)

    # apply dropout to the weights (Attention Dropout)
    weights = dropout_callback(weights)

    context = tf.matmul(weights, values)

    # transpose and reshape to shape [batch, time(q), v_channels]
    context_shape = tf.shape(context)
    context = tf.reshape(
        tf.transpose(context, perm=[0, 2, 1, 3]),
        [context_shape[0], context_shape[2], queries_dim])

    if num_heads > 1:
        # pylint: disable=redefined-variable-type
        # This seems like a pylint bug
        context = tf.layers.dense(
            context, queries_dim, use_bias=use_bias, name="output_proj")
        # pylint: enable=redefined-variable-type

    return context, weights
# pylint: enable=too-many-locals


def empty_multi_head_loop_state(
        batch_size: Union[int, tf.Tensor],
        num_heads: Union[int, tf.Tensor],
        length: Union[int, tf.Tensor],
        dimension: Union[int, tf.Tensor]) -> MultiHeadLoopState:

    return MultiHeadLoopState(
        contexts=tf.zeros(
            shape=[0, batch_size, dimension],
            dtype=tf.float32,
            name="contexts"),
        head_weights=[tf.zeros(
            shape=[0, batch_size, length],
            dtype=tf.float32,
            name="distributions_head{}".format(i)) for i in range(num_heads)])


class MultiHeadAttention(BaseAttention):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 n_heads: int,
                 keys_encoder: Attendable,
                 values_encoder: Attendable = None,
                 dropout_keep_prob: float = 1.0,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        BaseAttention.__init__(self, name, reuse, save_checkpoint,
                               load_checkpoint, initializers)

        self.n_heads = n_heads
        self.dropout_keep_prob = dropout_keep_prob

        if self.n_heads <= 0:
            raise ValueError("Number of heads must be greater than zero.")

        if self.dropout_keep_prob <= 0.0 or self.dropout_keep_prob > 1.0:
            raise ValueError("Dropout keep prob must be inside (0,1].")

        if values_encoder is None:
            values_encoder = keys_encoder

        self.attention_keys = get_attention_states(keys_encoder)
        self.attention_mask = get_attention_mask(keys_encoder)
        self.attention_values = get_attention_states(values_encoder)

        self._variable_scope.set_initializer(tf.variance_scaling_initializer(
            mode="fan_avg", distribution="uniform"))
    # pylint: enable=too-many-arguments

    def attention(self,
                  query: tf.Tensor,
                  decoder_prev_state: tf.Tensor,
                  decoder_input: tf.Tensor,
                  loop_state: MultiHeadLoopState) -> Tuple[tf.Tensor,
                                                           MultiHeadLoopState]:
        """Run a multi-head attention getting context vector for a given query.

        This method is an API-wrapper for the global function 'attention'
        defined in this module. Transforms a query of shape(batch, query_size)
        to shape(batch, 1, query_size) and applies the attention function.
        Output context has shape(batch, 1, value_size) and weights
        have shape(batch, n_heads, 1, time(k)). The output is then processed
        to produce output vector of contexts and the following attention
        loop state.

        Arguments:
            query: Input query for the current decoding step
                of shape(batch, query_size).
            decoder_prev_state: Previous state of the decoder.
            decoder_input: Input to the RNN cell of the decoder.
            loop_state: Attention loop state.

        Returns:
            Vector of contexts and the following attention loop state.
        """

        context_3d, weights_4d = attention(
            queries=tf.expand_dims(query, 1),
            keys=self.attention_keys,
            values=self.attention_values,
            keys_mask=self.attention_mask,
            num_heads=self.n_heads,
            dropout_callback=lambda x: dropout(
                x, self.dropout_keep_prob, self.train_mode))

        # head_weights_3d is HEAD-wise list of (batch, 1, 1, time(keys))
        head_weights_3d = tf.split(weights_4d, self.n_heads, axis=1)

        context = tf.squeeze(context_3d, axis=1)
        head_weights = [tf.squeeze(w, axis=[1, 2]) for w in head_weights_3d]

        next_contexts = tf.concat(
            [loop_state.contexts, tf.expand_dims(context, 0)], axis=0)
        next_head_weights = [
            tf.concat([loop_state.head_weights[i],
                       tf.expand_dims(head_weights[i], 0)], axis=0)
            for i in range(self.n_heads)]

        next_loop_state = MultiHeadLoopState(
            contexts=next_contexts,
            head_weights=next_head_weights)

        return context, next_loop_state

    def initial_loop_state(self) -> MultiHeadLoopState:
        return empty_multi_head_loop_state(
            self.batch_size, self.n_heads, tf.shape(self.attention_keys)[1],
            self.context_vector_size)

    def finalize_loop(self, key: str,
                      last_loop_state: MultiHeadLoopState) -> None:
        for i in range(self.n_heads):
            head_weights = last_loop_state.head_weights[i]
            self.histories["{}_head{}".format(key, i)] = head_weights

    @property
    def context_vector_size(self) -> int:
        return self.attention_values.get_shape()[-1].value

    def visualize_attention(self, key: str, max_outputs: int = 16) -> None:
        for i in range(self.n_heads):
            head_key = "{}_head{}".format(key, i)
            if head_key not in self.histories:
                raise ValueError(
                    "Key {} not among attention histories".format(head_key))

            alignments = tf.expand_dims(
                tf.transpose(self.histories[head_key], perm=[1, 2, 0]), -1)

            tf.summary.image("{}_head{}".format(self.name, i), alignments,
                             collections=["summary_att_plots"],
                             max_outputs=max_outputs)


class ScaledDotProdAttention(MultiHeadAttention):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 keys_encoder: Attendable,
                 values_encoder: Attendable = None,
                 dropout_keep_prob: float = 1.0,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        MultiHeadAttention.__init__(
            self, name, 1, keys_encoder, values_encoder, dropout_keep_prob,
            reuse, save_checkpoint, load_checkpoint, initializers)
    # pylint: enable=too-many-arguments
