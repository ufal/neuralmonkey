from typing import Callable, List
import tensorflow as tf

from neuralmonkey.attention.scaled_dot_product import attention
from neuralmonkey.tf_utils import layer_norm


# pylint: disable=too-many-arguments
def single(
        queries: tf.Tensor,
        states: tf.Tensor,
        mask: tf.Tensor,
        n_heads: int,
        attention_dropout_callback: Callable[[tf.Tensor], tf.Tensor],
        dropout_callback: Callable[[tf.Tensor], tf.Tensor],
        normalize: bool = True,
        use_dropout: bool = True,
        residual: bool = True,
        use_att_transform_bias: bool = False):

    # Layer normalization
    normalized_queries = layer_norm(queries) if normalize else queries

    # Attend to the encoder
    # TODO handle attention histories
    encoder_context, _ = attention(
        queries=normalized_queries,
        keys=states,
        values=states,
        keys_mask=mask,
        num_heads=n_heads,
        dropout_callback=attention_dropout_callback,
        use_bias=use_att_transform_bias)

    # Apply dropout
    if use_dropout:
        encoder_context = dropout_callback(encoder_context)

    # Add residual connections
    if residual:
        encoder_context += queries

    return encoder_context
# pylint: enable=too-many-arguments


def serial(queries: tf.Tensor,
           encoder_states: List[tf.Tensor],
           encoder_masks: List[tf.Tensor],
           heads: List[int],
           attention_dropout_callbacks: List[Callable[[tf.Tensor], tf.Tensor]],
           dropout_callback: Callable[[tf.Tensor], tf.Tensor]) -> tf.Tensor:
    # serial:
    # * repeat for every encoder:
    #   - lnorm + attend + dropout + add residual
    # * update queiies between layers

    context = queries
    for i, (states, mask, n_heads, attn_drop_cb) in enumerate(zip(
            encoder_states, encoder_masks, heads,
            attention_dropout_callbacks)):

        with tf.variable_scope("enc_{}".format(i)):
            context = single(context, states, mask, n_heads,
                             attention_dropout_callback=attn_drop_cb,
                             dropout_callback=dropout_callback)
    return context


def parallel(
        queries: tf.Tensor,
        encoder_states: List[tf.Tensor],
        encoder_masks: List[tf.Tensor],
        heads: List[int],
        attention_dropout_callbacks: List[Callable[[tf.Tensor], tf.Tensor]],
        dropout_callback: Callable[[tf.Tensor], tf.Tensor]) -> tf.Tensor:
    # parallel:
    # * normalize queries,
    # * attend and dropout independently for every encoder,
    # * sum up the results
    # * add residual and return

    normalized_queries = layer_norm(queries)
    contexts = []

    for i, (states, mask, n_heads, attn_drop_cb) in enumerate(zip(
            encoder_states, encoder_masks, heads,
            attention_dropout_callbacks)):

        with tf.variable_scope("enc_{}".format(i)):
            contexts.append(
                single(normalized_queries, states, mask, n_heads,
                       attention_dropout_callback=attn_drop_cb,
                       dropout_callback=dropout_callback,
                       normalize=False, residual=False))

    return sum(contexts) + queries


# pylint: disable=too-many-locals
def hierarchical(
        queries: tf.Tensor,
        encoder_states: List[tf.Tensor],
        encoder_masks: List[tf.Tensor],
        heads: List[int],
        heads_hier: int,
        attention_dropout_callbacks: List[Callable[[tf.Tensor], tf.Tensor]],
        dropout_callback: Callable[[tf.Tensor], tf.Tensor]) -> tf.Tensor:
    # hierarchical:
    # * normalize queries
    # * attend to every encoder
    # * attend to the resulting context vectors (reuse normalized queries)
    # * apply dropout, add residual connection and return

    normalized_queries = layer_norm(queries)
    contexts = []

    batch = tf.shape(queries)[0]
    time_q = tf.shape(queries)[1]
    dimension = tf.shape(queries)[2]

    for i, (states, mask, n_heads, attn_drop_cb) in enumerate(zip(
            encoder_states, encoder_masks, heads,
            attention_dropout_callbacks)):

        with tf.variable_scope("enc_{}".format(i)):
            contexts.append(
                single(normalized_queries, states, mask, n_heads,
                       attention_dropout_callback=attn_drop_cb,
                       dropout_callback=dropout_callback,
                       normalize=False, residual=False))

    # context is of shape [batch, time(q), channels(v)],
    # stack to [batch, time(q), n_encoders, channels(v)]
    # reshape to [batch x time(q), n_encoders, channels(v)]
    stacked_contexts = tf.reshape(
        tf.stack(contexts, axis=2),
        [batch * time_q, len(encoder_states), dimension])

    # hierarchical mask: ones of shape [batch x time(q), n_encoders]
    hier_mask = tf.ones([batch * time_q, len(encoder_states)])

    # reshape queries to [batch x time(q), 1, channels(v)]
    reshaped_queries = tf.reshape(
        normalized_queries, [batch * time_q, 1, dimension])

    # returned shape [batch x time(q), 1, channels(v)]
    with tf.variable_scope("enc_hier"):
        # NOTE as attention dropout keep probability, we use the
        # dropout_keep_prob value instead of attention_dropout_keep_prob.
        encoder_context_stacked_batch = single(
            reshaped_queries, stacked_contexts, hier_mask, heads_hier,
            attention_dropout_callback=dropout_callback,
            dropout_callback=lambda x: x, normalize=False, use_dropout=False,
            residual=False)

        # reshape back to [batch, time(q), channels(v)]
        encoder_context = tf.reshape(
            encoder_context_stacked_batch, [batch, time_q, dimension])

        encoder_context = dropout_callback(encoder_context)

    return encoder_context + queries
# pylint: enable=too-many-locals


def flat(queries: tf.Tensor,
         encoder_states: List[tf.Tensor],
         encoder_masks: List[tf.Tensor],
         heads: int,
         attention_dropout_callback: Callable[[tf.Tensor], tf.Tensor],
         dropout_callback: Callable[[tf.Tensor], tf.Tensor]) -> tf.Tensor:

    # flat:
    # * concatenate the states and mask along the time axis
    # * run attention over the concatenation

    concat_states = tf.concat(encoder_states, 1)
    concat_mask = tf.concat(encoder_masks, 1)

    return single(queries, concat_states, concat_mask, heads,
                  attention_dropout_callback, dropout_callback)
