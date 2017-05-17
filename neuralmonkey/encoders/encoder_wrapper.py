"""Attention combination state gores.

This modules implements attention combination strategies for multi-encoder
scenario when we may want to combine the hidden states of the encoders in
more complicated fashion.

Currently there are two attention combination strategies flat and hierarchical
(see paper `Attention Combination Strategies for Multi-Source
Sequence-to-Sequence Learning <https://arxiv.org/pdf/1704.06567.pdf>`_).

The combination strategies may use the sentinel mechanism which allows the
decoder not to attend to the, and extract information on its own hidden state
(see paper `Knowing when to Look: Adaptive Attention via a Visual Sentinel for
Image Captioning  <https://arxiv.org/pdf/1612.01887.pdf>`_).
"""
from abc import ABCMeta
from typing import Any, List, Union, Type
import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.checking import assert_shape
from neuralmonkey.nn.projection import linear


class EncoderWrapper(ModelPart, Attentive):
    """Wrapper doing attention combination behaving as a single encoder.

    This class wraps encoders and performs the attention combination in such a
    way that for the decoder, it looks like a single encoder capable to
    generate a single context vector.
    """

    def __init__(self,
                 name: str,
                 encoders: List[Any],
                 attention_type: Type,
                 attention_state_size: int,
                 use_sentinels=False,
                 share_attn_projections=False) -> None:
        """Initializes the encoder wrapper.

        Args:
            name: Name of the encoder / its scope.
            encoders: List of encoders to be wrapped.
            attention_type: Type of the attention combination.
            attention_state_size: Dimension of the state projection of
                attention energy computation.
            use_sentinels: Flag whether the sentinel mechanism should be added
                to the attention combination.
            share_attn_projections: Flag whether the hidden state projection
                should be shared for the both the energies computation and
                context vector computation.
        """

        ModelPart.__init__(self, name, None, None)
        Attentive.__init__(self, attention_type)
        self.encoders = encoders
        self._attention_type = attention_type
        self._attention_state_size = attention_state_size
        self._use_sentinels = use_sentinels
        self._share_attn_projections = share_attn_projections

        self.encoded = tf.concat([e.encoded for e in encoders], 1)

    def create_attention_object(self):
        return self._attention_type(
            self.encoders,
            self._attention_state_size,
            "attention_{}".format(self.name),
            use_sentinels=self._use_sentinels,
            share_projections=self._share_attn_projections)

    def feed_dict(self, dataset: Dataset, train: bool) -> FeedDict:
        return {}

    @property
    def _attention_tensor(self):
        raise NotImplementedError("Encoder wrapper does not contain the"
                                  " attention tensor")

    @property
    def _attention_mask(self):
        raise NotImplementedError("Encoder wrapper does not contain the"
                                  " attention mask")


class MultiAttention(metaclass=ABCMeta):
    """Base class for attention combination."""

    # pylint: disable=unused-argument
    def __init__(self,
                 encoders: List[Attentive],
                 state_size: int,
                 scope: Union[tf.VariableScope, str],
                 share_projections: bool = False,
                 use_sentinels: bool = False) -> None:
        self._encoders = encoders
        self._state_size = state_size
        self._scope = scope
        self.attentions_in_time = []  # type: List[tf.Tensor]
        self._share_projections = share_projections
        self._use_sentinels = use_sentinels

        with tf.variable_scope(self._scope):
            self.attn_v = tf.get_variable(
                "attn_v", [1, 1, self._state_size],
                initializer=tf.random_normal_initializer(stddev=.001))
    # pylint: enable=unused-argument

    def attention(self, decoder_state, decoder_prev_state, decoder_input):
        """Get context vector for given decoder state."""
        raise NotImplementedError("Abstract method")

    @property
    def attn_size(self):
        return self._state_size

    def _vector_logit(self,
                      projected_decoder_state: tf.Tensor,
                      vector_value: tf.Tensor,
                      scope: str) -> tf.Tensor:
        """Get logit for a single vector, e.g., sentinel vector."""
        assert_shape(projected_decoder_state, [-1, 1, -1])
        assert_shape(vector_value, [-1, -1])

        with tf.variable_scope("{}_logit".format(scope)):
            vector_bias = tf.get_variable(
                "vector_bias", [],
                initializer=tf.constant_initializer(0.0))

            proj_vector_for_logit = tf.expand_dims(
                linear(vector_value, self._state_size,
                       scope="vector_projection"), 1)

            if self._share_projections:
                proj_vector_for_ctx = proj_vector_for_logit
            else:
                proj_vector_for_ctx = tf.expand_dims(
                    linear(vector_value, self._state_size,
                           scope="vector_ctx_proj"), 1)

            vector_logit = tf.reduce_sum(
                self.attn_v *
                tf.tanh(projected_decoder_state + proj_vector_for_logit),
                [2]) + vector_bias
            assert_shape(vector_logit, [-1, 1])
            return proj_vector_for_ctx, vector_logit


class FlatMultiAttention(MultiAttention):
    """Flat attention combination strategy.

    Using this attention combination strategy, hidden states of the encoders
    are first projected to the same space (different projection for different
    encoders) and then we compute a joint distribution over all the hidden
    states. The context vector is then a weighted sum of another / then
    projection of the encoders hidden states. The sentinel vector can be added
    as an additional hidden state.

    See equations 8 to 10 in the Attention Combination Strategies paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # pylint: disable=protected-access
        self._encoders_tensors = [e._attention_tensor for e in self._encoders]
        self._encoders_masks = [e._attention_mask for e in self._encoders]
        # pylint: enable=protected-access

        for e_m in self._encoders_masks:
            assert_shape(e_m, [-1, -1])

        for e_t in self._encoders_tensors:
            assert_shape(e_t, [-1, -1, -1])

        with tf.variable_scope(self._scope):
            self.encoder_projections_for_logits = \
                self.get_encoder_projections("logits_projections")

            self.encoder_attn_biases = [
                tf.get_variable(name="attn_bias_{}".format(i),
                                shape=[],
                                initializer=tf.constant_initializer(0.))
                for i in range(len(self._encoders_tensors))]

            if self._share_projections:
                self.encoder_projections_for_ctx = \
                    self.encoder_projections_for_logits
            else:
                self.encoder_projections_for_ctx = \
                    self.get_encoder_projections("context_projections")

            if self._use_sentinels:
                self._encoders_masks.append(
                    tf.ones([tf.shape(self._encoders_masks[0])[0], 1]))

            self.masks_concat = tf.concat(self._encoders_masks, 1)

    def get_encoder_projections(self, scope):
        encoder_projections = []
        with tf.variable_scope(scope):
            for i, encoder_tensor in enumerate(self._encoders_tensors):
                encoder_state_size = encoder_tensor.get_shape()[2].value
                encoder_tensor_shape = tf.shape(encoder_tensor)

                proj_matrix = tf.get_variable(
                    "proj_matrix_{}".format(i),
                    [encoder_state_size, self._state_size],
                    initializer=tf.random_normal_initializer(stddev=0.001))

                proj_bias = tf.get_variable(
                    "proj_bias_{}".format(i),
                    initializer=tf.zeros_initializer([self._state_size]))

                encoder_tensor_2d = tf.reshape(
                    encoder_tensor, [-1, encoder_state_size])

                projected_2d = tf.matmul(
                    encoder_tensor_2d, proj_matrix) + proj_bias
                assert_shape(projected_2d, [-1, self._state_size])

                projection = tf.reshape(projected_2d, [encoder_tensor_shape[0],
                                                       encoder_tensor_shape[1],
                                                       self._state_size])

                encoder_projections.append(projection)
            return encoder_projections

    def attention(self, decoder_state, decoder_prev_state, decoder_input):
        with tf.variable_scope(self._scope):
            projected_state = linear(decoder_state, self._state_size)
            projected_state = tf.expand_dims(projected_state, 1)

            assert_shape(projected_state, [-1, 1, self._state_size])

            logits = []

            for proj, bias in zip(self.encoder_projections_for_logits,
                                  self.encoder_attn_biases):

                logits.append(tf.reduce_sum(
                    self.attn_v * tf.tanh(projected_state + proj), [2]) + bias)

            if self._use_sentinels:
                sentinel_value = _sentinel(decoder_state,
                                           decoder_prev_state,
                                           decoder_input)
                projected_sentinel, sentinel_logit = self._vector_logit(
                    projected_state, sentinel_value, scope="sentinel")
                logits.append(sentinel_logit)

            attentions = self._renorm_softmax(tf.concat(1, logits))

            self.attentions_in_time.append(attentions)

            projections_concat = tf.concat(
                1, self.encoder_projections_for_ctx +
                ([projected_sentinel] if self._use_sentinels else []))

            contexts = tf.reduce_sum(
                tf.expand_dims(attentions, 2) * projections_concat, [1])

            return contexts

    def _renorm_softmax(self, logits):
        """Renormalized softmax wrt. attention mask."""
        softmax_concat = tf.nn.softmax(logits) * self.masks_concat
        norm = tf.reduce_sum(softmax_concat, 1, keep_dims=True) + 1e-8
        attentions = softmax_concat / norm

        return attentions


def _sentinel(state, prev_state, input_):
    """Sentinel value given the decoder state."""
    with tf.variable_scope("sentinel"):

        decoder_state_size = state.get_shape()[-1].value
        concatenation = tf.concat([prev_state, input_], 1)

        gate = tf.nn.sigmoid(linear(concatenation, decoder_state_size))
        sentinel_value = gate * state

        assert_shape(sentinel_value, [-1, decoder_state_size])

        return sentinel_value


class HierarchicalMultiAttention(MultiAttention):
    """Hierarchical attention combination.

    Hierarchical attention combination strategy first computes the context
    vector for each encoder separately using whatever attention type the
    encoders have. After that it computes a second attention over the resulting
    context vectors and optionally the sentinel vector.

    See equations 6 and 7 in the Attention Combination Strategies paper.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with tf.variable_scope(self._scope):
            self._attn_objs = [
                e.create_attention_object() for e in self._encoders]

    def attention(self, decoder_state, decoder_prev_state, decoder_input):
        with tf.variable_scope(self._scope):
            projected_state = linear(decoder_state, self._state_size)
            projected_state = tf.expand_dims(projected_state, 1)

            assert_shape(projected_state, [-1, 1, self._state_size])
            attn_ctx_vectors = [
                a.attention(decoder_state, decoder_prev_state, decoder_input)
                for a in self._attn_objs]

            proj_ctxs, attn_logits = [list(t) for t in zip(*[
                self._vector_logit(projected_state, ctx_vec, scope=enc.name)
                for ctx_vec, enc in zip(attn_ctx_vectors, self._encoders)])]

            if self._use_sentinels:
                sentinel_value = _sentinel(decoder_state,
                                           decoder_prev_state,
                                           decoder_input)
                proj_sentinel, sentinel_logit = self._vector_logit(
                    projected_state, sentinel_value, scope="sentinel")
                proj_ctxs.append(proj_sentinel)
                attn_logits.append(sentinel_logit)

            attention_distr = tf.nn.softmax(tf.concat(attn_logits, 1))
            self.attentions_in_time.append(attention_distr)

            if self._share_projections:
                output_cxts = proj_ctxs
            else:
                output_cxts = [
                    tf.expand_dims(
                        linear(ctx_vec, self._state_size,
                               scope="proj_attn_{}".format(enc.name)), 1)
                    for ctx_vec, enc in zip(attn_ctx_vectors, self._encoders)]
                if self._use_sentinels:
                    output_cxts.append(tf.expand_dims(
                        linear(sentinel_value, self._state_size,
                               scope="proj_sentinel"), 1))

            projections_concat = tf.concat(output_cxts, 1)
            context = tf.reduce_sum(
                tf.expand_dims(attention_distr, 2) * projections_concat, [1])

            return context
