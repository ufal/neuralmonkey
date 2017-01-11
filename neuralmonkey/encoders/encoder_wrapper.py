# tests: mypy

from abc import ABCMeta, abstractproperty
from typing import List, Union
import tensorflow as tf

from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.checking import assert_shape
from neuralmonkey.nn.projection import linear

class EncoderWrapper(Attentive):

    def __init__(self, name, encoders, multi_attention_type, attention_state_size, use_sentinels=False):
        self.name = name
        self.encoders = encoders
        self._multi_attention_type = multi_attention_type
        self._attention_state_size = attention_state_size
        self._use_sentinels = use_sentinels

        self.encoded = tf.concat(1, [e.encoded for e in encoders])

    def create_attention_object(self, runtime=False):
        return self._multi_attention_type(
            [e._attention_tensor for e in self.encoders],
            [e._attention_mask for e in self.encoders],
            self._attention_state_size,
            "attention_{}".format(self.name),
            use_sentinels=self._use_sentinels)

    def feed_dict(*args, **kwargs):
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

    def __init__(self,
                 encoders_tensors: List[tf.Tensor],
                 encoders_masks: List[tf.Tensor],
                 state_size: int,
                 scope: Union[tf.VariableScope, str],
                 **kwargs) -> None:
        self._encoders_tensors = encoders_tensors
        self._encoders_masks = encoders_masks
        self._state_size = state_size
        self._scope = scope
        self.attentions_in_time = []

        for e_m in self._encoders_masks:
            assert_shape(e_m, [None, -1])

        for e_t in self._encoders_tensors:
            assert_shape(e_t, [None, -1, -1])


    @abstractproperty
    def attention(self, decoder_state, decoder_input):
        raise NotImplementedError("Abstract method")

    @property
    def attn_size(self):
        return self._state_size


class FlatMultiAttention(MultiAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "use_sentinels" in kwargs:
            self._use_sentinels = kwargs.get("use_sentinels")

        with tf.variable_scope(self._scope):
            self.attn_v = tf.get_variable(
                "attn_v", [1, 1, self._state_size],
                initializer=tf.random_normal_initializer(stddev=.001))

            self.encoder_state_projections = []
            self.encoder_attn_biases = []

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
                assert_shape(projected_2d, [None, self._state_size])

                projection = tf.reshape(projected_2d, [encoder_tensor_shape[0],
                                                       encoder_tensor_shape[1],
                                                       self._state_size])

                self.encoder_state_projections.append(projection)

                attn_bias = tf.get_variable(
                    "attn_bias_{}".format(i), [],
                    initializer=tf.constant_initializer(0.0))

                self.encoder_attn_biases.append(attn_bias)

            if self._use_sentinels:
                self._encoders_masks.append(
                    tf.ones([tf.shape(self._encoders_masks[0])[0], 1]))

            self.masks_concat = tf.concat(1, self._encoders_masks)

    def attention(self, decoder_state, decoder_prev_state, decoder_input):

        with tf.variable_scope(self._scope):
            projected_state = linear(decoder_state, self._state_size)
            projected_state = tf.expand_dims(projected_state, 1)

            assert_shape(projected_state, [None, 1, self._state_size])

            logits = []

            for proj, bias in zip(self.encoder_state_projections,
                                  self.encoder_attn_biases):

                logits.append(tf.reduce_sum(
                    self.attn_v * tf.tanh(projected_state + proj), [2]) + bias)

            if self._use_sentinels:
                sentinel_value = self._sentinel(decoder_state,
                                                decoder_prev_state,
                                                decoder_input)

                sentinel_bias = tf.get_variable(
                    "sentinel_bias", [],
                    initializer=tf.constant_initializer(0.0))

                projected_sentinel = tf.expand_dims(
                    linear(sentinel_value, self._state_size,
                           scope="sentinel_projection"), 1)
                sentinel_logit = tf.reduce_sum(
                    self.attn_v *
                    tf.tanh(projected_state + projected_sentinel),
                    [2]) + sentinel_bias
                assert_shape(sentinel_logit, [None, 1])

                logits.append(sentinel_logit)

            logits_concat = tf.concat(1, logits)
            softmax_concat = tf.nn.softmax(logits_concat) * self.masks_concat
            norm = tf.reduce_sum(softmax_concat, 1, keep_dims=True) + 1e-8
            attentions = softmax_concat / norm

            self.attentions_in_time.append(attentions)

            projections_concat = tf.concat(
                1, self.encoder_state_projections +
                [projected_sentinel] if self._use_sentinels else [])

            contexts = tf.reduce_sum(
                tf.expand_dims(attentions, 2) * projections_concat, [1])

            return contexts


    def _sentinel(self, state, prev_state, input_):

        with tf.variable_scope("sentinel"):

            decoder_state_size = state.get_shape()[-1].value
            concatenation = tf.concat(1, [prev_state, input_])

            gate = tf.nn.sigmoid(linear(concatenation, decoder_state_size))
            sentinel_value = gate * state

            assert_shape(sentinel_value, [None, decoder_state_size])

            return sentinel_value


class HierarchicalMultiAttention(MultiAttention):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
