from typing import Optional, Tuple

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.attention.base_attention import (
    BaseAttention, AttentionLoopStateTA, empty_attention_loop_state)
from neuralmonkey.model.stateful import Stateful
from neuralmonkey.decorators import tensor
from neuralmonkey.model.model_part import InitializerSpecs


class StatefulContext(BaseAttention):
    """Provides a `Stateful` encoder's output as context to a decoder.

    This is not really an attention mechanism, but rather a hack which
    (mis)uses the attention interface to provide a "static" context vector to
    the decoder cell. In other words, the context vector is the same for all
    positions in the sequence and doesn't depend on the query vector.

    To use this, simply pass an instance of this class to the decoder using
    the `attentions` parameter.
    """

    def __init__(self,
                 name: str,
                 encoder: Stateful,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        BaseAttention.__init__(self, name, save_checkpoint, load_checkpoint,
                               initializers)

        self.encoder = encoder

    @tensor
    def attention_states(self) -> tf.Tensor:
        return tf.expand_dims(self.encoder.output, 1)

    # pylint: disable=no-self-use
    @tensor
    def attention_mask(self) -> Optional[tf.Tensor]:
        return None
    # pylint: enable=no-self-use

    # pylint: disable=no-member
    # Pylint fault from resolving tensor decoration
    @property
    def context_vector_size(self) -> int:
        return self.attention_states.get_shape()[2].value
    # pylint: enable=no-member

    @property
    def state_size(self) -> int:
        return self.context_vector_size

    def attention(self,
                  query: tf.Tensor,
                  decoder_prev_state: tf.Tensor,
                  decoder_input: tf.Tensor,
                  loop_state: AttentionLoopStateTA,
                  step: tf.Tensor) -> Tuple[tf.Tensor, AttentionLoopStateTA]:
        context = tf.reshape(self.attention_states,
                             [-1, self.context_vector_size])
        weights = tf.ones(shape=[tf.shape(context)[0]])

        next_loop_state = AttentionLoopStateTA(
            contexts=loop_state.contexts.write(step, context),
            weights=loop_state.weights.write(step, weights))

        return context, next_loop_state

    def initial_loop_state(self) -> AttentionLoopStateTA:
        return empty_attention_loop_state()

    def finalize_loop(self, key: str,
                      last_loop_state: AttentionLoopStateTA) -> None:
        pass

    def visualize_attention(self, key: str) -> None:
        pass
