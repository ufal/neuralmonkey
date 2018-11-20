"""Coverage attention introduced in Tu et al. (2016).

See arxiv.org/abs/1601.04811

The CoverageAttention class inherites from the basic feed-forward attention
introduced by Bahdanau et al. (2015)
"""
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.attention.base_attention import Attendable
from neuralmonkey.attention.feed_forward import Attention
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.model.parameterized import InitializerSpecs


class CoverageAttention(Attention):
    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoder: Attendable,
                 dropout_keep_prob: float = 1.0,
                 state_size: int = None,
                 max_fertility: int = 5,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        Attention.__init__(self, name, encoder, dropout_keep_prob, state_size,
                           reuse, save_checkpoint, load_checkpoint,
                           initializers)

        self.max_fertility = max_fertility

        self.coverage_weights = tf.get_variable(
            "coverage_matrix", [1, 1, 1, self.state_size])
        self.fertility_weights = tf.get_variable(
            "fertility_matrix", [1, 1, self.context_vector_size])

        self.fertility = 1e-8 + self.max_fertility * tf.sigmoid(
            tf.reduce_sum(self.fertility_weights * self.attention_states, [2]))
    # pylint: enable=too-many-arguments

    def get_energies(self, y: tf.Tensor, weights_in_time: tf.Tensor):
        weight_sum = tf.cond(
            tf.greater(weights_in_time.size(), 0),
            lambda: tf.reduce_sum(weights_in_time, axis=0),
            lambda: 0.0)

        coverage = weight_sum / self.fertility * self.attention_mask
        coverage_exp = tf.expand_dims(tf.expand_dims(coverage, -1), -1)
        logits = tf.reduce_sum(
            self.similarity_bias_vector * tf.tanh(
                self.hidden_features + y
                + self.coverage_weights * coverage_exp),
            [2, 3])

        return logits
