from typing import List, Callable
import tensorflow as tf
from neuralmonkey.nn.projection import multilayer_projection


class MultilayerPerceptron:
    """General implementation of the multilayer perceptron."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 mlp_input: tf.Tensor,
                 layer_configuration: List[int],
                 dropout_keep_prob: float,
                 output_size: int,
                 train_mode: tf.Tensor,
                 activation_fn: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
                 name: str = "multilayer_perceptron") -> None:

        with tf.variable_scope(name):
            last_layer = multilayer_projection(
                mlp_input, layer_configuration, activation=activation_fn,
                dropout_keep_prob=dropout_keep_prob, train_mode=train_mode,
                scope="deep_output_mlp")

            self.logits = tf.layers.dense(
                last_layer, output_size, name="classification_layer")

    @property
    def softmax(self):
        with tf.variable_scope("classification_layer"):
            return tf.nn.softmax(self.logits, name="decision_softmax")

    @property
    def classification(self):
        with tf.variable_scope("classification_layer"):
            return tf.argmax(self.logits, 1)
