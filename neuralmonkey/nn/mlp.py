#tests: lint

import tensorflow as tf
from neuralmonkey.nn.projection import linear

class MultilayerPerceptron(object):
    """
    General implementation of the multilayer perceptron.
    """

    #pylint: disable=too-many-arguments
    def __init__(self, mlp_input, layer_configuration, dropout_plc,
                 output_size, name='multilayer_perceptron'):

        with tf.variable_scope(name):
            last_layer = mlp_input
            last_layer_size = mlp_input.get_shape()[1].value

            self.n_params = 0
            for i, size in enumerate(layer_configuration):
                last_layer = tf.tanh(
                    linear(last_layer, size,
                           scope="dense_layer_{}".format(i + 1)))
                last_layer = tf.nn.dropout(last_layer, dropout_plc)
                self.n_params += last_layer_size * size
                last_layer_size = size

            with tf.variable_scope("classification_layer"):
                self.n_params += last_layer_size * output_size
                w_out = tf.get_variable("W_out", shape=[last_layer_size, output_size])
                b_out = tf.Variable(tf.fill([output_size], 0.0), name="b_out")
                self.logits = tf.matmul(last_layer, w_out) + b_out


    @property
    def softmax(self):
        with tf.variable_scope("classification_layer"):
            return tf.nn.softmax(self.logits, name="decision_softmax")

    @property
    def classification(self):
        with tf.variable_scope("classification_layer"):
            return tf.argmax(self.logits, 1)
