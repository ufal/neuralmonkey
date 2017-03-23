import tensorflow as tf
from neuralmonkey.nn.projection import linear, multilayer_projection


class MultilayerPerceptron(object):
    """ General implementation of the multilayer perceptron. """

    # pylint: disable=too-many-arguments
    def __init__(self, mlp_input, layer_configuration, dropout_plc,
                 output_size, name: str='multilayer_perceptron',
                 activation_fn=tf.nn.relu) -> None:

        with tf.variable_scope(name):
            last_layer_size = mlp_input.get_shape()[1].value

            last_layer = multilayer_projection(mlp_input,
                                               layer_configuration,
                                               activation=activation_fn,
                                               dropout_plc=dropout_plc,
                                               scope="deep_output_mlp")
            self.n_params = 0
            for size in layer_configuration:
                self.n_params += last_layer_size * size
                last_layer_size = size

            with tf.variable_scope("classification_layer"):
                self.n_params += last_layer_size * output_size
                self.logits = linear(last_layer, output_size)

    @property
    def softmax(self):
        with tf.variable_scope("classification_layer"):
            return tf.nn.softmax(self.logits, name="decision_softmax")

    @property
    def classification(self):
        with tf.variable_scope("classification_layer"):
            return tf.argmax(self.logits, 1)
