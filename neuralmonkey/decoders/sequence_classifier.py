import tensorflow as tf

from neuralmonkey.nn.mlp import MultilayerPerceptron

# tests: lint, mypy

# pylint: disable=too-many-instance-attributes


class SequenceClassifier(object):
    """
    This is a implementation of a simple MLP classifier over encoders. The API
    pretends it is an RNN decoder which always generates a sequence of length
    exactly one.
    """
    # pylint: disable=dangerous-default-value

    def __init__(self, encoders, vocabulary, data_id, name,
                 layers=[], activation=tf.tanh, dropout_keep_prob=0.5):
        self.encoders = encoders
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.layers = layers
        self.activation = activation
        self.dropout_keep_prob = dropout_keep_prob
        self.name = name
        self.max_output_len = 1

        with tf.variable_scope(name):
            self.learning_step = tf.get_variable(
                "learning_step", [], trainable=False,
                initializer=tf.constant_initializer(0))

            self.dropout_placeholder = \
                tf.placeholder(tf.float32, name="dropout_plc")
            self.gt_inputs = [tf.placeholder(
                tf.int32, shape=[None], name="targets")]
            mlp_input = tf.concat(1, [enc.encoded for enc in encoders])
            mlp = MultilayerPerceptron(
                mlp_input, layers, self.dropout_placeholder, len(vocabulary))

            self.loss_with_gt_ins = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    mlp.logits, self.gt_inputs[0]))
            self.loss_with_decoded_ins = self.loss_with_gt_ins
            self.cost = self.loss_with_gt_ins

            self.decoded_seq = [mlp.classification]
            self.decoded_logits = [mlp.logits]

            tf.scalar_summary(
                'val_optimization_cost', self.cost,
                collections=["summary_val"])
            tf.scalar_summary(
                'train_optimization_cost',
                self.cost, collections=["summary_train"])

    @property
    def train_loss(self):
        return self.loss_with_gt_ins

    @property
    def runtime_loss(self):
        return self.loss_with_decoded_ins

    @property
    def decoded(self):
        return self.decoded_seq

    def feed_dict(self, dataset, train=False):
        sentences = dataset.get_series(self.data_id, allow_none=True)
        fd = {}

        label_tensors, _ = self.vocabulary.sentences_to_tensor(
            sentences, self.max_output_len)

        fd[self.gt_inputs[0]] = label_tensors[0]

        if train:
            fd[self.dropout_placeholder] = self.dropout_keep_prob
        else:
            fd[self.dropout_placeholder] = 1.0

        return fd
