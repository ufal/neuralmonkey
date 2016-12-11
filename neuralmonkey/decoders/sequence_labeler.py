import tensorflow as tf

from neuralmonkey.decoders.decoder import Decoder

# tests: lint, mypy

# pylint: disable=too-many-instance-attributes


class SequenceLabeler(Decoder):
    """Classifier assing a label to each encoder's state."""

    # pylint: disable=super-init-not-called
    def __init__(self, encoder, vocabulary, data_id, **kwargs):

        self.encoder = encoder
        self.vocabulary = vocabulary
        self.data_id = data_id

        dropout_keep_prob = kwargs.get("dropout_keep_prob", 1.0)

        self.rnn_size = self.encoder.rnn_size * 2
        self.max_output_len = self.encoder.max_input_len

        # pylint: disable=no-member
        self.weights, self.biases = self._state_to_output()

        logits = [tf.tanh(tf.matmul(state, self.weights) + self.biases)
                  for state in self.encoder.outputs_bidi]

        # [:, 1:] -- bans generating the start symbol (index 0 in
        #            the vocabulary; The start symbol is automatically
        #            prepended in the sentences_to_tensor()).
        #
        # tf.argmax(l[:, 1:], 1) -- argmax along the vertical dimension
        #
        # +1 -- because the [:, 1:] removed a symbol from argmax consideration,
        #       we need to compensate for the shortened array.
        self.decoded = [tf.argmax(l[:, 1:], 1) + 1 for l in logits]

        self.train_targets = [tf.placeholder(tf.int64, [None],
                                             name="seq_lab_{}".format(i))
                              for i in range(self.max_output_len)]

        train_onehots = [tf.one_hot(t, self.vocabulary_size)
                         for t in self.train_targets]

        self.train_weights = [
            tf.placeholder(tf.float32, [None],
                           name="seq_lab_padding_weights_{}".format(i))
            for i in range(self.max_output_len)]

        losses = [tf.nn.softmax_cross_entropy_with_logits(l, t)
                  for l, t in zip(logits, train_onehots)]

        weighted_losses = [l * w for l, w in zip(losses, self.train_weights)]

        summed_losses_in_time = [tf.reduce_sum(l) for l in weighted_losses]

        self.train_loss = sum(summed_losses_in_time)
        self.runtime_loss = self.train_loss

        # Learning step
        # TODO was here only because of scheduled sampling.
        # needs to be refactored out
        self.learning_step = tf.get_variable(
            "learning_step", [], initializer=tf.constant_initializer(0),
            trainable=False)

        self.dropout_placeholder = tf.placeholder_with_default(
            tf.constant(dropout_keep_prob, tf.float32),
            shape=[], name="decoder_dropout_placeholder")

    def feed_dict(self, dataset, train=False):

        fd = {}

        sentences = dataset.get_series(self.data_id, allow_none=True)

        if sentences is not None:
            inputs, weights = self.vocabulary.sentences_to_tensor(
                sentences, self.max_output_len)

            assert len(weights) == len(self.train_weights)
            assert len(inputs) == len(self.train_targets)

            for placeholder, weight in zip(self.train_weights, weights):
                fd[placeholder] = weight

            for placeholder, tensor in zip(self.train_targets, inputs):
                fd[placeholder] = tensor

        if not train:
            fd[self.dropout_placeholder] = 1.0

        return fd
