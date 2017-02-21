import math
from typing import cast, Iterable, List, Optional

import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.encoders.sentence_encoder import SentenceEncoder
from neuralmonkey.vocabulary import Vocabulary


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class SequenceLabeler(ModelPart):
    """Classifier assing a label to each encoder's state."""

    # pylint: disable=too-many-locals
    def __init__(self,
                 name: str,
                 encoder: SentenceEncoder,
                 vocabulary: Vocabulary,
                 data_id: str,
                 dropout_keep_prob: float=1.0,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:

        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        self.encoder = encoder
        self.vocabulary = vocabulary
        self.data_id = data_id

        self.rnn_size = self.encoder.rnn_size * 2
        self.max_output_len = self.encoder.max_input_len
        vocabulary_size = len(vocabulary)

        weights = tf.get_variable(
            name="state_to_word_W",
            shape=[self.rnn_size, vocabulary_size],
            initializer=tf.random_uniform_initializer(-0.5, 0.5))

        biases = tf.get_variable(
            name="state_to_word_b",
            shape=[vocabulary_size],
            initializer=tf.constant_initializer(- math.log(vocabulary_size)))

        logits = [tf.tanh(tf.matmul(state, weights) + biases)
                  for state in tf.unpack(self.encoder.hidden_states, axis=1)]

        self.runtime_logprobs = [tf.nn.log_softmax(l) for l in logits]

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

        train_onehots = [tf.one_hot(t, len(vocabulary))
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
        self.cost = self.train_loss

        self.dropout_placeholder = tf.placeholder_with_default(
            tf.constant(dropout_keep_prob, tf.float32),
            shape=[], name="decoder_dropout_placeholder")

    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        fd = {}  # type: FeedDict

        sentences = cast(Iterable[List[str]],
                         dataset.get_series(self.data_id, allow_none=True))

        if sentences is not None:
            sentences_list = list(sentences) if sentences is not None else None
            inputs, weights = self.vocabulary.sentences_to_tensor(
                sentences_list, self.max_output_len)

            assert len(weights) == len(self.train_weights)
            assert len(inputs) == len(self.train_targets)

            for placeholder, weight in zip(self.train_weights, weights):
                fd[placeholder] = weight

            for placeholder, tensor in zip(self.train_targets, inputs):
                fd[placeholder] = tensor

        if not train:
            fd[self.dropout_placeholder] = 1.0

        return fd
