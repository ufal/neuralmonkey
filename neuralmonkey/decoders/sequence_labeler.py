from typing import cast, Iterable, List, Optional

import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.encoders.sentence_encoder import SentenceEncoder
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.decorators import tensor


class SequenceLabeler(ModelPart):
    """Classifier assing a label to each encoder's state."""

    def __init__(self,
                 name: str,
                 encoder: SentenceEncoder,
                 vocabulary: Vocabulary,
                 data_id: str,
                 dropout_keep_prob: float = 1.0,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        self.encoder = encoder
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.dropout_keep_prob = dropout_keep_prob

        self.rnn_size = self.encoder.rnn_size * 2
        self.max_output_len = self.encoder.max_input_len

        self.train_targets = tf.placeholder(tf.int32, shape=[None, None],
                                            name="labeler_targets")

        self.train_weights = tf.placeholder(tf.float32, shape=[None, None],
                                            name="labeler_padding_weights")

        self.train_mode = tf.placeholder(tf.bool, name="train_mode")

    @property
    def train_loss(self) -> tf.Tensor:
        return self.cost

    @property
    def runtime_loss(self) -> tf.Tensor:
        return self.cost

    @tensor
    def cost(self) -> tf.Tensor:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.train_targets, logits=self.logits)

        # loss is now of shape [batch, time]. Need to mask it now by
        # element-wise multiplication with weights placeholder
        weighted_loss = loss * self.train_weights
        return tf.reduce_sum(weighted_loss)

    @tensor
    def decoded(self) -> tf.Tensor:
        # [:, :, 1:] -- bans generating the PAD symbol (index 0 in
        #            the vocabulary;
        #
        # tf.argmax(l[:, :, 1:], 2) -- argmax along the vocabulary dim
        #
        # +1 -- because the [:, :, 1:] removed a symbol from argmax
        #       consideration, we need to compensate for the shortened array.

        # pylint: disable=unsubscriptable-object
        return tf.argmax(self.logits[:, :, 1:], 2) + 1
        # pylint: enable=unsubscriptable-object

    @tensor
    def logprobs(self) -> tf.Tensor:
        return tf.nn.log_softmax(self.logits)

    @tensor
    def logits(self) -> tf.Tensor:
        vocabulary_size = len(self.vocabulary)

        weights = tf.get_variable(
            name="state_to_word_W",
            shape=[self.rnn_size, vocabulary_size],
            initializer=tf.random_uniform_initializer(-0.5, 0.5))

        biases = tf.get_variable(
            name="state_to_word_b",
            shape=[vocabulary_size],
            initializer=tf.zeros_initializer())

        weights_direct = tf.get_variable(
            name="emb_to_word_W",
            shape=[self.encoder.embedding_size, vocabulary_size],
            initializer=tf.random_uniform_initializer(-0.5, 0.5))

        # To multiply 3-D matrix (encoder hidden states) by a 2-D matrix
        # (weights), we use 1-by-1 convolution (similar trick can be found in
        # attention computation)

        encoder_states = tf.expand_dims(self.encoder.hidden_states, 2)
        weights_4d = tf.expand_dims(tf.expand_dims(weights, 0), 0)

        multiplication = tf.nn.conv2d(
            encoder_states, weights_4d, [1, 1, 1, 1], "SAME")
        multiplication_3d = tf.squeeze(multiplication, squeeze_dims=[2])

        biases_3d = tf.expand_dims(tf.expand_dims(biases, 0), 0)

        embedded_inputs = tf.expand_dims(self.encoder.embedded_inputs, 2)
        dweights_4d = tf.expand_dims(tf.expand_dims(weights_direct, 0), 0)

        dmultiplication = tf.nn.conv2d(
            embedded_inputs, dweights_4d, [1, 1, 1, 1], "SAME")
        dmultiplication_3d = tf.squeeze(dmultiplication, squeeze_dims=[2])

        logits = tf.tanh(multiplication_3d + dmultiplication_3d + biases_3d)
        return logits

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = {}  # type: FeedDict

        sentences = cast(Iterable[List[str]],
                         dataset.get_series(self.data_id, allow_none=True))

        fd[self.train_mode] = train

        if sentences is not None:
            vectors, paddings = self.vocabulary.sentences_to_tensor(
                list(sentences), self.max_output_len, pad_to_max_len=False,
                train_mode=train)

            fd[self.train_targets] = vectors.T
            fd[self.train_weights] = paddings.T

        return fd
