from typing import Dict, Union

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.encoders.recurrent import RecurrentEncoder
from neuralmonkey.encoders.facebook_conv import SentenceEncoder
from neuralmonkey.model.feedable import FeedDict
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.tf_utils import get_variable
from neuralmonkey.vocabulary import Vocabulary, pad_batch, sentence_mask


class SequenceLabeler(ModelPart):
    """Classifier assing a label to each encoder's state."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoder: Union[RecurrentEncoder, SentenceEncoder],
                 vocabulary: Vocabulary,
                 data_id: str,
                 dropout_keep_prob: float = 1.0,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        ModelPart.__init__(self, name, reuse, save_checkpoint, load_checkpoint,
                           initializers)

        self.encoder = encoder
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.dropout_keep_prob = dropout_keep_prob
    # pylint: enable=too-many-arguments

    @property
    def input_types(self) -> Dict[str, tf.DType]:
        return {self.data_id: tf.string}

    @property
    def input_shapes(self) -> Dict[str, tf.TensorShape]:
        return {self.data_id: tf.TensorShape([None, None])}

    @tensor
    def target_tokens(self) -> tf.Tensor:
        return self.dataset[self.data_id]

    @tensor
    def train_targets(self) -> tf.Tensor:
        return self.vocabulary.strings_to_indices(self.target_tokens)

    @tensor
    def train_mask(self) -> tf.Tensor:
        return sentence_mask(self.train_targets)

    @property
    def rnn_size(self) -> int:
        return int(self.encoder.temporal_states.get_shape()[-1])

    @tensor
    def decoding_w(self) -> tf.Variable:
        return get_variable(
            name="state_to_word_W",
            shape=[self.rnn_size, len(self.vocabulary)])

    @tensor
    def decoding_b(self) -> tf.Variable:
        return get_variable(
            name="state_to_word_b",
            shape=[len(self.vocabulary)],
            initializer=tf.zeros_initializer())

    @tensor
    def decoding_residual_w(self) -> tf.Variable:
        input_dim = self.encoder.input_sequence.dimension
        return get_variable(
            name="emb_to_word_W",
            shape=[input_dim, len(self.vocabulary)])

    @tensor
    def logits(self) -> tf.Tensor:
        # To multiply 3-D matrix (encoder hidden states) by a 2-D matrix
        # (weights), we use 1-by-1 convolution (similar trick can be found in
        # attention computation)

        # TODO dropout needs to be revisited

        encoder_states = tf.expand_dims(self.encoder.temporal_states, 2)
        weights_4d = tf.expand_dims(tf.expand_dims(self.decoding_w, 0), 0)

        multiplication = tf.nn.conv2d(
            encoder_states, weights_4d, [1, 1, 1, 1], "SAME")
        multiplication_3d = tf.squeeze(multiplication, squeeze_dims=[2])

        biases_3d = tf.expand_dims(tf.expand_dims(self.decoding_b, 0), 0)

        embedded_inputs = tf.expand_dims(
            self.encoder.input_sequence.temporal_states, 2)
        dweights_4d = tf.expand_dims(
            tf.expand_dims(self.decoding_residual_w, 0), 0)

        dmultiplication = tf.nn.conv2d(
            embedded_inputs, dweights_4d, [1, 1, 1, 1], "SAME")
        dmultiplication_3d = tf.squeeze(dmultiplication, squeeze_dims=[2])

        logits = multiplication_3d + dmultiplication_3d + biases_3d
        return logits

    @tensor
    def logprobs(self) -> tf.Tensor:
        return tf.nn.log_softmax(self.logits)

    @tensor
    def decoded(self) -> tf.Tensor:
        return tf.argmax(self.logits, 2)

    @tensor
    def cost(self) -> tf.Tensor:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.train_targets, logits=self.logits)

        # loss is now of shape [batch, time]. Need to mask it now by
        # element-wise multiplication with weights placeholder
        weighted_loss = loss * self.train_mask
        return tf.reduce_sum(weighted_loss)

    @property
    def train_loss(self) -> tf.Tensor:
        return self.cost

    @property
    def runtime_loss(self) -> tf.Tensor:
        return self.cost

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = ModelPart.feed_dict(self, dataset, train)

        sentences = dataset.maybe_get_series(self.data_id)
        if sentences is not None:
            fd[self.target_tokens] = pad_batch(list(sentences))

        return fd
