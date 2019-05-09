from typing import Any, Optional, Union, Callable

import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.model.model_part import ModelPart, FeedDict, InitializerSpecs
from neuralmonkey.encoders.recurrent import RecurrentEncoder
from neuralmonkey.encoders.facebook_conv import SentenceEncoder
from neuralmonkey.decorators import tensor
from neuralmonkey.tf_utils import get_variable
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.nn.utils import dropout


class SequenceLabelerNew(ModelPart):
    """Classifier assing a label to each encoder's state."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoder: Any,
                 vocabulary: Vocabulary,
                 data_id: str,
                 hidden_dim: int = None,
                 activation: Callable = tf.nn.relu,
                 dropout_keep_prob: float = 1.0,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None,
                 initializers: InitializerSpecs = None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint,
                           initializers)

        self.encoder = encoder
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.dropout_keep_prob = dropout_keep_prob
        self.hidden_dim = hidden_dim
        self.activation = activation

    # pylint: enable=too-many-arguments

    # pylint: disable=no-self-use
    @tensor
    def train_targets(self) -> tf.Tensor:
        return tf.placeholder(tf.int32, shape=[None, None],
                              name="labeler_targets")

    @tensor
    def train_weights(self) -> tf.Tensor:
        return tf.placeholder(tf.float32, shape=[None, None],
                              name="labeler_padding_weights")

    @tensor
    def train_mode(self) -> tf.Tensor:
        return tf.placeholder(tf.bool, name="train_mode")
    # pylint: enable=no-self-use

    @tensor
    def concatenated_inputs(self) -> tf.Tensor:
        return self.encoder.temporal_states

    @tensor
    def states(self) -> tf.Tensor:
        if self.hidden_dim is None:
            return self.concatenated_inputs
        states = tf.layers.dense(
            self.concatenated_inputs, self.hidden_dim, self.activation,
            name="hidden_layer")
        return dropout(states, self.dropout_keep_prob, self.train_mode)

    @tensor
    def logprobs(self) -> tf.Tensor:
        return tf.nn.log_softmax(self.logits)

    @tensor
    def decoded(self) -> tf.Tensor:
        return tf.argmax(self.logits, 2)

    @tensor
    def train_xents(self) -> tf.Tensor:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.train_targets, logits=self.logits)

        # loss is now of shape [batch, time]. Need to mask it now by
        # element-wise multiplication with weights placeholder
        return loss * self.train_mask

    @tensor
    def cost(self) -> tf.Tensor:
        # Cross entropy mean over all words in the batch
        # (could also be done as a mean over sentences)
        return (tf.reduce_sum(self.train_xents)
                / (tf.reduce_sum(self.train_mask) + 1e-9))

    @property
    def train_loss(self) -> tf.Tensor:
        return self.cost

    @property
    def runtime_loss(self) -> tf.Tensor:
        return self.cost

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = {}  # type: FeedDict
        fd[self.train_mode] = train

        sentences = dataset.maybe_get_series(self.data_id)
        if sentences is not None:
            vectors, paddings = self.vocabulary.sentences_to_tensor(
                list(sentences), pad_to_max_len=False, train_mode=train)

            fd[self.train_targets] = vectors.T
            fd[self.train_weights] = paddings.T

        return fd

    @tensor
    def logits(self) -> tf.Tensor:
        return tf.layers.dense(
            self.states, len(self.vocabulary), name="logits")

    @tensor
    def input_mask(self) -> tf.Tensor:
        mask_main = self.encoders[0].temporal_mask

        asserts = [
            tf.assert_equal(
                mask_main, enc.temporal_mask,
                message=("Encoders '{}' and '{}' does not have equal temporal "
                         "masks.".format(str(self.encoders[0]), str(enc))))
            for enc in self.encoders[1:]]

        with tf.control_dependencies(asserts):
            return mask_main

    @tensor
    def train_mask(self) -> tf.Tensor:
        return self.train_weights
