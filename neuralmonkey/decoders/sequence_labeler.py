from typing import List, Callable

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.dataset import Dataset
from neuralmonkey.model.model_part import ModelPart, FeedDict, InitializerSpecs
from neuralmonkey.model.stateful import TemporalStateful
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.decorators import tensor
from neuralmonkey.nn.utils import dropout


class SequenceLabeler(ModelPart):
    """Classifier assing a label to each encoder's state."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoders: List[TemporalStateful],
                 vocabulary: Vocabulary,
                 data_id: str,
                 max_output_len: int,
                 hidden_dim: int = None,
                 activation: Callable = tf.nn.relu,
                 dropout_keep_prob: float = 1.0,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        ModelPart.__init__(self, name, reuse, save_checkpoint, load_checkpoint,
                           initializers)

        self.encoders = encoders
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.max_output_len = max_output_len
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout_keep_prob = dropout_keep_prob

        assert self.encoders

        self.input_dim = sum(inp.dimension for inp in self.encoders)

        with self.use_scope():
            self.train_targets = tf.placeholder(
                tf.int32, [None, None], "labeler_targets")
            self.train_weights = tf.placeholder(
                tf.float32, [None, None], "labeler_padding_weights")
    # pylint: enable=too-many-arguments

    @tensor
    def concatenated_inputs(self) -> tf.Tensor:
        return tf.concat(
            [inp.temporal_states for inp in self.encoders], axis=2)

    @tensor
    def logits(self) -> tf.Tensor:

        state = dropout(
            self.concatenated_inputs, self.dropout_keep_prob, self.train_mode)

        if self.hidden_dim is not None:
            hidden = tf.layers.dense(
                state, self.hidden_dim, self.activation,
                name="hidden_layer")
            state = dropout(hidden, self.dropout_keep_prob, self.train_mode)

        return tf.layers.dense(state, len(self.vocabulary), name="logits")

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
        weighted_loss = loss * self.train_weights
        return tf.reduce_sum(weighted_loss) / tf.reduce_sum(self.train_weights)

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
            vectors, paddings = self.vocabulary.sentences_to_tensor(
                list(sentences), max_len=self.max_output_len,
                pad_to_max_len=False, train_mode=train)

            fd[self.train_targets] = vectors.T
            fd[self.train_weights] = paddings.T

        return fd
