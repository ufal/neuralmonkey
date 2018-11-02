from typing import List, Callable

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.dataset import Dataset
from neuralmonkey.model.model_part import ModelPart, FeedDict, InitializerSpecs
from neuralmonkey.model.sequence import EmbeddedSequence
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
    def states(self) -> tf.Tensor:
        states = dropout(
            self.concatenated_inputs, self.dropout_keep_prob, self.train_mode)

        if self.hidden_dim is not None:
            hidden = tf.layers.dense(
                states, self.hidden_dim, self.activation,
                name="hidden_layer")
            states = dropout(hidden, self.dropout_keep_prob, self.train_mode)
        return states

    @tensor
    def logits(self) -> tf.Tensor:
        return tf.layers.dense(self.states, len(self.vocabulary), name="logits")

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


class EmbeddingsLabeler(SequenceLabeler):
    """SequenceLabeler that uses an embedding matrix for output projection."""
    def __init__(self,
                 name: str,
                 encoders: List[TemporalStateful],
                 embedded_sequence: EmbeddedSequence,
                 data_id: str,
                 max_output_len: int,
                 hidden_dim: int = None,
                 activation: Callable = tf.nn.relu,
                 train_embeddings: bool = True,
                 dropout_keep_prob: float = 1.0,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:

        check_argument_types()
        SequenceLabeler.__init__(self, name, encoders,
                 embedded_sequence.vocabulary, data_id, max_output_len,
                 hidden_dim=hidden_dim, activation=activation,
                 dropout_keep_prob=dropout_keep_prob,
                 save_checkpoint=save_checkpoint, load_checkpoint=load_checkpoint,
                 initializers=initializers)

        self.embedded_sequence = embedded_sequence
        self.train_embeddings = train_embeddings

    @tensor
    def logits(self) -> tf.Tensor:
        embeddings = self.embedded_sequence.embedding_matrix
        if not self.train_embeddings:
            embeddings = tf.stop_gradient(embeddings)

        states = self.states
        states_dim = self.states.get_shape()[-1].value
        embedding_dim = self.embedded_sequence.embedding_sizes[0]
        if states_dim != embedding_dim:
           states = tf.layers.dense(
                states, embedding_dim, name="project_for_embeddings")

        reshaped_states = tf.reshape(states, [-1, embedding_dim])
        reshaped_logits = tf.matmul(
            reshaped_states, embeddings, transpose_b=True, name="logits")
        return tf.reshape(
            reshaped_logits, [self.batch_size, -1, len(self.vocabulary)])
