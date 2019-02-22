from typing import List, Dict, Callable

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.model.stateful import TemporalStateful
from neuralmonkey.model.feedable import FeedDict
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.model.sequence import EmbeddedSequence
from neuralmonkey.nn.utils import dropout
from neuralmonkey.vocabulary import Vocabulary, pad_batch, sentence_mask


class SequenceLabeler(ModelPart):
    """Classifier assing a label to each encoder's state."""

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 encoders: List[TemporalStateful],
                 vocabulary: Vocabulary,
                 data_id: str,
                 max_output_len: int,
                 hidden_dim: int = None,
                 activation: Callable = tf.nn.relu,
                 dropout_keep_prob: float = 1.0,
                 add_start_symbol: bool = False,
                 add_end_symbol: bool = False,
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
        self.add_start_symbol = add_start_symbol
        self.add_end_symbol = add_end_symbol
    # pylint: enable=too-many-arguments,too-many-locals

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
        return self.vocabulary.strings_to_indices(
            self.dataset[self.data_id])

    @tensor
    def train_mask(self) -> tf.Tensor:
        return sentence_mask(self.train_targets)

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
            # pylint: disable=redefined-variable-type
            states = dropout(hidden, self.dropout_keep_prob, self.train_mode)
            # pylint: enable=redefined-variable-type
        return states

    @tensor
    def logits(self) -> tf.Tensor:
        return tf.layers.dense(
            self.states, len(self.vocabulary), name="logits")

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
        return tf.reduce_sum(self.train_xents) / tf.reduce_sum(self.train_mask)

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
            fd[self.target_tokens] = pad_batch(
                list(sentences), self.max_output_len, self.add_start_symbol,
                self.add_end_symbol)

        return fd


class EmbeddingsLabeler(SequenceLabeler):
    """SequenceLabeler that uses an embedding matrix for output projection."""

    # pylint: disable=too-many-arguments,too-many-locals
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
                 add_start_symbol: bool = False,
                 add_end_symbol: bool = False,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:

        check_argument_types()
        SequenceLabeler.__init__(
            self, name, encoders, embedded_sequence.vocabulary, data_id,
            max_output_len, hidden_dim=hidden_dim, activation=activation,
            dropout_keep_prob=dropout_keep_prob,
            add_start_symbol=add_start_symbol, add_end_symbol=add_end_symbol,
            reuse=reuse, save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint, initializers=initializers)

        self.embedded_sequence = embedded_sequence
        self.train_embeddings = train_embeddings
    # pylint: enable=too-many-arguments,too-many-locals

    @tensor
    def logits(self) -> tf.Tensor:
        embeddings = self.embedded_sequence.embedding_matrix
        if not self.train_embeddings:
            embeddings = tf.stop_gradient(embeddings)

        states = self.states
        # pylint: disable=no-member
        states_dim = self.states.get_shape()[-1].value
        # pylint: enable=no-member
        embedding_dim = self.embedded_sequence.embedding_sizes[0]
        # pylint: disable=redefined-variable-type
        if states_dim != embedding_dim:
            states = tf.layers.dense(
                states, embedding_dim, name="project_for_embeddings")
        # pylint: enable=redefined-variable-type

        reshaped_states = tf.reshape(states, [-1, embedding_dim])
        reshaped_logits = tf.matmul(
            reshaped_states, embeddings, transpose_b=True, name="logits")
        return tf.reshape(
            reshaped_logits, [self.batch_size, -1, len(self.vocabulary)])
