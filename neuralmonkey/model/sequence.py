"""Module which impements the sequence class and a few of its subclasses."""

import os
from typing import List, Union

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from typeguard import check_argument_types

from neuralmonkey.model.model_part import ModelPart, FeedDict, InitializerSpecs
from neuralmonkey.model.stateful import TemporalStateful
from neuralmonkey.vocabulary import Vocabulary, CharacterVocabulary
from neuralmonkey.decorators import tensor
from neuralmonkey.dataset import Dataset
from neuralmonkey.tf_utils import get_variable
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell, NematusGRUCell

RNN_CELL_TYPES = {
    "NematusGRU": NematusGRUCell,
    "GRU": OrthoGRUCell,
    "LSTM": tf.contrib.rnn.LSTMCell
}


# pylint: disable=abstract-method
class Sequence(ModelPart, TemporalStateful):
    """Base class for a data sequence.

    This abstract class represents a batch of sequences of Tensors of possibly
    different lengths.

    Sequence is essentialy a temporal stateful object whose states and mask
    are fed, or computed from fed values. It is also a ModelPart, and
    therefore, it can store variables such as embedding matrices.
    """

    def __init__(self,
                 name: str,
                 max_length: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Construct a new `Sequence` object.

        Arguments:
            name: The name for the `ModelPart` object
            max_length: Maximum length of sequences in the object (not checked)
            save_checkpoint: The save_checkpoint parameter for `ModelPart`
            load_checkpoint: The load_checkpoint parameter for `ModelPart`
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint,
                           initializers)

        self.max_length = max_length
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError("Max sequence length must be a positive integer.")
# pylint: enable=abstract-method


class EmbeddedFactorSequence(Sequence):
    """A sequence that stores one or more embedded inputs (factors)."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 vocabularies: List[Union[Vocabulary, CharacterVocabulary]],
                 data_ids: List[str],
                 embedding_sizes: List[int],
                 max_length: int = None,
                 add_start_symbol: bool = False,
                 add_end_symbol: bool = False,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Construct a new instance of `EmbeddedFactorSequence`.

        Takes three lists of vocabularies, data series IDs, and embedding
        sizes and construct a `Sequence` object. The supplied lists must be
        equal in length and the indices to these lists must correspond
        to each other

        Arguments:
            name: The name for the `ModelPart` object
            vocabularies: A list of `Vocabulary` objects used for each factor
            data_ids: A list of strings identifying the data series used for
                each factor
            embedding_sizes: A list of integers specifying the size of the
                embedding vector for each factor
            max_length: The maximum length of the sequences
            add_start_symbol: Includes <s> in the sequence
            add_end_symbol: Includes </s> in the sequence
            save_checkpoint: The save_checkpoint parameter for `ModelPart`
            load_checkpoint: The load_checkpoint parameter for `ModelPart`
        """
        check_argument_types()
        Sequence.__init__(
            self, name, max_length, save_checkpoint, load_checkpoint,
            initializers)

        self.vocabularies = vocabularies
        self.vocabulary_sizes = [len(vocab) for vocab in self.vocabularies]
        self.data_ids = data_ids
        self.embedding_sizes = embedding_sizes
        self.add_start_symbol = add_start_symbol
        self.add_end_symbol = add_end_symbol

        if not (len(self.data_ids)
                == len(self.vocabularies)
                == len(self.embedding_sizes)):
            raise ValueError("data_ids, vocabularies, and embedding_sizes "
                             "lists need to have the same length")

        if any([esize <= 0 for esize in self.embedding_sizes]):
            raise ValueError("Embedding size must be a positive integer.")

        with self.use_scope():
            self.mask = tf.placeholder(tf.float32, [None, None], "mask")
            self.input_factors = [
                tf.placeholder(tf.int32, [None, None], "factor_{}".format(did))
                for did in self.data_ids]
    # pylint: enable=too-many-arguments

    # TODO this should be placed into the abstract embedding class
    def tb_embedding_visualization(self, logdir: str,
                                   prj: projector):
        """Link embeddings with vocabulary wordlist.

        Used for tensorboard visualization.

        Arguments:
            logdir: directory where model is stored
            projector: TensorBoard projector for storing linking info.
        """
        for i in range(len(self.vocabularies)):
            # the overriding is turned to true, because if the model would not
            # be allowed to override the output folder it would failed earlier.
            # TODO when vocabularies will have name parameter, change it
            wordlist = os.path.join(logdir, self.name + "_" + str(i) + ".tsv")
            self.vocabularies[i].save_wordlist(wordlist, True, True)

            embedding = prj.embeddings.add()
            # pylint: disable=unsubscriptable-object
            embedding.tensor_name = self.embedding_matrices[i].name
            embedding.metadata_path = wordlist

    @tensor
    def embedding_matrices(self) -> List[tf.Tensor]:
        """Return a list of embedding matrices for each factor."""

        # Note: Embedding matrices are numbered rather than named by the data
        # id so the data_id string does not need to be the same across
        # experiments

        return [
            get_variable(
                name="embedding_matrix_{}".format(i),
                shape=[vocab_size, emb_size],
                initializer=tf.glorot_uniform_initializer())
            for i, (data_id, vocab_size, emb_size) in enumerate(zip(
                self.data_ids, self.vocabulary_sizes, self.embedding_sizes))]

    @tensor
    def temporal_states(self) -> tf.Tensor:
        """Return the embedded factors.

        A 3D Tensor of shape (batch, time, dimension),
        where dimension is the sum of the embedding sizes supplied to the
        constructor.
        """
        embedded_factors = [
            tf.nn.embedding_lookup(embedding_matrix, factor)
            for factor, embedding_matrix in zip(
                self.input_factors, self.embedding_matrices)]

        return tf.concat(embedded_factors, 2)

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        return self.mask

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        """Feed the placholders with the data.

        Arguments:
            dataset: The dataset.
            train: A flag whether the train mode is enabled.

        Returns:
            The constructed feed dictionary that contains the factor data and
            the mask.
        """
        fd = {}  # type: FeedDict

        # for checking the lengths of individual factors
        arr_strings = []
        last_paddings = None

        for factor_plc, name, vocabulary in zip(
                self.input_factors, self.data_ids, self.vocabularies):
            factors = dataset.get_series(name)
            vectors, paddings = vocabulary.sentences_to_tensor(
                list(factors), self.max_length, pad_to_max_len=False,
                train_mode=train, add_start_symbol=self.add_start_symbol,
                add_end_symbol=self.add_end_symbol)

            fd[factor_plc] = list(zip(*vectors))

            arr_strings.append(paddings.tostring())
            last_paddings = paddings

        if len(set(arr_strings)) > 1:
            raise ValueError("The lenghts of factors do not match")

        fd[self.mask] = list(zip(*last_paddings))

        return fd


class EmbeddedSequence(EmbeddedFactorSequence):
    """A sequence of embedded inputs (for a single factor)."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 vocabulary: Union[Vocabulary, CharacterVocabulary],
                 data_id: str,
                 embedding_size: int,
                 max_length: int = None,
                 add_start_symbol: bool = False,
                 add_end_symbol: bool = False,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Construct a new instance of `EmbeddedSequence`.

        Arguments:
            name: The name for the `ModelPart` object
            vocabulary: A `Vocabulary` object used for the sequence data
            data_id: A string that identifies the data series used for
                the sequence data
            embedding_sizes: An integer that specifies the size of the
                embedding vector for the sequence data
            max_length: The maximum length of the sequences
            add_start_symbol: Includes <s> in the sequence
            add_end_symbol: Includes </s> in the sequence
            save_checkpoint: The save_checkpoint parameter for `ModelPart`
            load_checkpoint: The load_checkpoint parameter for `ModelPart`
        """
        check_argument_types()
        EmbeddedFactorSequence.__init__(
            self,
            name=name,
            vocabularies=[vocabulary],
            data_ids=[data_id],
            embedding_sizes=[embedding_size],
            max_length=max_length,
            add_start_symbol=add_start_symbol,
            add_end_symbol=add_end_symbol,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint,
            initializers=initializers)
    # pylint: enable=too-many-arguments

    @property
    def inputs(self) -> tf.Tensor:
        """Return a 2D placeholder for the sequence inputs."""
        return self.input_factors[0]

    # pylint: disable=unsubscriptable-object
    @property
    def embedding_matrix(self) -> tf.Tensor:
        """Return the embedding matrix for the sequence."""
        return self.embedding_matrices[0]
    # pylint: enable=unsubscriptable-object

    @property
    def vocabulary(self) -> Union[Vocabulary, CharacterVocabulary]:
        """Return the input vocabulary."""
        return self.vocabularies[0]

    @property
    def data_id(self) -> str:
        """Return the input data series indentifier."""
        return self.data_ids[0]


class CharacterLevelSequence(EmbeddedSequence):

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 vocabulary: CharacterVocabulary,
                 data_id: str,
                 embedding_size: int,
                 encoder_type: str = "recurrent",
                 rnn_cell: str = "GRU",
                 pooling: str = "maxpool",
                 conv_filters: List[int] = None,
                 max_length: int = None,
                 add_start_symbol: bool = False,
                 add_end_symbol: bool = False,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        EmbeddedSequence.__init__(
            self,
            name=name,
            vocabulary=vocabulary,
            data_id=data_id,
            embedding_size=embedding_size,
            max_length=max_length,
            add_start_symbol=add_start_symbol,
            add_end_symbol=add_end_symbol,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint,
            initializers=initializers)

        self.encoder_type = encoder_type
        self.pooling = pooling
        self.conv_filters = conv_filters

        self._rnn_cell_str = rnn_cell

        if self.pooling is None:
            raise ValueError("Pooling strategy must be specified")

        if self.encoder_type == "recurrent":
            if self._rnn_cell_str not in RNN_CELL_TYPES:
                raise ValueError(
                    "RNN cell must be a either 'GRU', 'LSTM', or "
                    "'NematusGRU'. Not {}".format(self._rnn_cell_str))
            if embedding_size % 2 != 0:
                raise ValueError(
                    "`embedding_size` for the character-level "
                    "recurrent encoder must be divisible by 2")
            self.char_emb_size = int(embedding_size / 2)
        elif self.encoder_type == "convolutional":
            if self.conv_filters is None:
                raise ValueError("conv_filters must be specified for the "
                                 "convolutional character-level encoder")
            if embedding_size % len(self.conv_filters) != 0:
                raise ValueError(
                    "`embedding_size` for the characterl-level "
                    "convolutional encoder must be divisible by the number "
                    "of the convolutional fitlers")
            self.char_emb_size = int(embedding_size / len(self.conv_filters))
        else:
            raise ValueError("Unknown encoder_type")

        with self.use_scope():
            # shape = (sent_len, batch_size, tok_len)
            self.mask = tf.placeholder(tf.float32, [None, None, None], "mask")
            self.input_factors = [
                tf.placeholder(tf.int32, [None, None, None],
                               "factor_{}".format(self.data_id))]
    # pylint: enable=too-many-arguments,too-many-locals

    @tensor
    def embedding_matrix(self) -> tf.Tensor:
        """Return a list of embedding matrices for each factor."""

        # Note: Embedding matrices are numbered rather than named by the data
        # id so the data_id string does not need to be the same across
        # experiments

        return get_variable(
            name="embedding_matrix",
            shape=[len(self.vocabulary), self.char_emb_size],
            initializer=tf.glorot_uniform_initializer())

    @tensor
    def temporal_states(self) -> tf.Tensor:
        """Embed tokens by applying encoder on their character sequence."""
        input_shape = tf.shape(self.inputs)
        embedded_chars = tf.nn.embedding_lookup(self.embedding_matrix,
                                                self.inputs)
        embedded_chars = tf.reshape(
            embedded_chars,
            [-1, input_shape[-1], self.char_emb_size])

        hidden_states, output_states = None, None
        if self.encoder_type == "recurrent":
            rnn_cell = \
                RNN_CELL_TYPES[self._rnn_cell_str](self.char_emb_size)
            input_lens = tf.reduce_sum(
                tf.reshape(self.mask, [-1, input_shape[2]]), 1)

            hidden_states, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_cell,
                cell_bw=rnn_cell,
                inputs=embedded_chars,
                sequence_length=tf.cast(input_lens, tf.int32),
                dtype=tf.float32)

            hidden_states = tf.concat(hidden_states, 2)
            output_states = tf.concat(output_states, 1)

        elif self.encoder_type == "convolutional":
            conv_layers = []
            for filt_size in self.conv_filters:
                filt = get_variable(
                    name="conv_filter_{}".format(filt_size),
                    shape=[filt_size, self.char_emb_size, self.char_emb_size],
                    initializer=tf.glorot_uniform_initializer())
                conv_layer = tf.nn.conv1d(
                    value=embedded_chars,
                    filters=filt,
                    stride=1,
                    padding="SAME",
                    name="conv_{}".format(filt_size))
            conv_layers.append(conv_layer)
            hidden_states = tf.concat(conv_layers, 3)

        hidden_states = tf.reshape(
            hidden_states,
            [-1, input_shape[1], input_shape[2], self.embedding_sizes[0]])
        output_states = tf.reshape(
            output_states,
            [-1, input_shape[1], self.embedding_sizes[0]])
        if self.pooling == "maxpool":
            return tf.reduce_max(hidden_states, axis=2)
        elif self.pooling == "average":
            return tf.reduce_mean(hidden_states, axis=2)
        return output_states

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        return tf.reduce_max(self.mask, 2)

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = {}

        vectors, paddings = self.vocabulary.sentences_to_tensor(
            list(dataset.get_series(self.data_id)),
            self.max_length,
            pad_to_max_len=False,
            train_mode=train,
            add_start_symbol=self.add_start_symbol,
            add_end_symbol=self.add_end_symbol)

        # shape(vectors) = (sent_len, batch_size, tok_len)
        # we need to transform the vectors to batch-major shape
        fd[self.inputs] = np.swapaxes(vectors, 0, 1)
        fd[self.mask] = np.swapaxes(paddings, 0, 1)

        return fd
