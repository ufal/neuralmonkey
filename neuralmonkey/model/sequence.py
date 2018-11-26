"""Module which impements the sequence class and a few of its subclasses."""

import os
from typing import List

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from typeguard import check_argument_types

from neuralmonkey.model.feedable import FeedDict
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.stateful import TemporalStateful
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.decorators import tensor
from neuralmonkey.dataset import Dataset
from neuralmonkey.tf_utils import get_variable


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
                 reuse: ModelPart = None,
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
        ModelPart.__init__(self, name, reuse, save_checkpoint, load_checkpoint,
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
                 vocabularies: List[Vocabulary],
                 data_ids: List[str],
                 embedding_sizes: List[int],
                 max_length: int = None,
                 add_start_symbol: bool = False,
                 add_end_symbol: bool = False,
                 scale_embeddings_by_depth: bool = False,
                 embeddings_source: "EmbeddedFactorSequence" = None,
                 reuse: ModelPart = None,
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
            scale_embeddings_by_depth: Set to True for T2T import compatibility
            embeddings_source: EmbeddedSequence from which the embeedings will
                be reused.
            save_checkpoint: The save_checkpoint parameter for `ModelPart`
            load_checkpoint: The load_checkpoint parameter for `ModelPart`
        """
        check_argument_types()
        Sequence.__init__(
            self, name, max_length, reuse, save_checkpoint, load_checkpoint,
            initializers)

        self.vocabularies = vocabularies
        self.vocabulary_sizes = [len(vocab) for vocab in self.vocabularies]
        self.data_ids = data_ids
        self.embedding_sizes = embedding_sizes
        self.add_start_symbol = add_start_symbol
        self.add_end_symbol = add_end_symbol
        self.scale_embeddings_by_depth = scale_embeddings_by_depth
        self.embeddings_source = embeddings_source

        if not (len(self.data_ids)
                == len(self.vocabularies)
                == len(self.embedding_sizes)):
            raise ValueError("data_ids, vocabularies, and embedding_sizes "
                             "lists need to have the same length")

        if any([esize <= 0 for esize in self.embedding_sizes]):
            raise ValueError("Embedding size must be a positive integer.")

        if embeddings_source is not None:
            if not all(v1 == v2 for v1, v2 in zip(
                    self.vocabularies, embeddings_source.vocabularies)):
                raise ValueError(
                    "When reusing embeedings, vocabularies must be the same.")
            if not all(s1 == s2 for s1, s2 in zip(
                    self.embedding_sizes, embeddings_source.embedding_sizes)):
                raise ValueError(
                    "When reusing embeedings, embeddings sizes must be equal.")

        self._variable_scope.set_initializer(
            tf.random_normal_initializer(stddev=0.001))
    # pylint: enable=too-many-arguments

    # pylint: disable=no-self-use
    @tensor
    def mask(self) -> tf.Tensor:
        return tf.placeholder(tf.float32, [None, None], "mask")

    @tensor
    def input_factors(self) -> List[tf.Tensor]:
        return [
            tf.placeholder(tf.int32, [None, None], "factor_{}".format(did))
            for did in self.data_ids]
    # pylint: enable=no-self-use

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
            metadata_path = self.name + "_" + str(i) + ".tsv"
            self.vocabularies[i].save_wordlist(
                os.path.join(logdir, metadata_path), True, True)

            embedding = prj.embeddings.add()
            # pylint: disable=unsubscriptable-object
            embedding.tensor_name = self.embedding_matrices[i].name
            embedding.metadata_path = metadata_path

    @tensor
    def embedding_matrices(self) -> List[tf.Tensor]:
        """Return a list of embedding matrices for each factor."""

        # Note: Embedding matrices are numbered rather than named by the data
        # id so the data_id string does not need to be the same across
        # experiments

        if self.embeddings_source is not None:
            return self.embeddings_source.embedding_matrices

        return [
            get_variable(
                name="embedding_matrix_{}".format(i),
                shape=[vocab_size, emb_size])
            for i, (data_id, vocab_size, emb_size) in enumerate(zip(
                self.data_ids, self.vocabulary_sizes, self.embedding_sizes))]

    @tensor
    def temporal_states(self) -> tf.Tensor:
        """Return the embedded factors.

        A 3D Tensor of shape (batch, time, dimension),
        where dimension is the sum of the embedding sizes supplied to the
        constructor.
        """
        embedded_factors = []
        for (factor, embedding_matrix) in zip(
                self.input_factors, self.embedding_matrices):
            emb_factor = tf.nn.embedding_lookup(embedding_matrix, factor)

            # github.com/tensorflow/tensor2tensor/blob/v1.5.6/tensor2tensor/
            #            layers/modalities.py#L104
            if self.scale_embeddings_by_depth:
                emb_size = embedding_matrix.shape.as_list()[-1]
                emb_factor *= emb_size**0.5

            # We explicitly set paddings to zero-value vectors
            # TODO: remove unnecessary masking in the subesquent modules
            emb_factor = emb_factor * tf.expand_dims(self.mask, -1)
            embedded_factors.append(emb_factor)

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
        fd = ModelPart.feed_dict(self, dataset, train)

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

        assert last_paddings is not None
        fd[self.mask] = list(zip(*last_paddings))

        return fd


class EmbeddedSequence(EmbeddedFactorSequence):
    """A sequence of embedded inputs (for a single factor)."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 embedding_size: int,
                 max_length: int = None,
                 add_start_symbol: bool = False,
                 add_end_symbol: bool = False,
                 scale_embeddings_by_depth: bool = False,
                 embeddings_source: "EmbeddedSequence" = None,
                 reuse: ModelPart = None,
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
            scale_embeddings_by_depth: Set to True for T2T import compatibility
            embeddings_source: `EmbeddedSequence` from which the embeedings
                will be reused.
            save_checkpoint: The save_checkpoint parameter for `ModelPart`
            load_checkpoint: The load_checkpoint parameter for `ModelPart`
        """
        EmbeddedFactorSequence.__init__(
            self,
            name=name,
            vocabularies=[vocabulary],
            data_ids=[data_id],
            embedding_sizes=[embedding_size],
            max_length=max_length,
            add_start_symbol=add_start_symbol,
            add_end_symbol=add_end_symbol,
            scale_embeddings_by_depth=scale_embeddings_by_depth,
            embeddings_source=embeddings_source,
            reuse=reuse,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint,
            initializers=initializers)
    # pylint: enable=too-many-arguments

    # pylint: disable=unsubscriptable-object
    @property
    def inputs(self) -> tf.Tensor:
        """Return a 2D placeholder for the sequence inputs."""
        return self.input_factors[0]

    @property
    def embedding_matrix(self) -> tf.Tensor:
        """Return the embedding matrix for the sequence."""
        return self.embedding_matrices[0]
    # pylint: enable=unsubscriptable-object

    @property
    def vocabulary(self) -> Vocabulary:
        """Return the input vocabulary."""
        return self.vocabularies[0]

    @property
    def data_id(self) -> str:
        """Return the input data series indentifier."""
        return self.data_ids[0]
