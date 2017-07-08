"""This module impements the sequence class and a few of its subclasses"""

import os
from typing import List

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from typeguard import check_argument_types

from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.decorators import tensor
from neuralmonkey.dataset import Dataset


class Sequence(ModelPart):
    """ Base class for a data sequence.

    This class represents a batch of sequences of Tensors of possibly
    different lengths.
    """
    def __init__(self,
                 name: str,
                 max_length: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Construct a new `Sequence` object.

        Arguments:
            name: The name for the `ModelPart` object
            max_length: Maximum length of sequences in the object (not checked)
            save_checkpoint: The save_checkpoint parameter for `ModelPart`
            load_checkpoint: The load_checkpoint parameter for `ModelPart`
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        check_argument_types()

        self._max_length = max_length
        if self._max_length is not None and self._max_length <= 0:
            raise ValueError("Max sequence length must be a positive integer.")

    @property
    def data(self) -> tf.Tensor:
        """A `Tensor` representing the data in the sequence. The first and
        second dimension correspond to batch size and time respectively.
        """
        raise NotImplementedError("Accessing abstract property")

    @property
    def mask(self) -> tf.Tensor:
        """A 2D `Tensor` of type `float32` and shape (batch size, time) that
        masks the sequences in the batch.
        """
        raise NotImplementedError("Accessing abstract property")

    @property
    def dimension(self) -> int:
        """The dimension of the sequence. For 3D sequences, this is the size
        of the last dimension of the `data` tensor.
        """
        # TODO make this work for higher dimensional tensors
        raise NotImplementedError("Accessing abstract property")

    @property
    def max_length(self) -> int:
        """The maximum length of sequences in the `data` tensor."""
        return self._max_length

    @tensor
    def lengths(self) -> tf.Tensor:
        """A 1D `Tensor` of type `int32` that stores the lengths of the
        sequences in the batch
        """
        return tf.to_int32(tf.reduce_sum(self.mask, 1))


class EmbeddedFactorSequence(Sequence):
    """A `Sequence` that stores one or more embedded inputs (factors)."""

    def __init__(self,
                 name: str,
                 vocabularies: List[Vocabulary],
                 data_ids: List[str],
                 embedding_sizes: List[int],
                 max_length: int = None,
                 pad_to_max_len: bool = False,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Construct a new instance of `EmbeddedFactorSequence`

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
            save_checkpoint: The save_checkpoint parameter for `ModelPart`
            load_checkpoint: The load_checkpoint parameter for `ModelPart`
        """
        Sequence.__init__(self, name, max_length,
                          save_checkpoint, load_checkpoint)
        check_argument_types()

        self.vocabularies = vocabularies
        self.vocabulary_sizes = [len(vocab) for vocab in self.vocabularies]
        self.data_ids = data_ids
        self.embedding_sizes = embedding_sizes
        self.pad_to_max_len = pad_to_max_len

        if not (len(self.data_ids)
                == len(self.vocabularies)
                == len(self.embedding_sizes)):
            raise ValueError("data_ids, vocabularies, and embedding_sizes "
                             "lists need to have the same length")

        if any([esize <= 0 for esize in self.embedding_sizes]):
            raise ValueError("Embedding size must be a positive integer.")

    # TODO this should be placed into the abstract embedding class
    def tb_embedding_visualization(self, logdir: str,
                                   prj: projector):
        """Links embeddings with vocabulary wordlist for tensorboard
        visualization

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
    def input_factors(self) -> List[tf.Tensor]:
        """A list of 2D placeholders for each factor. Each placeholder has
        shape (batch size, time).
        """
        plc_names = ["sequence_data_{}".format(data_id)
                     for data_id in self.data_ids]

        return [tf.placeholder(tf.int32, [None, None], name)
                for name in plc_names]

    # pylint: disable=no-self-use
    @tensor
    def mask(self) -> tf.Tensor:
        """A 2D placeholder for the sequence mask. This is shared across
        factors and must be the same for each of them.
        """
        return tf.placeholder(tf.float32, [None, None], "sequence_mask")
    # pylint: enable=no-self-use

    @tensor
    def embedding_matrices(self) -> List[tf.Tensor]:
        """A list of embedding matrices for each factor"""
        # TODO better initialization
        # embedding matrices are numbered rather than named by the data id so
        # the data_id string does not need to be the same across experiments
        return [
            tf.get_variable(
                name="embedding_matrix_{}".format(i),
                shape=[vocab_size, emb_size],
                initializer=tf.random_normal_initializer(stddev=0.01))
            for i, (data_id, vocab_size, emb_size) in enumerate(zip(
                self.data_ids, self.vocabulary_sizes, self.embedding_sizes))]

    @tensor
    def data(self) -> tf.Tensor:
        """The sequence data. A 3D Tensor of shape (batch, time, dimension),
        where dimension is the sum of the embedding sizes supplied to the
        constructor.
        """
        embedded_factors = [
            tf.nn.embedding_lookup(embedding_matrix, factor)
            for factor, embedding_matrix in zip(
                self.input_factors, self.embedding_matrices)]

        return tf.concat(embedded_factors, 2)

    @property
    def dimension(self) -> int:
        """The sequence dimension. The sum of the embedding sizes supplied to
        the constructor.
        """
        return sum(self.embedding_sizes)

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
                list(factors), self.max_length, pad_to_max_len=self.pad_to_max_len,
                train_mode=train)

            fd[factor_plc] = list(zip(*vectors))

            arr_strings.append(paddings.tostring())
            last_paddings = paddings

        if len(set(arr_strings)) > 1:
            raise ValueError("The lenghts of factors do not match")

        fd[self.mask] = list(zip(*last_paddings))

        return fd


class EmbeddedSequence(EmbeddedFactorSequence):
    """A sequence of embedded inputs (for a single factor)"""

    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 embedding_size: int,
                 max_length: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Construct a new instance of `EmbeddedSequence`

        Arguments:
            name: The name for the `ModelPart` object
            vocabulary: A `Vocabulary` object used for the sequence data
            data_id: A string that identifies the data series used for
                the sequence data
            embedding_sizes: An integer that specifies the size of the
                embedding vector for the sequence data
            max_length: The maximum length of the sequences
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
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint)

    # pylint: disable=unsubscriptable-object
    @property
    def inputs(self) -> tf.Tensor:
        """A 2D placeholder for the sequence inputs."""
        return self.input_factors[0]

    @property
    def embedding_matrix(self) -> tf.Tensor:
        """The embedding matrix for the sequence"""
        return self.embedding_matrices[0]
    # pylint: enable=unsubscriptable-object

    @property
    def vocabulary(self) -> Vocabulary:
        """The input vocabulary"""
        return self.vocabularies[0]

    @property
    def data_id(self) -> str:
        """The input data series indentifier"""
        return self.data_ids[0]
