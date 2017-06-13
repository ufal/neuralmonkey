from typing import List

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.decorators import tensor
from neuralmonkey.dataset import Dataset


class Sequence(ModelPart):

    def __init__(self,
                 name: str,
                 max_length: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        check_argument_types()

        self._max_length = max_length
        if self._max_length is not None and self._max_length <= 0:
            raise ValueError("Max sequence length must be a positive integer.")

    @property
    def data(self) -> tf.Tensor:
        raise NotImplementedError("Accessing abstract property")

    @property
    def mask(self) -> tf.Tensor:
        raise NotImplementedError("Accessing abstract property")

    @property
    def dimension(self) -> int:
        raise NotImplementedError("Accessing abstract property")

    @property
    def max_length(self) -> int:
        return self._max_length


class EmbeddedFactorSequence(Sequence):

    def __init__(self,
                 name: str,
                 vocabularies: List[Vocabulary],
                 data_ids: List[str],
                 embedding_sizes: List[int],
                 max_length: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        Sequence.__init__(self, name, max_length,
                          save_checkpoint, load_checkpoint)
        check_argument_types()

        self.vocabularies = vocabularies
        self.vocabulary_sizes = [len(vocab) for vocab in self.vocabularies]
        self.data_ids = data_ids
        self.embedding_sizes = embedding_sizes

        if not (len(self.data_ids)
                == len(self.vocabularies)
                == len(self.embedding_sizes)):
            raise ValueError("data_ids, vocabularies, and embedding_sizes "
                             "lists need to have the same length")

        if any([esize <= 0 for esize in self.embedding_sizes]):
            raise ValueError("Embedding size must be a positive integer.")

    @tensor
    def input_factors(self) -> List[tf.Tensor]:
        plc_names = ["sequence_data_{}".format(data_id)
                     for data_id in self.data_ids]

        return [tf.placeholder(tf.int32, [None, None], name)
                for name in plc_names]

    # pylint: disable=no-self-use
    @tensor
    def mask(self) -> tf.Tensor:
        return tf.placeholder(tf.float32, [None, None], "sequence_mask")
    # pylint: enable=no-self-use

    @tensor
    def embedding_matrices(self) -> List[tf.Tensor]:
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
        embedded_factors = [
            tf.nn.embedding_lookup(embedding_matrix, factor)
            for factor, embedding_matrix in zip(
                self.input_factors, self.embedding_matrices)]

        return tf.concat(embedded_factors, 2)

    @property
    def dimension(self) -> int:
        return sum(self.embedding_sizes)

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = {}  # type: FeedDict

        # for checking the lengths of individual factors
        arr_strings = []
        last_paddings = None

        for factor_plc, name, vocabulary in zip(
                self.input_factors, self.data_ids, self.vocabularies):
            factors = dataset.get_series(name)
            vectors, paddings = vocabulary.sentences_to_tensor(
                list(factors), self.max_length, pad_to_max_len=False,
                train_mode=train)

            fd[factor_plc] = list(zip(*vectors))

            arr_strings.append(paddings.tostring())
            last_paddings = paddings

        if len(set(arr_strings)) > 1:
            raise ValueError("The lenghts of factors do not match")

        fd[self.mask] = list(zip(*last_paddings))

        return fd


class EmbeddedSequence(EmbeddedFactorSequence):
    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 embedding_size: int,
                 max_length: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
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
        return self.input_factors[0]

    @property
    def embedding_matrix(self) -> tf.Tensor:
        return self.embedding_matrices[0]
    # pylint: enable=unsubscriptable-object

    @property
    def vocabulary(self) -> Vocabulary:
        return self.vocabularies[0]

    @property
    def data_id(self) -> str:
        return self.data_ids[0]

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = {}  # type: FeedDict

        sentences = dataset.get_series(self.data_id)
        vectors, paddings = self.vocabulary.sentences_to_tensor(
            list(sentences), self.max_length, pad_to_max_len=False,
            train_mode=train)

        # as sentences_to_tensor returns lists of shape (time, batch),
        # we need to transpose
        fd[self.inputs] = list(zip(*vectors))
        fd[self.mask] = list(zip(*paddings))

        return fd
