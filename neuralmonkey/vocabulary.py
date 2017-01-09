"""This module implements the Vocabulary class and the helper functions that
can be used to obtain a Vocabulary instance.
"""
# tests: lint, mypy

from typing import List, Tuple

import os
import collections
import random
import pickle as pickle
import numpy as np

from neuralmonkey.logging import log
from neuralmonkey.dataset import Dataset, LazyDataset

PAD_TOKEN = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

_SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

PAD_TOKEN_INDEX = 0
START_TOKEN_INDEX = 1
END_TOKEN_INDEX = 2
UNK_TOKEN_INDEX = 3


def _is_special_token(word: str) -> bool:
    """Check whether word is a special token (such as <pad> or <s>).

    Arguments:
        word: The word to check

    Returns:
        True if the word is special, False otherwise.
    """
    return (word == PAD_TOKEN
            or word == START_TOKEN
            or word == END_TOKEN
            or word == UNK_TOKEN)


def from_file(path: str) -> 'Vocabulary':
    """Loads vocabulary from a pickled file

    Arguments:
        path: The path to the pickle file

    Returns:
        The newly created vocabulary.
    """
    if not os.path.exists(path):
        raise Exception("Vocabulary file does not exist: {}".format(path))

    with open(path, 'rb') as f_pickle:
        vocabulary = pickle.load(f_pickle)
    assert isinstance(vocabulary, Vocabulary)

    log("Pickled vocabulary loaded. Size: {} words".format(len(vocabulary)))
    vocabulary.log_sample()
    return vocabulary


# pylint: disable=too-many-arguments
# helper function, this number of parameters is needed
def from_dataset(datasets: List[Dataset], series_ids: List[str], max_size: int,
                 save_file: str=None, overwrite: bool=False,
                 unk_sample_prob: float=0.5) -> 'Vocabulary':
    """Loads vocabulary from a dataset with an option to save it.

    Arguments:
        datasets: A list of datasets from which to create the vocabulary
        series_ids: A list of ids of series of the datasets that should be used
                    producing the vocabulary
        max_size: The maximum size of the vocabulary
        save_file: A file to save the vocabulary to. If None (default),
                   the vocabulary will not be saved.
        unk_sample_prob: The probability with which to sample unks out of
                         words with frequency 1. Defaults to 0.5.

    Returns:
        The new Vocabulary instance.
    """
    vocabulary = Vocabulary(unk_sample_prob=unk_sample_prob)

    for dataset in datasets:
        if isinstance(dataset, LazyDataset):
            log("Warning: inferring vocabulary from lazy dataset", color="red")

        for series_id in series_ids:
            series = dataset.get_series(series_id, allow_none=True)
            if series:
                vocabulary.add_tokenized_text(
                    [token for sent in series for token in sent])

    vocabulary.trunkate(max_size)

    log("Vocabulary for series {} initialized, containing {} words"
        .format(series_ids, len(vocabulary)))

    vocabulary.log_sample()

    if save_file is not None:
        directory = os.path.dirname(save_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        vocabulary.save_to_file(save_file, overwrite)

    return vocabulary


def from_bpe(path: str, encoding: str="utf-8") -> 'Vocabulary':
    """Loads vocabulary from Byte-pair encoding merge list.

    NOTE: The frequencies of words in this vocabulary are not computed from
    data. Instead, they correspond to the number of times the subword units
    occurred in the BPE merge list. This means that smaller words will tend to
    have larger frequencies assigned and therefore the truncation of the
    vocabulary can be somehow performed (but not without a great deal of
    thought).

    Arguments:
        path: File name to load the vocabulary from.
        encoding: The encoding of the merge file (defaults to UTF-8)
    """
    if not os.path.exists(path):
        raise Exception("BPE file does not exist: {}".format(path))

    vocab = Vocabulary()

    with open(path, encoding=encoding) as f_bpe:
        for line in f_bpe:
            pair = line.split()
            assert len(pair) == 2

            if pair[1].endswith("</w>"):
                pair[1] = pair[1][:-4]
            else:
                pair[1] += "@@"

            vocab.add_word(pair[0] + "@@")
            vocab.add_word(pair[1])
            vocab.add_word("".join(pair))

    log("Vocabulary from BPE merges loaded. Size: {} subwords"
        .format(len(vocab)))
    vocab.log_sample()
    return vocab


def initialize_vocabulary(directory: str, name: str,
                          datasets: List[Dataset]=None,
                          series_ids: List[str]=None,
                          max_size: int=None) -> 'Vocabulary':
    """This function is supposed to initialize vocabulary when called from
    the configuration file. It first checks whether the vocabulary is already
    loaded on the provided path and if not, it tries to generate it from
    the provided dataset.

    Args:
        directory: Directory where the vocabulary should be stored.

        name: Name of the vocabulary which is also the name of the file
              it is stored it.

        datasets: A a list of datasets from which the vocabulary can be
                  created.

        series_ids: A list of ids of series of the datasets that should be used
                    for producing the vocabulary.

        max_size: The maximum size of the vocabulary

    Returns:
        The new vocabulary
    """
    log("Warning! Use of deprecated initialize_vocabulary method. "
        "Did you think this through?", color="red")

    file_name = os.path.join(directory, name + ".pickle")
    if os.path.exists(file_name):
        return from_file(file_name)

    if datasets is None or series_ids is None or max_size is None:
        raise Exception("Vocabulary does not exist in \"{}\"," +
                        "neither dataset and series_id were provided.")

    return from_dataset(datasets, series_ids, max_size,
                        save_file=file_name, overwrite=False)


class Vocabulary(collections.Sized):

    def __init__(self, tokenized_text: List[str]=None,
                 unk_sample_prob: float=0.0) -> None:
        """Create a new instance of a vocabulary.

        Arguments:
            tokenized_text: The initial list of words to add.
        """
        self.word_to_index = {}  # type: Dict[str, int]
        self.index_to_word = []  # type: List[str]
        self.word_count = {}  # type: Dict[str, int]

        self.unk_sample_prob = unk_sample_prob

        self.add_word(PAD_TOKEN)
        self.add_word(START_TOKEN)
        self.add_word(END_TOKEN)
        self.add_word(UNK_TOKEN)

        if tokenized_text:
            self.add_tokenized_text(tokenized_text)

    def __len__(self) -> int:
        """Get the size of the vocabulary.

        Returns:
            The number of distinct words in the vocabulary.
        """
        return len(self.index_to_word)

    def __contains__(self, word: str) -> bool:
        """Check if a word is in the vocabulary.

        Arguments:
            word: The word to look up.

        Returns:
            True if the word was added to the vocabulary, False otherwise.
        """
        return word in self.word_to_index

    def add_word(self, word: str) -> None:
        """Add a word to the vocablulary.

        Arguments:
            word: The word to add. If it's already there, increment the count.
        """
        if word not in self:
            self.word_to_index[word] = len(self.index_to_word)
            self.index_to_word.append(word)
            self.word_count[word] = 0
        self.word_count[word] += 1

    def add_tokenized_text(self, tokenized_text: List[str]) -> None:
        """Add words from a list to the vocabulary.

        Arguments:
            tokenized_text: The list of words to add.
        """
        for word in tokenized_text:
            self.add_word(word)

    def get_word_index(self, word: str) -> int:
        """ Return index of the specified word.

        Arguments:
            word: The word to look up.

        Returns:
            Index of the word or index of the unknown token if the word is not
            present in the vocabulary.
        """
        if word not in self:
            return self.get_word_index(UNK_TOKEN)
        return self.word_to_index[word]

    def get_unk_sampled_word_index(self, word):
        """Return index of the specified word with sampling of unknown words.

        This method returns the index of the specified word in the vocabulary.
        If the frequency of the word in the vocabulary is 1 (the word was only
        seen once in the whole training dataset), with probability of
        self.unk_sample_prob, generate the index of the unknown token instead.

        Arguments:
            word: The word to look up.

        Returns:
            Index of the word, index of the unknown token if sampled, or index
            of the unknown token if the word is not present in the vocabulary.
        """
        idx = self.word_to_index.get(word, self.get_word_index(UNK_TOKEN))
        freq = self.word_count.get(word, 0)

        if freq <= 1 and random.random() < self.unk_sample_prob:
            return self.get_word_index(UNK_TOKEN)

        return idx

    def trunkate(self, size: int) -> None:
        """Truncate the vocabulary to the requested size by discarding
        infrequent tokens.

        Arguments:
            size: The final size of the vocabulary
        """
        # sort by frequency
        words_by_freq = sorted(list(self.word_count.keys()),
                               key=lambda w: self.word_count[w])

        # keep the least frequent words which are not special symbols
        words_to_delete = [w for w in words_by_freq[:-size]
                           if not _is_special_token(w)]

        # sort by index ... bigger indices needs to be removed first
        # to keep the lists propertly shaped
        delete_words_by_index = sorted(
            [(w, self.word_to_index[w]) for w in words_to_delete],
            key=lambda p: -p[1])

        for word, index in delete_words_by_index:
            del self.word_count[word]
            del self.index_to_word[index]

        self.word_to_index = {}
        for index, word in enumerate(self.index_to_word):
            self.word_to_index[word] = index

    def sentences_to_tensor(
            self,
            sentences: List[List[str]],
            max_len: int,
            train_mode: bool=False,
            add_start_symbol: bool=False,
            add_end_symbol: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the tensor representation for the provided sentences.

        Arguments:
            sentences: List of sentences as lists of tokens.
            max_len: Maximum lengh of a sentence toward which they will be
                padded to.
            train_mode: Flag whether we are training or not
                (enables/disables unk sampling).
            add_start_symbol: If True, the `<s>` token will be added to the
                beginning of each sentence vector. Enabling this option extends
                the vector size (`max_len`) by one.
            add_end_symbol: If True, the `</s>` token will be added to the end
                of each sentence vector, provided that the sentence is shorter
                than `max_len`. If not, the end token is not added. Unlike
                `add_start_symbol`, enabling this option **does not alter**
                the size of the vectors.

        Returns:
            A tuple of a sentence tensor and a padding weight vector.

            The shape of the tensor representing the sentences is either
            (max_len, batch_size) or (max_len, batch_size), depending on
            the value of the `add_start_symbol` argument.

            The shape of the padding vector is the same as of the sentence
            vector.
        """
        word_indices = np.full(
            [max_len, len(sentences)], self.get_word_index(PAD_TOKEN),
            dtype=np.int32)
        weights = np.zeros([max_len, len(sentences)])

        for i in range(max_len):
            for j, sent in enumerate(sentences):
                if i < len(sent):
                    w_idx = (self.get_unk_sampled_word_index(sent[i])
                             if train_mode else self.get_word_index(sent[i]))
                    word_indices[i, j] = w_idx
                    weights[i, j] = 1

                elif i == len(sent) and add_end_symbol:
                    word_indices[i, j] = self.get_word_index(END_TOKEN)
                    weights[i, j] = 1

        if add_start_symbol:
            prepend_indices = np.full(
                [1, len(sentences)], self.get_word_index(START_TOKEN),
                dtype=np.int32)
            prepend_weights = np.ones([1, len(sentences)])

            word_indices = np.concatenate((prepend_indices, word_indices))
            weights = np.concatenate((prepend_weights, weights))

        return word_indices, weights

    def vectors_to_sentences(self,
                             vectors: List[np.ndarray]) -> List[List[str]]:
        """Convert vectors of indexes of vocabulary items to lists of words.

        Arguments:
            vectors: List of vectors of vocabulary indices.

        Returns:
            List of lists of words.
        """
        sentences = [[] for _ in range(vectors[0].shape[0])]
        # type: List[List[str]]

        for vec in vectors:
            for sentence, word_i in zip(sentences, vec):
                if not sentence or sentence[-1] != END_TOKEN:
                    sentence.append(self.index_to_word[word_i])

        return [s[:-1] if s[-1] == END_TOKEN else s for s in sentences]

    def save_to_file(self, path: str, overwrite: bool=False) -> None:
        """Save the vocabulary to a file.

        Arguments:
            path: The path to save the file to.
            overwrite: Flag whether to overwrite existing file.
                       Defaults to False.
        Raises:
            FileExistsError if the file exists and overwrite flag is
            disabled.
        """
        if os.path.exists(path) and not overwrite:
            raise FileExistsError("Cannot save vocabulary: File exists and "
                                  "overwrite is disabled. {}".format(path))

        with open(path, 'wb') as f_pickle:
            pickle.dump(self, f_pickle)

    def log_sample(self, size: int=5):
        """Logs a sample of the vocabulary

        Arguments:
            size: How many sample words to log.
        """
        log("Sample of the vocabulary: {}"
            .format([self.index_to_word[i]
                     for i in np.random.randint(0, len(self), size)]))
