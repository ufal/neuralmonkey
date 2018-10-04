"""Vocabulary class module.

This module implements the Vocabulary class and the helper functions that
can be used to obtain a Vocabulary instance.
"""
# pylint: disable=too-many-lines

import collections
import json
import os
import random

# pylint: disable=unused-import
from typing import List, Optional, Tuple, Dict, Union
# pylint: enable=unused-import

import numpy as np
from typeguard import check_argument_types

from neuralmonkey.logging import log, warn
from neuralmonkey.dataset import Dataset

PAD_TOKEN = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

PAD_TOKEN_INDEX = 0
START_TOKEN_INDEX = 1
END_TOKEN_INDEX = 2
UNK_TOKEN_INDEX = 3


def is_special_token(word: str) -> bool:
    """Check whether word is a special token (such as <pad> or <s>).

    Arguments:
        word: The word to check

    Returns:
        True if the word is special, False otherwise.
    """
    return word in SPECIAL_TOKENS


# pylint: disable=unused-argument
def from_file(*args, **kwargs) -> "Vocabulary":
    raise NotImplementedError("Use loading by from_wordlist")
# pylint: enable=unused-argument


def from_wordlist(path: str,
                  encoding: str = "utf-8",
                  contains_header: bool = True,
                  contains_frequencies: bool = True) -> "Vocabulary":
    """Load a vocabulary from a wordlist.

    The file can contain either list of words with no header.
    Or it can contain words and their counts separated
    by tab and a header on the first line.

    Arguments:
        path: The path to the wordlist file
        encoding: The encoding of the wordlist file (defaults to UTF-8)
        contains_header: if the file have a header on first line
        contains_frequencies: if the file contains frequencies in second column

    Returns:
        The new Vocabulary instance.
    """
    vocabulary = Vocabulary()

    with open(path, encoding=encoding) as wordlist:
        line_number = 1
        if contains_header:
            # skip the header
            line_number += 1
            next(wordlist)

        for line in wordlist:
            line = line.strip()
            # check if line is empty
            if not line:
                warn("Vocabulary file {}:{}: line empty"
                     .format(path, line_number))
                line_number += 1
                continue

            if contains_frequencies:
                info = line.split("\t")
                if len(info) != 2:
                    raise ValueError(
                        "Vocabulary file {}:{}: line does not have two columns"
                        .format(path, line_number))
                vocabulary.add_word(info[0], 0)
            else:
                if "\t" in line:
                    warn("Vocabulary file {}:{}: line contains a tabulator"
                         .format(path, line_number))
                vocabulary.add_word(line)
            line_number += 1

    log("Vocabulary from wordlist loaded, containing {} words"
        .format(len(vocabulary)))
    vocabulary.log_sample()
    return vocabulary


def from_t2t_vocabulary(path: str,
                        encoding: str = "utf-8") -> "Vocabulary":
    """Load a vocabulary generated during tensor2tensor training.

    Arguments:
        path: The path to the vocabulary file.
        encoding: The encoding of the vocabulary file (defaults to UTF-8).

    Returns:
        The new Vocabulary instantce.
    """
    vocabulary = Vocabulary()

    with open(path, encoding=encoding) as wordlist:
        for line in wordlist:
            line = line.strip()

            # T2T vocab tends to wrap words in single quotes
            if ((line.startswith("'") and line.endswith("'"))
                    or (line.startswith('"') and line.endswith('"'))):
                line = line[1:-1]

            if line in ["<pad>", "<EOS>"]:
                continue

            vocabulary.add_word(line)

    log("Vocabulary form wordlist loaded, containing {} words"
        .format(len(vocabulary)))
    vocabulary.log_sample()
    return vocabulary


def from_nematus_json(path: str, max_size: int = None,
                      pad_to_max_size: bool = False) -> "Vocabulary":
    """Load vocabulary from Nematus JSON format.

    The JSON format is a flat dictionary that maps words to their index in the
    vocabulary.

    Args:
        path: Path to the file.
        max_size: Maximum vocabulary size including 'unk' and 'eos' symbols,
            but not including <pad> and <s> symbol.
        pad_to_max_size: If specified, the vocabulary is padded with dummy
            symbols up to the specified maximum size.
    """
    with open(path, "r", encoding="utf-8") as f_json:
        contents = json.load(f_json)

    vocabulary = Vocabulary()
    for word in sorted(contents.keys(), key=lambda x: contents[x]):
        if contents[word] < 2:
            continue
        vocabulary.add_word(word)
        if max_size is not None and len(vocabulary) == max_size:
            break

    if max_size is None:
        max_size = len(vocabulary) - 2  # the "2" is ugly HACK

    if pad_to_max_size and max_size is not None:
        current_length = len(vocabulary)
        for i in range(max_size - current_length + 2):  # the "2" is ugly HACK
            word = "<pad_{}>".format(i)
            vocabulary.add_word(word)

    return vocabulary


# pylint: disable=too-many-arguments
# helper function, this number of parameters is needed
def from_dataset(datasets: List[Dataset], series_ids: List[str], max_size: int,
                 save_file: str = None, overwrite: bool = False,
                 min_freq: Optional[int] = None,
                 unk_sample_prob: float = 0.5) -> "Vocabulary":
    """Load a vocabulary from a dataset with an option to save it.

    Arguments:
        datasets: A list of datasets from which to create the vocabulary
        series_ids: A list of ids of series of the datasets that should be used
                    producing the vocabulary
        max_size: The maximum size of the vocabulary
        save_file: A file to save the vocabulary to. If None (default),
                   the vocabulary will not be saved.
        overwrite: Overwrite existing file.
        min_freq: Do not include words with frequency smaller than this.
        unk_sample_prob: The probability with which to sample unks out of
                         words with frequency 1. Defaults to 0.5.

    Returns:
        The new Vocabulary instance.
    """
    check_argument_types()

    vocabulary = Vocabulary(unk_sample_prob=unk_sample_prob)
    vocabulary.correct_counts = True

    for dataset in datasets:
        if dataset.lazy:
            warn("Inferring vocabulary from lazy dataset!")

        for series_id in series_ids:
            if not dataset.has_series(series_id):
                warn("Data series '{}' not present in the dataset"
                     .format(series_id))

            series = dataset.maybe_get_series(series_id)
            if series is not None:
                vocabulary.add_tokenized_text(
                    [token for sent in series for token in sent])

    vocabulary.truncate(max_size)

    if min_freq is not None:
        if min_freq > 1:
            vocabulary.truncate_by_min_freq(min_freq)

    log("Vocabulary for series {} initialized, containing {} words"
        .format(series_ids, len(vocabulary)))

    vocabulary.log_sample()

    if save_file is not None:
        directory = os.path.dirname(save_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        vocabulary.save_wordlist(save_file, overwrite, True)

    return vocabulary


def initialize_vocabulary(directory: str, name: str,
                          datasets: List[Dataset] = None,
                          series_ids: List[str] = None,
                          max_size: int = None) -> "Vocabulary":
    """Initialize a vocabulary.

    This function is supposed to initialize vocabulary when called from
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
    warn("Use of deprecated initialize_vocabulary method. "
         "Did you think this through?")

    file_name = os.path.join(directory, name + ".pickle")
    if os.path.exists(file_name):
        return from_wordlist(file_name)

    if datasets is None or series_ids is None or max_size is None:
        raise Exception("Vocabulary does not exist in '{}', "
                        "neither dataset and series_id were provided.")

    return from_dataset(datasets, series_ids, max_size,
                        save_file=file_name, overwrite=False)


class Vocabulary(collections.Sized):

    def __init__(self, tokenized_text: List[str] = None,
                 unk_sample_prob: float = 0.0) -> None:
        """Create a new instance of a vocabulary.

        Arguments:
            tokenized_text: The initial list of words to add.
        """
        self.word_to_index = {}  # type: Dict[str, int]
        self.index_to_word = []  # type: List[str]
        self.word_count = {}  # type: Dict[str, int]
        self.alphabet = {tok for tok in SPECIAL_TOKENS}

        # flag if the word count are in use
        self.correct_counts = False

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

    def add_word(self, word: str, occurences: int = 1) -> None:
        """Add a word to the vocablulary.

        Arguments:
            word: The word to add. If it's already there, increment the count.
            occurences: increment the count of word by the number of occurences
        """
        if word not in self:
            self.word_to_index[word] = len(self.index_to_word)
            self.index_to_word.append(word)
            self.word_count[word] = 0
            if not is_special_token(word):
                self.add_characters(word)
        self.word_count[word] += occurences

    def add_characters(self, word: str) -> None:
        self.alphabet |= {c for c in word}

    def add_tokenized_text(self, tokenized_text: List[str]) -> None:
        """Add words from a list to the vocabulary.

        Arguments:
            tokenized_text: The list of words to add.
        """
        for word in tokenized_text:
            self.add_word(word)

    def get_word_index(self, word: str) -> int:
        """Return index of the specified word.

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
            if not self.correct_counts:
                raise ValueError("The vocabulary does not have correct "
                                 "word_counts to use with unknown sampling")
            return self.get_word_index(UNK_TOKEN)

        return idx

    def truncate(self, size: int) -> None:
        """Truncate the vocabulary to the requested size.

        The infrequent tokens are discarded.

        Arguments:
            size: The final size of the vocabulary
        """

        if not self.correct_counts:
            raise ValueError("The vocabulary does not have correct "
                             "word_counts to use for vocabulary truncate")

        # sort by frequency
        # sorting words first makes vocabulary generation deterministic
        words_by_freq = sorted(list(sorted(self.word_count.keys())),
                               key=lambda w: self.word_count[w])

        # keep the least frequent words which are not special symbols
        to_delete = len(self) - size
        if to_delete < 0:
            to_delete = 0
            warn("Actual vocabulary size ({}) is smaller than max_size ({})"
                 .format(len(self), size))
        words_to_delete = []  # type: List[str]
        for word in words_by_freq:
            if len(words_to_delete) == to_delete:
                break
            if not is_special_token(word):
                words_to_delete.append(word)

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

    def truncate_by_min_freq(self, min_freq: int) -> None:
        """Truncate the vocabulary only keeping words with a minimum frequency.

        Arguments:
            min_freq: The minimum frequency of included words.
        """
        if min_freq > 1:
            # count how many words there are with frequency < min_freq
            # ignoring special tokens
            infreq_word_count = sum([1 for w in self.word_count
                                     if self.word_count[w] < min_freq
                                     and not is_special_token(w)])
            log("Removing {} infrequent (<{}) words from vocabulary".format(
                infreq_word_count, min_freq))
            new_size = len(self) - infreq_word_count
            self.truncate(new_size)

    def sentences_to_tensor(
            self,
            sentences: List[List[str]],
            max_len: int = None,
            pad_to_max_len: bool = True,
            train_mode: bool = False,
            add_start_symbol: bool = False,
            add_end_symbol: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the tensor representation for the provided sentences.

        Arguments:
            sentences: List of sentences as lists of tokens.
            max_len: If specified, all sentences will be truncated to this
                length.
            pad_to_max_len: If True, the tensor will be padded to `max_len`,
                even if all of the sentences are shorter. If False, the shape
                of the tensor will be determined by the maximum length of the
                sentences in the batch.
            train_mode: Flag whether we are training or not
                (enables/disables unk sampling).
            add_start_symbol: If True, the `<s>` token will be added to the
                beginning of each sentence vector. Enabling this option extends
                the maximum length by one.
            add_end_symbol: If True, the `</s>` token will be added to the end
                of each sentence vector, provided that the sentence is shorter
                than `max_len`. If not, the end token is not added. Unlike
                `add_start_symbol`, enabling this option **does not alter**
                the maximum length.

        Returns:
            A tuple of a sentence tensor and a padding weight vector.

            The shape of the tensor representing the sentences is either
            `(batch_max_len, batch_size)` or `(batch_max_len+1, batch_size)`,
            depending on the value of the `add_start_symbol` argument.
            `batch_max_len` is the length of the longest sentence in the
            batch (including the optional `</s>` token), limited by `max_len`
            (if specified).

            The shape of the padding vector is the same as of the sentence
            vector.
        """
        if pad_to_max_len and max_len is not None:
            batch_max_len = max_len
        else:
            batch_max_len = max(len(s) for s in sentences)
            if add_end_symbol:
                batch_max_len += 1
            if max_len is not None:
                batch_max_len = min(max_len, batch_max_len)

        word_indices = np.full(
            [batch_max_len, len(sentences)], self.get_word_index(PAD_TOKEN),
            dtype=np.int32)
        weights = np.zeros([batch_max_len, len(sentences)])

        for i in range(batch_max_len):
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
            word_indices = np.insert(word_indices, 0,
                                     self.get_word_index(START_TOKEN), axis=0)
            weights = np.insert(weights, 0, 1, axis=0)

        return word_indices, weights

    def vectors_to_sentences(
            self,
            vectors: Union[List[np.ndarray], np.ndarray]) -> List[List[str]]:
        """Convert vectors of indexes of vocabulary items to lists of words.

        Arguments:
            vectors: List of vectors of vocabulary indices.

        Returns:
            List of lists of words.
        """
        if isinstance(vectors, list):
            if not vectors:
                raise ValueError(
                    "Cannot infer batch size because decoder returned an "
                    "empty output.")
            batch_size = vectors[0].shape[0]
        elif isinstance(vectors, np.ndarray):
            batch_size = vectors.shape[1]
        else:
            raise TypeError(
                "Unexpected type of decoder output: {}".format(type(vectors)))

        sentences = [[] for _ in range(batch_size)]  # type: List[List[str]]

        for vec in vectors:
            for sentence, word_i in zip(sentences, vec):
                if not sentence or sentence[-1] != END_TOKEN:
                    sentence.append(self.index_to_word[word_i])

        return [s[:-1] if s and s[-1] == END_TOKEN else s for s in sentences]

    def save_wordlist(self, path: str, overwrite: bool = False,
                      save_frequencies: bool = False,
                      encoding: str = "utf-8") -> None:
        """Save the vocabulary as a wordlist.

        The file is ordered by the ids of words.
        This function is used mainly for embedding visualization.

        Arguments:
            path: The path to save the file to.
            overwrite: Flag whether to overwrite existing file.
                Defaults to False.
            save_frequencies: flag if frequencies should be stored. This
                parameter adds header into the output file.

        Raises:
            FileExistsError if the file exists and overwrite flag is
            disabled.
        """
        if os.path.exists(path) and not overwrite:
            raise FileExistsError("Cannot save vocabulary: File exists and "
                                  "overwrite is disabled. {}".format(path))

        with open(path, "w", encoding=encoding) as output_file:
            if save_frequencies and self.correct_counts:
                # this header is important for the TensorBoard to properly
                # handle the frequencies.
                #
                # IMPORTANT NOTICE: when saving only wordlist without
                # frequencies it MUST NOT contain the header. It is an
                # exception from Tensorboard. More at
                # https://www.tensorflow.org/get_started/embedding_viz
                output_file.write("Word\tWord counts\n")
            elif save_frequencies and not self.correct_counts:
                log("Storing vocabulary without frequencies.")

            for i in range(len(self.index_to_word)):
                output_file.write(self.index_to_word[i])
                if save_frequencies and self.correct_counts:
                    output_file.write(
                        "\t" + str(self.word_count[self.index_to_word[i]]))

                output_file.write("\n")

    def log_sample(self, size: int = 5) -> None:
        """Log a sample of the vocabulary.

        Arguments:
            size: How many sample words to log.
        """
        if size > len(self):
            log("Vocabulary: {}".format(self.index_to_word))
        else:
            sample_ids = np.random.permutation(np.arange(len(self)))[:size]
            log("Sample of the vocabulary: {}".format(
                [self.index_to_word[i] for i in sample_ids]))
