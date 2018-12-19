"""Vocabulary class module.

This module implements the Vocabulary class and the helper functions that
can be used to obtain a Vocabulary instance.
"""

import collections
import json
import os

from typing import List, Set, Union

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.logging import log, warn, notice

PAD_TOKEN = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

PAD_TOKEN_INDEX = 0
START_TOKEN_INDEX = 1
END_TOKEN_INDEX = 2
UNK_TOKEN_INDEX = 3


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
        contains_frequencies: if the file contains a second column

    Returns:
        The new Vocabulary instance.
    """
    check_argument_types()
    vocabulary = []  # type: List[str]

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
                word = info[0]
            else:
                if "\t" in line:
                    warn("Vocabulary file {}:{}: line contains a tabulator"
                         .format(path, line_number))
                word = line

            if line_number <= len(SPECIAL_TOKENS) + int(contains_header):
                should_be = SPECIAL_TOKENS[
                    line_number - 1 - int(contains_header)]
                if word != should_be:
                    notice("Expected special token {} but encountered a "
                           "different word: {}".format(should_be, word))
                    vocabulary.append(word)
                line_number += 1
                continue

            vocabulary.append(word)
            line_number += 1

    log("Vocabulary from wordlist loaded, containing {} words"
        .format(len(vocabulary)))
    log_sample(vocabulary)
    return Vocabulary(vocabulary)


def from_t2t_vocabulary(path: str,
                        encoding: str = "utf-8") -> "Vocabulary":
    """Load a vocabulary generated during tensor2tensor training.

    Arguments:
        path: The path to the vocabulary file.
        encoding: The encoding of the vocabulary file (defaults to UTF-8).

    Returns:
        The new Vocabulary instantce.
    """
    check_argument_types()
    vocabulary = []  # type: List[str]

    with open(path, encoding=encoding) as wordlist:
        for line in wordlist:
            line = line.strip()

            # T2T vocab tends to wrap words in single quotes
            if ((line.startswith("'") and line.endswith("'"))
                    or (line.startswith('"') and line.endswith('"'))):
                line = line[1:-1]

            if line in ["<pad>", "<EOS>"]:
                continue

            vocabulary.append(line)

    log("Vocabulary form wordlist loaded, containing {} words"
        .format(len(vocabulary)))
    log_sample(vocabulary)

    return Vocabulary(vocabulary)


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
    check_argument_types()
    with open(path, "r", encoding="utf-8") as f_json:
        contents = json.load(f_json)

    vocabulary = []  # type: List[str]
    for word in sorted(contents.keys(), key=lambda x: contents[x]):
        if contents[word] < 2:
            continue
        vocabulary.append(word)
        if max_size is not None and len(vocabulary) == max_size:
            break

    if max_size is None:
        max_size = len(vocabulary) - 2  # the "2" is ugly HACK

    if pad_to_max_size and max_size is not None:
        current_length = len(vocabulary)
        for i in range(max_size - current_length + 2):  # the "2" is ugly HACK
            word = "<pad_{}>".format(i)
            vocabulary.append(word)

    return Vocabulary(vocabulary)


class Vocabulary(collections.Sized):

    def __init__(self, words: List[str], num_oov_buckets: int = 0) -> None:
        """Create a new instance of a vocabulary.

        Arguments:
            words: The mapping of indices to words.
        """

        self._vocabulary = SPECIAL_TOKENS + words
        self._alphabet = {c for word in words for c in word}

        self._index_to_string = (
            tf.contrib.lookup.index_to_string_table_from_tensor(
                mapping=self._vocabulary,
                default_value=UNK_TOKEN))

        self._string_to_index = tf.contrib.lookup.index_table_from_tensor(
            mapping=self._vocabulary,
            num_oov_buckets=num_oov_buckets,
            default_value=UNK_TOKEN_INDEX)

    def __len__(self) -> int:
        """Get the size of the vocabulary.

        Returns:
            The number of distinct words in the vocabulary.
        """
        return len(self._vocabulary)

    def __contains__(self, word: str) -> bool:
        """Check if a word is in the vocabulary.

        Arguments:
            word: The word to look up.

        Returns:
            True if the word was added to the vocabulary, False otherwise.
        """
        return word in self._vocabulary

    @property
    def alphabet(self) -> Set[str]:
        return self._alphabet

    @property
    def index_to_word(self) -> List[str]:
        return self._vocabulary

    def strings_to_indices(self,
                           sentences: tf.Tensor,
                           max_length: int = None,
                           add_start_symbol: bool = False,
                           add_end_symbol: bool = False) -> tf.Tensor:
        """Generate the tensor representation for the provided sentences.

        Arguments:
            sentences: A 2D Tensor with shape (batch, time) with the input
                tokens.
            max_length: Truncate sentences to this length (optional).
            add_start_symbol: If True, the `<s>` token will be added to the
                beginning of each sentence vector. Enabling this option extends
                the maximum length by one.
            add_end_symbol: If True, the `</s>` token will be added to the end
                of each sentence vector, provided that the sentence is shorter
                than `max_len`. If not, the end token is not added. Unlike
                `add_start_symbol`, enabling this option **does not alter**
                the maximum length.

        Returns:
            Tensor of indices of the words.
        """
        # First, lookup the symbols in the vocabulary
        index_tensor = self._string_to_index.lookup(sentences)
        index_shape = tf.shape(index_tensor, out_type=tf.int64)
        batch = index_shape[0]
        max_time = index_shape[1]

        # Second, include end symbols (if needed)
        if add_end_symbol:
            # Append one more column of paddings to the end
            end_paddings = tf.expand_dims(
                tf.fill([batch], tf.to_int64(PAD_TOKEN_INDEX)), axis=1)
            index_tensor = tf.concat([index_tensor, end_paddings], axis=1)

            lengths = tf.reduce_sum(
                tf.to_int64(tf.not_equal(index_tensor, PAD_TOKEN_INDEX)),
                axis=1)

            # In case PAD_TOKEN_INDEX is not zero, we should compute the
            # difference to be able to transform PAD to END.
            end_token_diff = END_TOKEN_INDEX - PAD_TOKEN_INDEX

            sparse_indices = tf.stack([tf.range(batch), lengths], axis=1)
            sparse_values = tf.fill([batch], tf.to_int64(end_token_diff))

            mask = tf.SparseTensor(
                sparse_indices, sparse_values, [batch, max_time + 1])

            dense_mask = tf.sparse.to_dense(mask)
            index_tensor += dense_mask

        # Third, chop the index tensor to adequate length.
        # We assume that the batch is already padded to the maximum length (+1)
        if max_length is not None:
            index_tensor = index_tensor[:, :max_length]

        # Finally, prepend the start symbol.
        if add_start_symbol:
            starts = tf.expand_dims(
                tf.fill([batch], tf.to_int64(START_TOKEN_INDEX)), axis=1)
            index_tensor = tf.concat([starts, index_tensor], axis=1)

        return index_tensor

    def indices_to_strings(self, vectors: tf.Tensor) -> tf.Tensor:
        """Convert tensors of indexes of vocabulary items to lists of words.

        Arguments:
            vectors: An int Tensor with indices to the vocabulary.

        Returns:
            A string Tensor with the corresponding words.
        """
        return self._index_to_string.lookup(vectors)

    def vectors_to_sentences(
            self,
            vectors: Union[List[np.ndarray], np.ndarray]) -> List[List[str]]:
        """Convert vectors of indexes of vocabulary items to lists of words.

        Arguments:
            vectors: TIME-MAJOR List of vectors of vocabulary indices.

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
                      encoding: str = "utf-8") -> None:
        """Save the vocabulary as a wordlist.

        The file is ordered by the ids of words.
        This function is used mainly for embedding visualization.

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

        with open(path, "w", encoding=encoding) as output_file:
            log("Storing vocabulary without frequencies.")

            for word in self._vocabulary:
                output_file.write("{}\n".format(word))


def log_sample(vocabulary: List[str], size: int = 5) -> None:
    """Log a sample of the vocabulary.

    Arguments:
        size: How many sample words to log.
    """
    if size > len(vocabulary):
        log("Vocabulary: {}".format(vocabulary))
    else:
        sample_ids = np.random.permutation(np.arange(len(vocabulary)))[:size]
        log("Sample of the vocabulary: {}".format(
            [vocabulary[i] for i in sample_ids]))


def pad_batch(sentences: List[List[str]],
              max_length: int = None,
              add_start_symbol: bool = False,
              add_end_symbol: bool = False) -> List[List[str]]:

    max_len = max(len(s) for s in sentences)
    if add_end_symbol:
        max_len += 1

    if max_length is not None:
        max_len = min(max_length, max_len)

    padded_sentences = []
    for sent in sentences:
        if add_end_symbol:
            padded = (sent + [END_TOKEN] + [PAD_TOKEN] * max_len)[:max_len]
        else:
            padded = (sent + [PAD_TOKEN] * max_len)[:max_len]

        if add_start_symbol:
            padded.insert(0, START_TOKEN)
        padded_sentences.append(padded)

    return padded_sentences


def sentence_mask(sentences: tf.Tensor) -> tf.Tensor:
    return tf.to_float(tf.not_equal(sentences, PAD_TOKEN_INDEX))
