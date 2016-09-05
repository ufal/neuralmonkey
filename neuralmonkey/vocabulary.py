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


class Vocabulary(collections.Sized):
    def __init__(self, tokenized_text: List[str]=None,
                 unk_sample_prob: float=0.0) -> None:
        """Create a new instance of a vocabulary.

        Arguments:
            tokenized_text: The initial list of words to add.
        """
        self.word_to_index = {} # type: Dict[str, int]
        self.index_to_word = [] # type: List[str]
        self.word_count = {} # type: Dict[str, int]

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
            train: bool=False) -> Tuple[np.array, np.array]:
        """Generate the tensor representation for the provided sentences.

        Arguments:
            sentences: List of sentences as lists of tokens.
            max_len: Maximum lengh of a sentence toward which they will be
                     padded to.
            train: Flag whether we are training or not
                   (enables/disables unk sampling).

        Returns:
            A tensor representing the sentences ((max_length + 2) x batch)
            and a weight tensor ((max_length + 1) x batch) that inidicates
            padding.
        """
        start_indices = [np.repeat(self.get_word_index(START_TOKEN),
                                   len(sentences))]
        pad_indices = [np.repeat(self.get_word_index(PAD_TOKEN), len(sentences))
                       for _ in range(max_len + 1)]

        word_indices = np.stack(start_indices + pad_indices)
        weights = [np.zeros([len(sentences)]) for _ in range(max_len + 1)]

        for i in range(max_len + 1):
            for j, sent in enumerate(sentences):
                if i < len(sent):
                    word_indices[i + 1][j] = (
                        self.get_unk_sampled_word_index(sent[i])
                        if train else self.get_word_index(sent[i]))
                    weights[i][j] = 1.0

                elif i == len(sent):
                    word_indices[i + 1][j] = self.get_word_index(END_TOKEN)
                    weights[i][j] = 1.0

        return word_indices, weights


    def vectors_to_sentences(self, vectors: List[np.array]) -> List[List[str]]:
        """Convert vectors of indexes of vocabulary items to lists of words.

        Arguments:
            vectors: List of vectors of vocabulary indices.

        Returns:
            List of lists of words.
        """
        sentences = [[] for _ in range(vectors[0].shape[0])]

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


    @staticmethod
    def from_datasets(
            datasets: List[Dataset], series_ids: List[str], max_size: int,
            unk_sample_prob: float=0.5) -> 'Vocabulary':
        """Create new vocabulary instance from datasets.

        Arguments:
            datasets: A list of datasets from which to create the vocabulary
            series_ids: A list of ids of series of the datasets that should be
                        used producing the vocabulary
            max_size: The maximum size of the vocabulary
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
        return vocabulary


    @staticmethod
    def from_pickled(path: str) -> 'Vocabulary':
        """Create new vocabulary instance from a pickle file.

        Arguments:
            path: The path to the pickle file

        Returns:
            The newly created vocabulary.
        """
        with open(path, 'rb') as f_pickle:
            vocabulary = pickle.load(f_pickle)
        assert isinstance(vocabulary, Vocabulary)

        log("Pickled vocabulary loaded. Size: {} words".format(len(vocabulary)))
        vocabulary.log_sample()
        return vocabulary


    @staticmethod
    def from_bpe(path: str, encoding: str="utf-8") -> 'Vocabulary':
        """Create new closed vocabulary instance from BPE merge file.

        Arguments:
            path: The path to the merge file.
            encoding: The encoding of the merge file (defaults to UTF-8)

        Returns:
            The new instance of the Vocabulary
        """
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
