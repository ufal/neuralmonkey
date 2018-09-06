"""This module provides functionality needed to work with word2vec files."""

from typing import Callable, List

import numpy as np
from typeguard import check_argument_types

from neuralmonkey.vocabulary import Vocabulary, is_special_token, UNK_TOKEN


class Word2Vec:

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        """Loads the word2vec file."""
        check_argument_types()

        # Create the vocabulary object, load the words and vectors from the
        # file

        self.vocabulary = Vocabulary()
        self.embedding_vectors = []  # type: List[np.ndarray]
        emb_size = None

        with open(path, encoding=encoding) as f_data:
            for line in f_data:
                fields = line.split()
                word = fields[0]
                vector = np.fromiter((float(x) for x in fields[1:]),
                                     dtype=np.float)

                if emb_size is None:
                    emb_size = vector.shape[0]

                    # Add zero embeddings for padding, start, and end token
                    self.embedding_vectors.append(np.zeros(emb_size))
                    self.embedding_vectors.append(np.zeros(emb_size))
                    self.embedding_vectors.append(np.zeros(emb_size))
                    # Add placeholder for embedding of the unknown symbol
                    self.embedding_vectors.append(None)
                else:
                    assert vector.shape[0] == emb_size

                # Embedding of unknown token should be at index 3 to match the
                # vocabulary implementation
                if is_special_token(word):
                    assert word == UNK_TOKEN
                    self.embedding_vectors[3] = vector
                else:
                    self.vocabulary.add_word(word)
                    self.embedding_vectors.append(vector)

        assert self.embedding_vectors[3] is not None
        assert emb_size is not None

    @property
    def vocabulary(self) -> Vocabulary:
        """Get a vocabulary object generated from this word2vec instance."""
        return self.vocabulary

    @property
    def embeddings(self) -> np.ndarray:
        """Get the embedding matrix."""
        return np.array(self.embedding_vectors)


def get_word2vec_initializer(w2v: Word2Vec) -> Callable:
    """Create a word2vec initializer.

    A higher-order function that can be called from configuration.
    """
    check_argument_types()

    def init(shape: List[int], _) -> np.ndarray:

        if shape != w2v.embeddings.shape:
            raise ValueError(
                "Shapes of model and word2vec embeddings do not match. "
                "Word2Vec shape: {}, Should have been: {}"
                .format(w2v.embeddings.shape, shape))
        return w2v.embeddings

    return init


def word2vec_vocabulary(word2vec: Word2Vec) -> Vocabulary:
    """Return the vocabulary from a word2vec object.

    This is a helper method used from configuration.
    """
    check_argument_types()
    return word2vec.vocabulary
