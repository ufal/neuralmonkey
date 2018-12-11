"""Word2vec plug-in module.

This module provides functionality needed to work with word2vec files.
"""

from typing import Callable, List

import numpy as np
from typeguard import check_argument_types

from neuralmonkey.vocabulary import Vocabulary, SPECIAL_TOKENS


class Word2Vec:

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        """Load the word2vec file."""
        check_argument_types()

        # Create the vocabulary object, load the words and vectors from the
        # file

        words = []  # List[str]
        embedding_vectors = []  # type: List[np.ndarray]

        with open(path, encoding=encoding) as f_data:

            header = next(f_data)
            emb_size = int(header.split()[1])

            # Add zero embeddings for padding, start, and end token
            embedding_vectors.append(np.zeros(emb_size))
            embedding_vectors.append(np.zeros(emb_size))
            embedding_vectors.append(np.zeros(emb_size))
            # Add placeholder for embedding of the unknown symbol
            embedding_vectors.append(None)

            for line in f_data:
                fields = line.split()
                word = fields[0]
                vector = np.fromiter((float(x) for x in fields[1:]),
                                     dtype=np.float)

                assert vector.shape[0] == emb_size

                # Embedding of unknown token should be at index 3 to match the
                # vocabulary implementation
                if word in SPECIAL_TOKENS:
                    embedding_vectors[SPECIAL_TOKENS.index(word)] = vector
                else:
                    words.append(word)
                    embedding_vectors.append(vector)

        self.vocab = Vocabulary(words)

        assert embedding_vectors[3] is not None
        assert emb_size is not None

        self.embedding_matrix = np.stack(embedding_vectors)

    @property
    def vocabulary(self) -> Vocabulary:
        """Get a vocabulary object generated from this word2vec instance."""
        return self.vocab

    @property
    def embeddings(self) -> np.ndarray:
        """Get the embedding matrix."""
        return self.embedding_matrix


def get_word2vec_initializer(w2v: Word2Vec) -> Callable:
    """Create a word2vec initializer.

    A higher-order function that can be called from configuration.
    """
    check_argument_types()

    def init(shape: List[int], **kwargs) -> np.ndarray:
        if shape != list(w2v.embeddings.shape):
            raise ValueError(
                "Shapes of model and word2vec embeddings do not match. "
                "Word2Vec shape: {}, Should have been: {}"
                .format(w2v.embeddings.shape, shape))
        return w2v.embeddings

    return init


def word2vec_vocabulary(w2v: Word2Vec) -> Vocabulary:
    """Return the vocabulary from a word2vec object.

    This is a helper method used from configuration.
    """
    check_argument_types()
    return w2v.vocabulary
