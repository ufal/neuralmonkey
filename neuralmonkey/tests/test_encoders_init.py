#!/usr/bin/env python3.5
"""Test init methods of encoders."""

import random
import string
import unittest
import copy

from typing import Dict, List, Any, Iterable

from neuralmonkey.decoding_function import Attention, CoverageAttention
from neuralmonkey.encoders.numpy_encoder import (VectorEncoder,
                                                 PostCNNImageEncoder)
from neuralmonkey.encoders.sentence_encoder import SentenceEncoder
from neuralmonkey.encoders.sentence_cnn_encoder import SentenceCNNEncoder
from neuralmonkey.tests.test_vocabulary import VOCABULARY


SENTENCE_ENCODER_GOOD = {
    "name": ["encoder"],
    "vocabulary": [VOCABULARY],
    "data_id": ["marmelade"],
    "embedding_size": [20],
    "rnn_size": [30],
    "max_input_len": [None, 15],
    "dropout_keep_prob": [0.5, 1.],
    "attention_type": [Attention, CoverageAttention, None],
    "attention_fertility": [1],
    "use_noisy_activations": [False],
    "parent_encoder": [None]
}

SENTENCE_ENCODER_BAD = {
    "nonexistent": ["ahoj"],
    "name": [None, 1],
    "vocabulary": [0, None, "ahoj", dict()],
    "data_id": [0, None, VOCABULARY],
    "embedding_size": [-1, 0, "ahoj", 3.14, VOCABULARY, SentenceEncoder, None],
    "rnn_size": [-1, 0, "ahoj", 3.14, VOCABULARY, SentenceEncoder, None],
    "max_input_len": [-1, 0, "ahoj", 3.14, VOCABULARY, SentenceEncoder],
    "dropout_keep_prob": [0.0, 0, -1.0, 2.0, "ahoj", VOCABULARY, None],
    "attention_type": [-1, "ahoj", VOCABULARY, SentenceEncoder],
    "attention_fertility": [None, "ahoj", VOCABULARY, SentenceEncoder],
    "use_noisy_activations": [None, SentenceEncoder],
    "parent_encoder": [0, "ahoj", VOCABULARY, SentenceEncoder]
}

SENTENCE_CNN_ENCODER_GOOD = {
    "name": ["cnn_encoder"],
    "vocabulary": [VOCABULARY],
    "data_id": ["marmelade"],
    "embedding_size": [20],
    "segment_size": [10],
    "highway_depth": [11],
    "rnn_size": [30],
    "filters": [[(2, 10)], [(3, 20), (4, 10)]],
    "max_input_len": [None, 17],
    "dropout_keep_prob": [0.5, 1.],
    "attention_type": [Attention, CoverageAttention, None],
    "use_noisy_activations": [False]
}

SENTENCE_CNN_ENCODER_BAD = {
    "nonexistent": ["ahoj"],
    "name": [None, 1],
    "vocabulary": [0, None, "ahoj", dict()],
    "data_id": [0, None, VOCABULARY],
    "embedding_size": [-1, 0, "ahoj", 3.14, VOCABULARY, SentenceEncoder, None],
    "segment_size": [-1, 0, "ahoj", 3.14, VOCABULARY, None],
    "highway_depth": [-1, "ahoj", 3.14, SentenceEncoder, None],
    "rnn_size": [-1, 0, "ahoj", 3.14, VOCABULARY, SentenceEncoder, None],
    "filters": ["ahoj", [], [(0, 0)], [(1, 2, 3)], [VOCABULARY, None],
                [(None, None)]],
    "max_input_len": [-1, 0, "ahoj", 3.14, VOCABULARY, SentenceEncoder],
    "dropout_keep_prob": [0.0, 0, -1.0, 2.0, "ahoj", VOCABULARY, None],
    "attention_type": [-1, "ahoj", VOCABULARY, SentenceEncoder],
    "attention_fertility": [None, "ahoj", VOCABULARY, SentenceEncoder],
    "use_noisy_activations": [None, SentenceEncoder]
}

VECTOR_ENCODER_GOOD = {
    "name": ["vector_encoder"],
    "dimension": [10],
    "data_id": ["marmelade"],
    "output_shape": [1, None, 100]
}

VECTOR_ENCODER_BAD = {
    "nonexistent": ["ahoj"],
    "name": [None, 1],
    "dimension": [0, -1, "ahoj", 3.14, VOCABULARY, SentenceEncoder, None],
    "data_id": [3.14, VOCABULARY, None],
    "output_shape": [0, -1, "ahoj", 3.14, VOCABULARY]
}

POST_CNN_IMAGE_ENCODER_GOOD = {
    "name": ["vector_encoder"],
    "input_shape": [[1, 2, 3], [10, 20, 3]],
    "output_shape": [10],
    "data_id": ["marmelade"],
    "attention_type": [Attention, CoverageAttention, None]
}

POST_CNN_IMAGE_ENCODER_BAD = {
    "nonexistent": ["ahoj"],
    "name": [None, 1],
    "data_id": [3.14, VOCABULARY, None],
    "attention_type": [-1, "ahoj", VOCABULARY, SentenceEncoder],
    "output_shape": [0, -1, "hoj", 3.14, None, VOCABULARY, SentenceEncoder],
    "input_shape": [3, [10, 20], [-1, 10, 20], "123", "ahoj", 3.14,
                    VOCABULARY, []]
}


def traverse_combinations(
        params: Dict[str, List[Any]],
        partial_params: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    params = copy.copy(params)

    if len(params) > 0:
        pivot_key, values = params.popitem()

        for val in values:
            partial_params[pivot_key] = val
            yield from traverse_combinations(params, partial_params)
    else:
        yield partial_params


class TestEncodersInit(unittest.TestCase):

    def _run_constructors(self, encoder_type, good_params, bad_params):
        good_index = 0
        good_options = {par: value[good_index]
                        for par, value in good_params.items()}

        name_suffix = 0

        for key, bad_values in bad_params.items():
            for value in bad_values:
                options = copy.copy(good_options)
                options[key] = value
                if key != "name":
                    options["name"] = "{}_{}".format(options["name"],
                                                     name_suffix)
                name_suffix += 1

                try:
                    with self.assertRaises(Exception):
                        encoder_type(**options)
                except:
                    print("FAILED '{}', configuration: {}".format(
                        encoder_type, str(options)))
                    raise

        for good_param_combo in traverse_combinations(good_params, {}):
            try:
                options = copy.copy(good_param_combo)
                options["name"] = "{}_{}".format(options["name"], name_suffix)
                name_suffix += 1

                encoder_type(**options)
            except:
                print("Good param combo FAILED: {}, configuration: {}".format(
                    encoder_type, str(options)))
                raise

    def test_sentence_encoder(self):
        with self.assertRaises(Exception):
            SentenceEncoder()

        self._run_constructors(SentenceEncoder,
                               SENTENCE_ENCODER_GOOD,
                               SENTENCE_ENCODER_BAD)

    def test_sentence_cnn_encoder(self):
        with self.assertRaises(Exception):
            SentenceCNNEncoder()

        self._run_constructors(SentenceCNNEncoder,
                               SENTENCE_CNN_ENCODER_GOOD,
                               SENTENCE_CNN_ENCODER_BAD)

    def test_vector_encoder(self):
        with self.assertRaises(Exception):
            VectorEncoder()

        self._run_constructors(VectorEncoder,
                               VECTOR_ENCODER_GOOD,
                               VECTOR_ENCODER_BAD)

    def test_post_cnn_encoder(self):
        with self.assertRaises(Exception):
            PostCNNImageEncoder()

        self._run_constructors(PostCNNImageEncoder,
                               POST_CNN_IMAGE_ENCODER_GOOD,
                               POST_CNN_IMAGE_ENCODER_BAD)


if __name__ == "__main__":
    unittest.main()
