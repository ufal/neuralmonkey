#!/usr/bin/env python3.5
"""Test init methods of encoders."""

import unittest
import copy

from typing import Dict, List, Any, Iterable

from neuralmonkey.encoders.recurrent import SentenceEncoder
from neuralmonkey.encoders.sentence_cnn_encoder import SentenceCNNEncoder
from neuralmonkey.model.sequence import EmbeddedSequence
from neuralmonkey.vocabulary import Vocabulary


VOCABULARY = Vocabulary(["ich", "bin", "der", "walrus"])
INPUT_SEQUENCE = EmbeddedSequence("seq", VOCABULARY, "marmelade", 300)

SENTENCE_ENCODER_GOOD = {
    "name": ["encoder"],
    "vocabulary": [VOCABULARY],
    "data_id": ["marmelade"],
    "embedding_size": [20],
    "rnn_size": [30],
    "max_input_len": [None, 15],
    "dropout_keep_prob": [0.5, 1.],
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
}

TRANSFORMER_ENCODER_GOOD = {
    "name": ["transformer_encoder"],
    "input_sequence": [INPUT_SEQUENCE],
    "ff_hidden_size": [10],
    "depth": [6],
    "n_heads": [3],
    "dropout_keep_prob": [0.5],
}

TRANSFORMER_ENCODER_BAD = {
    "nonexistent": ["ahoj"],
    "name": [None, 1],
    "input_sequence": [0, None, VOCABULARY],
    "ff_hidden_size": [-1, 0, "ahoj", 3.14, VOCABULARY, SentenceEncoder, None],
    "depth": [-1, "ahoj", 3.14, SentenceEncoder, None],
    "n_heads": [-1, "ahoj", 3.14, SentenceEncoder, None],
    "dropout_keep_prob": [0.0, 0, -1.0, 2.0, "ahoj", VOCABULARY, None]
}

SENTENCE_CNN_ENCODER_GOOD = {
    "name": ["cnn_encoder"],
    "input_sequence": [INPUT_SEQUENCE],
    "segment_size": [10],
    "highway_depth": [11],
    "rnn_size": [30],
    "filters": [[(2, 10)], [(3, 20), (4, 10)]],
    "dropout_keep_prob": [0.5, 1.],
    "use_noisy_activations": [False]
}

SENTENCE_CNN_ENCODER_BAD = {
    "nonexistent": ["ahoj"],
    "name": [None, 1],
    "input_sequence": [0, None, VOCABULARY],
    "segment_size": [-1, 0, "ahoj", 3.14, VOCABULARY, None],
    "highway_depth": [-1, "ahoj", 3.14, SentenceEncoder, None],
    "rnn_size": [-1, 0, "ahoj", 3.14, VOCABULARY, SentenceEncoder, None],
    "filters": ["ahoj", [], [(0, 0)], [(1, 2, 3)], [VOCABULARY, None],
                [(None, None)]],
    "dropout_keep_prob": [0.0, 0, -1.0, 2.0, "ahoj", VOCABULARY, None],
    "use_noisy_activations": [None, SentenceEncoder]
}


def traverse_combinations(
        params: Dict[str, List[Any]],
        partial_params: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    params = copy.copy(params)

    if params:
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
                except Exception:
                    print("FAILED '{}', configuration: {}".format(
                        encoder_type, str(options)))
                    raise

        for good_param_combo in traverse_combinations(good_params, {}):
            try:
                options = copy.copy(good_param_combo)
                options["name"] = "{}_{}".format(options["name"], name_suffix)
                name_suffix += 1

                encoder_type(**options)
            except Exception:
                print("Good param combo FAILED: {}, configuration: {}".format(
                    encoder_type, str(options)))
                raise

    def test_sentence_encoder(self):
        with self.assertRaises(Exception):
            # pylint: disable=no-value-for-parameter
            # on purpose, should fail
            SentenceEncoder()
            # pylint: enable=no-value-for-parameter

        self._run_constructors(SentenceEncoder,
                               SENTENCE_ENCODER_GOOD,
                               SENTENCE_ENCODER_BAD)

    def test_sentence_cnn_encoder(self):
        with self.assertRaises(Exception):
            # pylint: disable=no-value-for-parameter
            # on purpose, should fail
            SentenceCNNEncoder()
            # pylint: enable=no-value-for-parameter

        self._run_constructors(SentenceCNNEncoder,
                               SENTENCE_CNN_ENCODER_GOOD,
                               SENTENCE_CNN_ENCODER_BAD)


if __name__ == "__main__":
    unittest.main()
