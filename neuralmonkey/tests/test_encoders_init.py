#!/usr/bin/env python3.5
"""Test init methods of encoders."""

import random
import string
import unittest

from neuralmonkey.decoding_function import Attention, CoverageAttention
from neuralmonkey.encoders.numpy_encoder import (VectorEncoder,
                                                 PostCNNImageEncoder)
from neuralmonkey.encoders.sentence_encoder import SentenceEncoder
from neuralmonkey.encoders.sentence_cnn_encoder import SentenceCNNEncoder
from neuralmonkey.tests.test_vocabulary import VOCABULARY

SENTENCE_ENCODER = {
    "max_input_len": ([1, 10, 100], [-1, 0, "ahoj", 3.14]),
    "embedding_size": ([1, 10, 100], [-1, 0, "ahoj", 3.14]),
    "rnn_size": ([1, 10, 100], [-1, 0, "ahoj", 3.14]),
    "dropout_keep_prob": ([0.1, 1.0], [-1, 1.5, "huhu"]),
    "attention_type": ([Attention, CoverageAttention, None], ["baf"]),
    "use_noisy_activations": ([False], [0, "", "bflm"]),
    "data_id": (["marmelade"], [0]),
    "vocabulary": ([VOCABULARY], [0])
}

SENTENCE_CNN_ENCODER = {
    "max_input_len": ([1, 10, 100], [-1, 0, "ahoj", 3.14]),
    "embedding_size": ([1, 10, 100], [-1, 0, "ahoj", 3.14]),
    "rnn_size": ([1, 10, 100], [-1, 0, "ahoj", 3.14]),
    "dropout_keep_prob": ([0.1, 1.0], [-1, 1.5, "huhu"]),
    "attention_type": ([Attention, CoverageAttention, None], ["baf"]),
    "use_noisy_activations": ([False], [0, "", "bflm"]),
    "data_id": (["marmelade"], [0]),
    "vocabulary": ([VOCABULARY], [0]),
    "highway_depth": ([1, 10, 100], [-1, "ahoj", 3.14]),
    "filters": ([[(2, 10)], [(2, 50), (3, 20), (4, 10)]],
                ["ahoj", [], [(0, 0)]]),
    "segment_size": ([1, 10, 10], [-1, 0, "ahoj", 3.14])
}

VECTOR_ENCODER = {
    "data_id": (["marmelade"], [0]),
    "dimension": ([1, 100], [0, -1, "hoj", 3.14]),
    "output_shape": ([1, 100], [0, -1, "hoj", 3.14])
}

POST_CNN_IMAGE_ENCODER = {
    "data_id": (["marmelade"], [0]),
    "attention_type": ([Attention, CoverageAttention, None], ["baf"]),
    "output_shape": ([1, 100], [0, -1, "hoj", 3.14]),
    "input_shape": ([[1, 2, 3], [10, 20, 3]],
                    [3, [10, 20], [-1, 10, 20], "baf"])
}


def get_all_combinations(rest_arg_names, params):
    """Recursively get all combinations of arguments."""
    this_param = rest_arg_names[0]
    good_values, bad_values = params[this_param]
    if len(rest_arg_names) == 1:
        for value in good_values:
            yield {this_param: value}, True
        for value in bad_values:
            yield {this_param: value}, False
    else:
        rest_combinations = get_all_combinations(rest_arg_names[1:], params)
        for combination, good in rest_combinations:
            for value in good_values:
                res = {this_param: value}
                res.update(combination)
                yield res, good
            for value in bad_values:
                res = {this_param: value}
                res.update(combination)
                yield res, False


class TestEncodersInit(unittest.TestCase):
    def _construct_all_objects(self, enc_type, params):
        """Construct all object with given type and param combinations."""
        arg_names = list(params.keys())

        all_args = list(get_all_combinations(arg_names, params))

        for args, good in all_args:
            name = ''.join(random.choice(string.ascii_lowercase)
                           for i in range(20))
            log_args = ", ".join(["{}={}".format(k, v)
                                  for k, v in args.items()])
            args['name'] = name

            try:
                if good:
                    enc_type(**args)
                else:
                    with self.assertRaises(Exception):
                        enc_type(**args)
            except Exception:
                print("FAILED '{}', configuration: {}".format(
                    enc_type, log_args))
                raise

    def test_sentence_encoder(self):
        self._construct_all_objects(SentenceEncoder, SENTENCE_ENCODER)

    def test_sentence_cnn_encoder(self):
        self._construct_all_objects(SentenceCNNEncoder, SENTENCE_CNN_ENCODER)

    def test_vector_encoder(self):
        self._construct_all_objects(VectorEncoder, VECTOR_ENCODER)

    def test_post_cnn_encoder(self):
        self._construct_all_objects(PostCNNImageEncoder,
                                    POST_CNN_IMAGE_ENCODER)


if __name__ == "__main__":
    unittest.main()
