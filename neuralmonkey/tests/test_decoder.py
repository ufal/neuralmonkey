#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
""" Unit tests for the decoder. (Tests only initialization so far) """

import unittest
import tensorflow as tf

from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.vocabulary import Vocabulary


class TestDecoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tf.reset_default_graph()

    def setUp(self):
        self.decoder_params = dict(
            encoders=[],
            vocabulary=Vocabulary(["a", "b", "c"]),
            data_id="foo",
            name="test-decoder",
            max_output_len=5,
            dropout_keep_prob=1.0,
            embedding_size=10,
            rnn_size=10)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_init(self):
        decoder = Decoder(**self.decoder_params)
        self.assertIsNotNone(decoder)

    def test_max_output_len(self):
        dparams = self.decoder_params
        dparams["max_output_len"] = -10
        with self.assertRaises(ValueError):
            Decoder(**dparams)

    def test_dropout(self):
        dparams = self.decoder_params
        dparams["dropout_keep_prob"] = -0.5
        with self.assertRaises(ValueError):
            Decoder(**dparams)

        dparams["dropout_keep_prob"] = 1.5
        with self.assertRaises(ValueError):
            Decoder(**dparams)

    def test_embedding_size(self):
        dparams = self.decoder_params
        dparams["embedding_size"] = None
        with self.assertRaises(ValueError):
            dec = Decoder(**dparams)
            print(dec.embedding_size)

        dparams["embedding_size"] = -10
        with self.assertRaises(ValueError):
            Decoder(**dparams)

    def test_cell_type(self):
        dparams = self.decoder_params

        dparams.update({"rnn_cell": "bogus_cell"})
        with self.assertRaises(ValueError):
            Decoder(**dparams)

        for cell_type in ("GRU", "LSTM", "NematusGRU"):
            print(dparams)
            dparams["rnn_cell"] = cell_type
            dparams["name"] = "test-decoder-{}".format(cell_type)
            Decoder(**dparams)


if __name__ == "__main__":
    unittest.main()
