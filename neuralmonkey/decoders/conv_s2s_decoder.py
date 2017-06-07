"""From a paper Convolutional Sequence to Sequence Learning

http://arxiv.org/abs/1705.03122
"""

import tensorflow as tf
import numpy as np
from typing import cast, Iterable, Any, List, Union, Type, Optional
from typeguard import check_argument_types

from neuralmonkey.vocabulary import Vocabulary, START_TOKEN, END_TOKEN_INDEX
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.encoders.conv_s2s_encoder import ConvolutionalSentenceEncoder
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.logging import log
from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.decorators import tensor
from neuralmonkey.nn.projection import glu, linear

# todo remove ipdb
import ipdb


class ConvolutionalSentenceDecoder(ModelPart):

    def __init__(self,
                 name: str,
                 encoder: ConvolutionalSentenceEncoder,
                 vocabulary: Vocabulary,
                 data_id: str,
                 max_output_len: int,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        self.encoder = encoder
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.max_output_len = max_output_len
        self.embedding_size = encoder.embedding_size

        with self.use_scope():
            with tf.variable_scope("decoder") as self.step_scope:
                decoded = self.decoding_loop(train_mode=True)

    def decoding_loop(self, train_mode):
        decoded_words = []
        prev_word = None

        for i in range(self.max_output_len):
            if i > 0:
                self.step_scope.reuse_variables()

            if i == 0:
                decoded_words.append(self.go_symbols)
            elif train_mode:
                # TODO zkontrolovat, ze to dela to, co ma
                decoded_words.append(self.train_targets[i - 1])
            else:
                assert prev_word is not None
                decoded_words.append(prev_word)

            embedded = self.embed(decoded_words)

        ipdb.set_trace()















    @tensor
    def go_symbols(self):
        return tf.placeholder(
            tf.int32, shape=[None], name="decoder_go_symbols")

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    @tensor
    def word_embeddings(self) -> tf.Tensor:
        # initialization in the same way as in original CS2S implementation
        with tf.variable_scope("decoder_input_projection"):
            return tf.get_variable(
                "word_embeddings", [self.vocabulary_size, self.embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.1))

    @tensor
    def order_embeddings(self) -> tf.Tensor:
        # initialization in the same way as in original CS2S implementation
        with tf.variable_scope("decoder_input_projection"):
            return tf.get_variable(
                "order_embeddings", [self.max_output_len, self.embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.1))

    def embed(self, inputs) -> tf.Tensor:
        # shape (batch, time, embedding_size)
        emb_inp = tf.nn.embedding_lookup(self.word_embeddings, inputs)
        ordering_additive = tf.expand_dims(self.order_embeddings, 0)
        return emb_inp + ordering_additive

    # pylint: disable=no-self-use
    @tensor
    def train_targets(self) -> tf.Tensor:
        # shape (batch_max_len, batch_size)
        return tf.placeholder(tf.int32, shape=[None, None], name="targets")

    # @tensor
    # def train_weights(self) -> tf.Tensor:
    #     return tf.placeholder(
    #         tf.float32, shape=[None, None], name="padding_weights")

    @tensor
    def train_mode(self) -> tf.Tensor:
        return tf.placeholder(tf.bool, name="train_mode")
    # pylint: enable=no-self-use

    @property
    def train_loss(self) -> tf.Tensor:
        return self.cost

    @property
    def runtime_loss(self) -> tf.Tensor:
        return self.cost

    @tensor
    def cost(self) -> tf.Tensor:
        return cost

    @tensor
    def decoded(self) -> tf.Tensor:
        return decoded

    @tensor
    def logprobs(self) -> tf.Tensor:
        return tf.nn.log_softmax(self.logits)

    @tensor
    def logits(self) -> tf.Tensor:
        return logits

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = {}  # type: FeedDict

        fd[self.train_mode] = train

        go_symbol_idx = self.vocabulary.get_word_index(START_TOKEN)
        fd[self.go_symbols] = np.full([len(dataset)], go_symbol_idx,
                                      dtype=np.int32)

        sentences = cast(Iterable[List[str]],
                         dataset.get_series(self.data_id, allow_none=True))

        if sentences is None and train:
            raise ValueError("When training, you must feed "
                             "reference sentences")

        sentences_list = list(sentences) if sentences is not None else None
        if sentences is not None:
            # train_mode=False, since we don't want to <unk>ize target words!
            inputs, weights = self.vocabulary.sentences_to_tensor(
                sentences_list, self.max_output_len, train_mode=False,
                add_start_symbol=False, add_end_symbol=True,
                pad_to_max_len=True)

            assert inputs.shape == (self.max_output_len, len(sentences_list))
            assert weights.shape == (self.max_output_len, len(sentences_list))

            fd[self.train_targets] = inputs
            # fd[self.train_weights] = weights

        return fd
