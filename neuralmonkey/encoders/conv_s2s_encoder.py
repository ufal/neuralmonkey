
"""From a paper Convolutional Sequence to Sequence Learning

http://arxiv.org/abs/1705.03122
"""

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.logging import log
from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.decorators import tensor


class ConvolutionalSentenceEncoder(ModelPart):#, Attentive):


    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 embedding_size: int,

                 max_input_len: int,

                 encoder_stack_size: int,
                 encoder_hidden_sizes: List[int],

                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:

        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        #Attentive.__init__(self, None) # TODO attention

        #assert check_argument_types()

        self.vocabulary = vocabulary
        self.data_id = data_id
        self.max_input_len = max_input_len
        self.encoder_stack_size = encoder_stack_size

        if max_input_len is not None and max_input_len <= 0:
            raise ValueError("Input length must be a positive integer.")
        if embedding_size <= 0:
            raise ValueError("Embedding size must be a positive integer.")
        if encoder_stack_size <= 0:
            raise ValueError("Encoder stack size must be a positive integer.")
        if len(encoder_hidden_sizes) != encoder_stack_size:
            raise ValueError("Number of encoder hidden sizes must be equal to "
                             "encoder stack size.")


        log("Initializing convolutional seq2seq encoder, name {}"
            .format(self.name))

        with self.use_scope():




            convoluted_ord = tf.nn.conv1d(
                self.ordered_embedded_inputs, self.convolution_filters, 1,
                "SAME")




    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)


    @tensor
    def ordered_embedded_inputs(self) -> tf.Tensor:
        # shape (batch, time, embedding_size)
        emb_inp = tf.nn.embedding_lookup(self.word_embeddings, self.inputs)
        ordering_additive = tf.expand_dims(self.order_embeddings, 0)
        return emb_inp + ordering_additive

    @tensor
    def convolution_filters(self) -> tf.Tensor:
        # shape (encoder_stack_size, embedding_size, embedding_size)
        with tf.variable_scope("convolution"):
            log("TODO Better initialize convolution filters", color="red")
            return tf.get_variable(
                "convolution_filters", [self.encoder_stack_size, self.embedding_size, self.embedding_size]



    @tensor
    def word_embeddings(self) -> tf.Tensor:
        with tf.variable_scope("input_projection"):
            log("TODO Better initialize word embeddings", color="red")
            return tf.get_variable(
                "word_embeddings", [self.vocabulary_size, self.embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.01))

    @tensor
    def order_embeddings(self) -> tf.Tensor:
        with tf.variable_scope("input_projection"):
            log("TODO Better initialize order embeddings", color="red")
            return tf.get_variable(
                "order_embeddings", [self.max_input_len, self.embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.01))

    @tensor
    def inputs(self):
        # shape (batch, time)
        return tf.placeholder(tf.int32, shape=[None, None],
                              name="conv_s2s_encoder_inputs")

    @tensor
    def train_mode(self):
        # scalar tensor
        return tf.placeholder(tf.bool, shape=[], name="mode_placeholder")

    @tensor
    def input_mask(self):
        # shape (batch, time)
        return tf.placeholder(tf.float32, shape=[None, None],
            name="conv_s2s_encoder_input_mask")

    @tensor
    def sentence_lengths(self) -> tf.Tensor:
        # shape (batch)
        return tf.to_int32(tf.reduce_sum(self.input_mask, 0))
