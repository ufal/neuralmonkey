# tests: lint, mypy

from typing import Optional, Any, Tuple, List

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.logging import log
from neuralmonkey.nn.noisy_gru_cell import NoisyGRUCell
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell
from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary

import ipdb

# pylint: disable=invalid-name
AttType = Any  # Type[] or union of types do not work here
RNNCellTuple = Tuple[tf.contrib.rnn.RNNCell, tf.contrib.rnn.RNNCell]
# pylint: enable=invalid-name
# TODO REMOVE DEBUGGING SKIP-FILE
# pylint: skip-file


# pylint: disable=too-many-instance-attributes
class FleosEncoder(ModelPart, Attentive):
    """A class that manages parts of the computation graph that are
    used for encoding of input sentences.
    """

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 data_id: str,
                 name: str,
                 vocabulary: Vocabulary=None,
                 max_input_len: int=0,
                 embedding_size: int=0,
                 dropout_keep_prob: float=1.0,
                 use_noisy_activations: bool=False,
                 version: int=0,
                 blocks: List[Any]=None,
                 parent_encoder: Optional["FleosEncoder"]=None,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:

        if parent_encoder is not None:
            name = parent_encoder.name
            vocabulary = parent_encoder.vocabulary
            max_input_len = parent_encoder.max_input_len
            embedding_size = parent_encoder.embedding_size
            dropout_keep_prob = parent_encoder.dropout_keep_prob
            use_noisy_activations = parent_encoder.use_noisy_activations
            version = parent_encoder.version
            blocks = parent_encoder.blocks
            save_checkpoint = None  # neukladej duplikat
            load_checkpoint = None

        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        assert check_argument_types()

        self.vocabulary = vocabulary
        self.data_id = data_id
        self.version = version
        self.blocks = blocks

        self.max_input_len = max_input_len
        self.embedding_size = embedding_size
        self.dropout_keep_prob = dropout_keep_prob
        self.use_noisy_activations = use_noisy_activations
        self.parent_encoder = parent_encoder

        #debugging variable, or how many inputs is visible at one position in vector of particular layer
        self.visibility_range = 1

        log("Initializing sentence encoder, name: '{}'"
            .format(self.name))

        with tf.variable_scope(self.name):
            self._create_input_placeholders()

            self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout_placeholder")

            with tf.variable_scope('input_projection'):
                self._create_embedding_matrix()

                # todo experimentovat s dropingovanim vstupu, na characterech by to mohlo pomoci

                embedded_inputs = self._embed(self.inputs)  # type: tf.Tensor

            packed = tf.expand_dims(embedded_inputs, -2)
            output_state = packed

            # packed = self._dropout(packed)
            if version == 0:
                h_conv1 = self.conv2d(packed, 64)
                h_conv2 = self.conv2d(h_conv1, 64)
                h_pool1 = self.max_pool_3x3(h_conv2)

                h_conv3 = self.conv2d(h_pool1, 128)
                h_conv4 = self.conv2d(h_conv3, 128)
                h_pool2 = self.max_pool_3x3(h_conv4)

                h_conv5 = self.conv2d(h_pool2, 256)
                h_conv6 = self.conv2d(h_conv5, 256)
                h_pool3 = self.max_pool_3x3(h_conv6)

                h_conv7 = self.conv2d(h_pool3, 512)
                h_conv8 = self.conv2d(h_conv7, 512, None)
                output_state = self.max_pool_3x3(h_conv8)

                final_state = tf.reduce_mean(output_state, 1)
            elif version == 1:
                for i, block in enumerate(blocks):
                    if len(block)>2:
                        padding = block[2] # can be VALID or SAME
                    else:
                        padding="SAME"
                    output_state = self.conv2d(output_state, block[0], tf.nn.relu, block[1], padding)
                    if i == len(blocks)-1:
                        output_state = self.conv2d(output_state, block[0], None, block[1], padding)
                    else:
                        output_state = self.conv2d(output_state, block[0], tf.nn.relu, block[1], padding)
                    output_state = self.max_pool_3x3(output_state, padding)
                    final_state = tf.reduce_mean(output_state, 1)
            elif version == 2:
                for i, block in enumerate(blocks):
                    if len(block)>2:
                        padding = block[2] # can be VALID or SAME
                    else:
                        padding="SAME"
                    output_state = self.conv2d(output_state, block[0], tf.nn.relu, block[1], padding)
                    if i == len(blocks)-1:
                        output_state = self.conv2d(output_state, block[0], None, block[1], padding, stride=2)
                    else:
                        output_state = self.conv2d(output_state, block[0], tf.nn.relu, block[1], padding, stride=2)
                    output_state = self.max_pool_3x3(output_state, padding)
                    final_state = tf.reduce_mean(output_state, 1)
            elif version == 3:
                for i, block in enumerate(blocks):
                    if len(block)>2:
                        padding = block[2] # can be VALID or SAME
                    else:
                        padding="SAME"
                    output_state = self.conv2d(output_state, block[0], tf.nn.relu, block[1], padding)
                    output_state = self.conv2d(output_state, block[0], tf.nn.relu, block[1], padding, stride=2)
                    if i == len(blocks)-1:
                        output_state = self.conv2d(output_state, block[0], None, block[1], padding)
                    else:
                        output_state = self.conv2d(output_state, block[0], tf.nn.relu, block[1], padding)
                    output_state = self.max_pool_3x3(output_state, padding)
                    final_state = tf.reduce_mean(output_state, 1)
            elif version == 4:
                for i, block in enumerate(blocks):
                    if len(block)>2:
                        padding = block[2] # can be VALID or SAME
                    else:
                        padding="SAME"
                    output_state = self.conv2d(output_state, block[0], tf.nn.relu, block[1]+2, padding)
                    if i == len(blocks)-1:
                        output_state = self.conv2d(output_state, block[0], None, block[1], padding, stride=2)
                    else:
                        output_state = self.conv2d(output_state, block[0], tf.nn.relu, block[1], padding, stride=2)
                    output_state = self.max_pool_3x3(output_state, padding)
                    final_state = tf.reduce_mean(output_state, 1)
            elif version == 5:
                for i, block in enumerate(blocks):
                    if len(block)>2:
                        padding = block[2] # can be VALID or SAME
                    else:
                        padding="SAME"
                    output_state = self.conv2d(output_state, block[0], tf.nn.relu, block[1]+2, padding)
                    output_state = self.conv2d(output_state, block[0], tf.nn.relu, block[1], padding, stride=2)
                    if i == len(blocks)-1:
                        output_state = self.conv2d(output_state, block[0], None, block[1]+2, padding)
                    else:
                        output_state = self.conv2d(output_state, block[0], tf.nn.relu, block[1]+2, padding)
                    output_state = self.max_pool_3x3(output_state, padding)
                    final_state = tf.reduce_mean(output_state, 1)
            elif version == 6:
                # nearly exactly as in "Very Deep Convolutional Networks"

                # it represents (features, number of blocks)
                blocks = [(64, 2), (128, 2), (256, 2), (512, 2)]

                # temp convolution
                output_state = self.conv2d(output_state, 64, tf.nn.relu, 3)

                for i, block in enumerate(blocks):
                    for j in range(block[1]):
                        output_state = self.resconv(output_state, block[0])

                # TODO v clanku od Coneau je nesorteny a navic tahle funkce nejede pres druhou dimenzi ale posledni coz je spatne a take neni derivovatelna nejspis
                # final_state, _ = tf.nn.top_k(output_state, 8, True)

                final_state = tf.reduce_mean(output_state, 1)
            elif version == 7:
                # inspired by Deep Residual Learning for Image Recognition

                # blocks = it represents (features, number of blocks)

                # temp convolution
                output_state = self.conv2d(output_state, 64, tf.nn.relu, 3)

                for i, block in enumerate(blocks):
                    for j in range(block[1]):
                        output_state = self.resnetconv(output_state, block[0])

                final_state = tf.reduce_mean(output_state, 1)


            self.encoded = tf.contrib.layers.flatten(final_state)

        log("Sentence encoder initialized")

    def resconv(self, x, features):
        if int(x.shape[-1]) == features:
            output = self.conv2d(x, features, tf.nn.relu)
            shortcut = x
        else:
            output = self.conv2d(x, features, tf.nn.relu, stride=2)
            # nahradit linearni projekci
            shortcut = self.conv2d(x, features, None, stride=2, kernel=1)

        output = self.conv2d(output, features, None)
        output = tf.nn.relu(tf.add(output, shortcut))
        return output

    def resnetconv(self, x, features):
        if int(x.shape[-1]) == features:
            output = self.conv2d(x, 64, tf.nn.relu, kernel=1)
            shortcut = x
        else:
            output = self.conv2d(x, 64, tf.nn.relu, stride=2, kernel=1)
            # nahradit linearni projekci
            shortcut = self.conv2d(x, features, None, stride=2, kernel=1)

        output = self.conv2d(output, 64, tf.nn.relu, kernel=3)
        output = self.conv2d(output, features, None, kernel=1)
        output = tf.nn.relu(tf.add(output, shortcut))
        return output

    def conv2d(self, x, features, activation_fn=tf.nn.relu, kernel=3, padding="SAME", stride=1):
        #padding=SAME zatim nejlepsi
        # what about dropout ... ne, pouzivam batch normalization
        # TODO ideas ... pouzit biases misto normalizace, hrat si se stride
        output = tf.contrib.layers.convolution2d(inputs=x, num_outputs=features,
                                                 kernel_size=[kernel, 1], stride=[stride, 1],
                                                 padding=padding, activation_fn=activation_fn,
                                                 normalizer_fn=tf.contrib.layers.batch_norm)
        self.visibility_range += (kernel - 1) + 2*(stride -1)
        if self.parent_encoder is None:
            log("Conv from {} into {}. Visibility range {} chars.".format(x.shape, output.shape, self.visibility_range))
        return output

    def max_pool_3x3(self, x, padding="SAME", kernel=3, stride=2):
        # in literature only two types are used,
        # most common maxout with kernel=2, stride=2, or
        # overlapping maxout with kernel=3, stride=2
        output = tf.nn.max_pool(x, ksize=[1, kernel, 1, 1], strides=[1, stride, 1, 1], padding=padding)
        self.visibility_range += (kernel - 1) + 2*(stride -1)
        if self.parent_encoder is None:
            log("Maxpool from {} into {}. Visibility range {} chars.".format(x.shape, output.shape, self.visibility_range))
        return output

    # def maxout(self, inputs, num_units, axis=None):
    #     shape = inputs.get_shape().as_list()
    #     if shape[0] is None:
    #         shape[0] = -1
    #     if axis is None:  # Assume that channel is the last dimension
    #         axis = -1
    #     num_channels = shape[axis]
    #     if num_channels % num_units:
    #         raise ValueError('number of features({}) is not '
    #                          'a multiple of num_units({})'.format(num_channels, num_units))
    #     shape[axis] = num_units
    #     shape += [num_channels // num_units]
    #     outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    #     return outputs

    @property
    def _attention_tensor(self):
        return self.__attention_tensor

    @property
    def _attention_mask(self):
        return self._input_mask

    @property
    def vocabulary_size(self):
        return len(self.vocabulary)

    def _create_input_placeholders(self):
        """Creates input placeholder nodes in the computation graph"""
        self.train_mode = tf.placeholder(tf.bool, shape=[],
                                         name="mode_placeholder")

        self.inputs = tf.placeholder(tf.int32,
                                     shape=[None, self.max_input_len],
                                     name="encoder_input")

        self._input_mask = tf.placeholder(
            tf.float32, shape=[None, self.max_input_len],
            name="encoder_padding")

        self.sentence_lengths = tf.to_int32(
            tf.reduce_sum(self._input_mask, 1))

    def _create_embedding_matrix(self):
        """Create variables and operations for embedding the input words.

        If parent encoder is specified, we reuse its embedding matrix
        """
        # NOTE the note from the decoder's embedding matrix function applies
        # here also
        if self.parent_encoder is not None:
            self.embedding_matrix = self.parent_encoder.embedding_matrix
        else:
            self.embedding_matrix = tf.get_variable(
                "word_embeddings", [self.vocabulary_size, self.embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.01))

    def _dropout(self, variable: tf.Tensor) -> tf.Tensor:
        """Perform dropout on a variable

        Arguments:
            variable: The variable to be dropped out

        Returns:
            The dropped value of the variable
        """
        if self.dropout_keep_prob == 1.0:
            return variable

        # TODO as soon as TF.12 is out, remove this line and use train_mode
        # directly
        train_mode_batch = tf.fill(tf.shape(variable)[:1], self.train_mode)
        dropped_value = tf.nn.dropout(variable, self.dropout_keep_prob)
        return tf.where(train_mode_batch, dropped_value, variable)

    def _embed(self, inputs: tf.Tensor) -> tf.Tensor:
        """Embed the input using the embedding matrix and apply dropout

        Arguments:
            inputs: The Tensor to be embedded and dropped out.
        """
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        return self._dropout(embedded)


    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        """Populate the feed dictionary with the encoder inputs.

        Encoder input placeholders:
            ``encoder_input``: Stores indices to the vocabulary,
                shape (batch, time)
            ``encoder_padding``: Stores the padding (ones and zeros,
                indicating valid words and positions after the end
                of sentence, shape (batch, time)
            ``train_mode``: Boolean scalar specifying the mode (train
                vs runtime)

        Arguments:
            dataset: The dataset to use
            train: Boolean flag telling whether it is training time
        """
        # pylint: disable=invalid-name
        fd = {}  # type: FeedDict
        fd[self.train_mode] = train
        sentences = dataset.get_series(self.data_id)

        vectors, paddings = self.vocabulary.sentences_to_tensor(
            list(sentences), self.max_input_len, train_mode=train)

        # as sentences_to_tensor returns lists of shape (time, batch),
        # we need to transpose
        fd[self.inputs] = list(zip(*vectors))
        fd[self._input_mask] = list(zip(*paddings))

        if train:
            fd[self.dropout_placeholder] = self.dropout_keep_prob
        else:
            fd[self.dropout_placeholder] = 1.0

        return fd


class FleosEncoderClone(ModelPart, Attentive):
    """A class that manages parts of the computation graph that are
    used for encoding of input sentences.
    """

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 data_id: str,
                 parent_encoder: FleosEncoder=None,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:

        ModelPart.__init__(self, parent_encoder.name, save_checkpoint, load_checkpoint)

        assert check_argument_types()

        self.vocabulary = parent_encoder.vocabulary
        self.data_id = data_id
        self.max_input_len = parent_encoder.max_input_len
        self.embedding_size = parent_encoder.embedding_size
        self.dropout_keep_prob = parent_encoder.dropout_keep_prob
        self.use_noisy_activations = parent_encoder.use_noisy_activations
        self.parent_encoder = parent_encoder

        log("Initializing sentence encoder, name: '{}'"
            .format(parent_encoder.name))

        with tf.variable_scope(self.name):
            self._create_input_placeholders()

            self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout_placeholder")

            with tf.variable_scope('input_projection'):
                self._create_embedding_matrix()

                # todo experimentovat s dropingovanim vstupu, na characterech by to mohlo pomoci

                embedded_inputs = self._embed(self.inputs)  # type: tf.Tensor

            packed = tf.expand_dims(embedded_inputs, -2)

            # if version == 2:
            #     packed = self._dropout(packed)
            h_conv1 = self.conv2d(packed, 64)
            h_conv2 = self.conv2d(h_conv1, 64)
            h_pool1 = self.max_pool_3x3(h_conv2)

            h_conv3 = self.conv2d(h_pool1, 128)
            h_conv4 = self.conv2d(h_conv3, 128)
            h_pool2 = self.max_pool_3x3(h_conv4)

            h_conv5 = self.conv2d(h_pool2, 256)
            h_conv6 = self.conv2d(h_conv5, 256)
            h_pool3 = self.max_pool_3x3(h_conv6)

            h_conv7 = self.conv2d(h_pool3, 512)
            h_conv8 = self.conv2d(h_conv7, 512, None)
            h_pool4 = self.max_pool_3x3(h_conv8)

            final_state = tf.reduce_mean(h_pool4, 1)

            self.encoded = tf.contrib.layers.flatten(final_state)

        log("Sentence encoder initialized")

    def conv2d(self, x, features, activation_fn=tf.nn.relu):
        return tf.contrib.layers.convolution2d(inputs=x, num_outputs=features, kernel_size=[3, 1], stride=[1, 1], padding="SAME", activation_fn=activation_fn,
                                               normalizer_fn=tf.contrib.layers.batch_norm)

    def max_pool_3x3(self, x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # def maxout(self, inputs, num_units, axis=None):
    #     shape = inputs.get_shape().as_list()
    #     if shape[0] is None:
    #         shape[0] = -1
    #     if axis is None:  # Assume that channel is the last dimension
    #         axis = -1
    #     num_channels = shape[axis]
    #     if num_channels % num_units:
    #         raise ValueError('number of features({}) is not '
    #                          'a multiple of num_units({})'.format(num_channels, num_units))
    #     shape[axis] = num_units
    #     shape += [num_channels // num_units]
    #     outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    #     return outputs

    @property
    def _attention_tensor(self):
        return self.__attention_tensor

    @property
    def _attention_mask(self):
        return self._input_mask

    @property
    def vocabulary_size(self):
        return len(self.vocabulary)

    def _create_input_placeholders(self):
        """Creates input placeholder nodes in the computation graph"""
        self.train_mode = tf.placeholder(tf.bool, shape=[],
                                         name="mode_placeholder")

        self.inputs = tf.placeholder(tf.int32,
                                     shape=[None, self.max_input_len],
                                     name="encoder_input")

        self._input_mask = tf.placeholder(
            tf.float32, shape=[None, self.max_input_len],
            name="encoder_padding")

        self.sentence_lengths = tf.to_int32(
            tf.reduce_sum(self._input_mask, 1))

    def _create_embedding_matrix(self):
        """Create variables and operations for embedding the input words.

        If parent encoder is specified, we reuse its embedding matrix
        """
        # NOTE the note from the decoder's embedding matrix function applies
        # here also
        if self.parent_encoder is not None:
            self.embedding_matrix = self.parent_encoder.embedding_matrix
        else:
            self.embedding_matrix = tf.get_variable(
                "word_embeddings", [self.vocabulary_size, self.embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.01))

    def _dropout(self, variable: tf.Tensor) -> tf.Tensor:
        """Perform dropout on a variable

        Arguments:
            variable: The variable to be dropped out

        Returns:
            The dropped value of the variable
        """
        if self.dropout_keep_prob == 1.0:
            return variable

        # TODO as soon as TF.12 is out, remove this line and use train_mode
        # directly
        train_mode_batch = tf.fill(tf.shape(variable)[:1], self.train_mode)
        dropped_value = tf.nn.dropout(variable, self.dropout_keep_prob)
        return tf.where(train_mode_batch, dropped_value, variable)

    def _embed(self, inputs: tf.Tensor) -> tf.Tensor:
        """Embed the input using the embedding matrix and apply dropout

        Arguments:
            inputs: The Tensor to be embedded and dropped out.
        """
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        return self._dropout(embedded)

    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        """Populate the feed dictionary with the encoder inputs.

        Encoder input placeholders:
            ``encoder_input``: Stores indices to the vocabulary,
                shape (batch, time)
            ``encoder_padding``: Stores the padding (ones and zeros,
                indicating valid words and positions after the end
                of sentence, shape (batch, time)
            ``train_mode``: Boolean scalar specifying the mode (train
                vs runtime)

        Arguments:
            dataset: The dataset to use
            train: Boolean flag telling whether it is training time
        """
        # pylint: disable=invalid-name
        fd = {}  # type: FeedDict
        fd[self.train_mode] = train
        sentences = dataset.get_series(self.data_id)

        vectors, paddings = self.vocabulary.sentences_to_tensor(
            list(sentences), self.max_input_len, train_mode=train)

        # as sentences_to_tensor returns lists of shape (time, batch),
        # we need to transpose
        fd[self.inputs] = list(zip(*vectors))
        fd[self._input_mask] = list(zip(*paddings))

        if train:
            fd[self.dropout_placeholder] = self.dropout_keep_prob
        else:
            fd[self.dropout_placeholder] = 1.0

        return fd
