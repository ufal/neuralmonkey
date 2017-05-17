from typing import Optional, Tuple, List

from typeguard import check_argument_types
import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.nn.utils import dropout
from neuralmonkey.vocabulary import Vocabulary


class SequenceCNNEncoder(ModelPart):
    """Encoder processing a sequence using a CNN."""

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 embedding_size: int,
                 filters: List[Tuple[int, int]],
                 max_input_len: Optional[int] = None,
                 dropout_keep_prob: float = 1.0,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None) -> None:
        """Creates a new instance of the CNN sequence encoder.

        Based on: Yoon Kim: Convolutional Neural Networks for Sentence
        Classification (http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf)

        Arguments:
            vocabulary: Input vocabulary
            data_id: Identifier of the data series fed to this encoder
            name: An unique identifier for this encoder
            max_input_len: Maximum length of an encoded sequence
            embedding_size: The size of the embedding vector assigned
                to each word
            filters: Specification of CNN filters. It is a list of tuples
                specifying the filter size and number of channels.
            dropout_keep_prob: The dropout keep probability
                (default 1.0)
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        assert check_argument_types()

        self.vocabulary = vocabulary
        self.data_id = data_id
        self.max_input_len = max_input_len

        with self.use_scope():
            self.train_mode = tf.placeholder(tf.bool, shape=[],
                                             name="train_mode")

            self.inputs = tf.placeholder(tf.int32,
                                         shape=[None, None],
                                         name="encoder_input")

            self._input_mask = tf.placeholder(
                tf.float32, shape=[None, None],
                name="encoder_padding")

            with tf.variable_scope("input_projection"):
                self.embedding_matrix = tf.get_variable(
                    "word_embeddings", [len(vocabulary), embedding_size],
                    initializer=tf.random_normal_initializer(stddev=0.01))
                embedded_inputs = dropout(
                    tf.nn.embedding_lookup(self.embedding_matrix, self.inputs),
                    dropout_keep_prob,
                    self.train_mode)

            pooled_outputs = []
            for filter_size, num_filters in filters:
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, num_filters]
                    w_filter = tf.get_variable(
                        "conv_W", filter_shape,
                        initializer=tf.random_uniform_initializer(-0.5, 0.5))
                    b_filter = tf.get_variable(
                        "conv_bias", [num_filters],
                        initializer=tf.constant_initializer(0.0))
                    conv = tf.nn.conv1d(
                        embedded_inputs,
                        w_filter,
                        stride=1,
                        padding="VALID",
                        name="conv")

                    # Apply nonlinearity
                    conv_relu = tf.nn.relu(tf.nn.bias_add(conv, b_filter))

                    # Max-pooling over the outputs
                    pooled = tf.reduce_max(conv_relu, 1)
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            self.encoded = tf.concat(pooled_outputs, axis=1)
    # pylint: enable=too-many-arguments,too-many-locals

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
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
            list(sentences), self.max_input_len, pad_to_max_len=False,
            train_mode=train)

        # as sentences_to_tensor returns lists of shape (time, batch),
        # we need to transpose
        fd[self.inputs] = list(zip(*vectors))
        fd[self._input_mask] = list(zip(*paddings))

        return fd
