from typing import cast, Iterable, List

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.model.feedable import FeedDict
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.model.stateful import TemporalStateful
from neuralmonkey.tf_utils import get_variable
from neuralmonkey.vocabulary import Vocabulary, END_TOKEN


class CTCDecoder(ModelPart):
    """Connectionist Temporal Classification.

    See `tf.nn.ctc_loss`, `tf.nn.ctc_greedy_decoder` etc.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoder: TemporalStateful,
                 vocabulary: Vocabulary,
                 data_id: str,
                 max_length: int = None,
                 merge_repeated_targets: bool = False,
                 merge_repeated_outputs: bool = True,
                 beam_width: int = 1,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        ModelPart.__init__(self, name, reuse, save_checkpoint, load_checkpoint,
                           initializers)

        self.encoder = encoder
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.max_length = max_length

        self.merge_repeated_targets = merge_repeated_targets
        self.merge_repeated_outputs = merge_repeated_outputs
        self.beam_width = beam_width
    # pylint: enable=too-many-arguments

    # pylint: disable=no-self-use
    @tensor
    def train_targets(self) -> tf.Tensor:
        return tf.sparse_placeholder(tf.int32, name="targets")
    # pylint: disable=no-self-use

    @tensor
    def decoded(self) -> tf.Tensor:
        if self.beam_width == 1:
            decoded, _ = tf.nn.ctc_greedy_decoder(
                inputs=self.logits, sequence_length=self.encoder.lengths,
                merge_repeated=self.merge_repeated_outputs)
        else:
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                inputs=self.logits, sequence_length=self.encoder.lengths,
                beam_width=self.beam_width,
                merge_repeated=self.merge_repeated_outputs)

        return tf.sparse_tensor_to_dense(
            tf.sparse_transpose(decoded[0]),
            default_value=self.vocabulary.get_word_index(END_TOKEN))

    @property
    def train_loss(self) -> tf.Tensor:
        return self.cost

    @property
    def runtime_loss(self) -> tf.Tensor:
        return self.cost

    @tensor
    def cost(self) -> tf.Tensor:
        loss = tf.nn.ctc_loss(
            labels=self.train_targets, inputs=self.logits,
            sequence_length=self.encoder.lengths,
            preprocess_collapse_repeated=self.merge_repeated_targets,
            ignore_longer_outputs_than_inputs=True,
            ctc_merge_repeated=self.merge_repeated_outputs)

        return tf.reduce_sum(loss)

    @tensor
    def logits(self) -> tf.Tensor:
        vocabulary_size = len(self.vocabulary)

        encoder_states = self.encoder.temporal_states

        weights = get_variable(
            name="state_to_word_W",
            shape=[encoder_states.shape[2], vocabulary_size + 1],
            initializer=tf.random_uniform_initializer(-0.5, 0.5))

        biases = get_variable(
            name="state_to_word_b",
            shape=[vocabulary_size + 1],
            initializer=tf.zeros_initializer())

        # To multiply 3-D matrix (encoder hidden states) by a 2-D matrix
        # (weights), we use 1-by-1 convolution (similar trick can be found in
        # attention computation)

        encoder_states = tf.expand_dims(encoder_states, 2)
        weights_4d = tf.expand_dims(tf.expand_dims(weights, 0), 0)

        multiplication = tf.nn.conv2d(
            encoder_states, weights_4d, [1, 1, 1, 1], "SAME")
        multiplication_3d = tf.squeeze(multiplication, squeeze_dims=[2])

        biases_3d = tf.expand_dims(tf.expand_dims(biases, 0), 0)

        logits = multiplication_3d + biases_3d
        return tf.transpose(logits, perm=[1, 0, 2])  # time major

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = ModelPart.feed_dict(self, dataset, train)

        sentences = cast(Iterable[List[str]],
                         dataset.maybe_get_series(self.data_id))

        if sentences is None and train:
            raise ValueError("When training, you must feed "
                             "reference sentences")

        if sentences is not None:
            vectors, paddings = self.vocabulary.sentences_to_tensor(
                list(sentences), train_mode=train, max_len=self.max_length)

            # sentences_to_tensor returns time-major tensors, targets need to
            # be batch-major
            vectors = vectors.T
            paddings = paddings.T

            # Need to convert the data to a sparse representation
            bool_mask = (paddings > 0.5)
            indices = np.stack(np.where(bool_mask), axis=1)
            values = vectors[bool_mask]

            fd[self.train_targets] = tf.SparseTensorValue(
                indices=indices, values=values,
                dense_shape=vectors.shape)

        return fd
