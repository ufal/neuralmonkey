from typing import cast, Any, Callable, Iterable, Optional, List

import tensorflow as tf

from typeguard import check_argument_types
from neuralmonkey.nn.projection import multilayer_projection
from neuralmonkey.dataset import Dataset
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.checking import assert_shape

# tests: lint, mypy

# pylint: disable=too-many-instance-attributes


class SequenceRegressor(ModelPart):
    """A simple MLP regression over encoders.

    The API pretends it is an RNN decoder which always generates a sequence of
    length exactly one.
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoders: List[Any],
                 data_id: str,
                 layers: Optional[List[int]] = None,
                 activation_fn: Callable[[tf.Tensor], tf.Tensor]=tf.tanh,
                 dropout_keep_prob: float = 0.5,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        assert check_argument_types()

        self.encoders = encoders
        self.data_id = data_id
        self.layers = layers
        self.activation_fn = activation_fn
        self.dropout_keep_prob = dropout_keep_prob
        self.max_output_len = 1

        with tf.variable_scope(name):
            self.learning_step = tf.get_variable(
                "learning_step", [], trainable=False,
                initializer=tf.constant_initializer(0))

            self.dropout_placeholder = \
                tf.placeholder(tf.float32, name="dropout_plc")
            self.gt_inputs = tf.placeholder(tf.float32, shape=[None],
                                            name="targets")

            mlp_input = tf.concat([enc.encoded for enc in encoders], 1)
            # TODO extend it to output into multidimensional space
            layers.append(1)
            mlp = multilayer_projection(
                mlp_input, layers, activation=self.activation_fn,
                dropout_plc=self.dropout_placeholder)

            assert_shape(mlp, [-1, 1])

            self.predicted = mlp
            self.cost = tf.reduce_mean(
                tf.square(mlp - tf.expand_dims(self.gt_inputs, 1)))

            tf.summary.scalar(
                'val_optimization_cost', self.cost,
                collections=["summary_val"])
            tf.summary.scalar(
                'train_optimization_cost',
                self.cost, collections=["summary_train"])
    # pylint: enable=too-many-arguments

    @property
    def train_loss(self):
        return self.cost

    @property
    def runtime_loss(self):
        return self.cost

    @property
    def decoded(self):
        return self.predicted

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        sentences = cast(Iterable[List[str]],
                         dataset.get_series(self.data_id, allow_none=True))

        sentences_list = list(sentences) if sentences is not None else None

        fd = {}  # type: FeedDict
        if sentences_list is not None:
            fd[self.gt_inputs] = list(zip(*sentences_list))[0]

        if train:
            fd[self.dropout_placeholder] = self.dropout_keep_prob
        else:
            fd[self.dropout_placeholder] = 1.0

        return fd
