from typing import cast, Any, Callable, Iterable, Optional, List

import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.nn.mlp import MultilayerPerceptron
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
                 layers: Optional[List[int]]=None,
                 activation_fn: Callable[[tf.Tensor], tf.Tensor]=tf.tanh,
                 dropout_keep_prob: float=0.5,
                 dropout_input: float = 1.0,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        self.encoders = encoders
        self.data_id = data_id
        self.layers = layers
        self.activation_fn = activation_fn
        self.dropout_keep_prob = dropout_keep_prob
        self.max_output_len = 1
        self.dropout_input = dropout_input

        with tf.variable_scope(name):
            self.learning_step = tf.get_variable(
                "learning_step", [], trainable=False,
                initializer=tf.constant_initializer(0))

            self.dropout_placeholder = \
                tf.placeholder(tf.float32, name="dropout_plc")
            self.dropout_input_placeholder = tf.placeholder(tf.float32, name="dropout_plc")
            self.gt_inputs = tf.placeholder(tf.float32, shape=[None],
                                            name="targets")

            mlp_input = tf.concat([enc.encoded for enc in encoders], 1)

            # todo pridano jako hack pro fleos
            if self.dropout_input < 1.0:
                mlp_input = tf.nn.dropout(mlp_input, self.dropout_input_placeholder)

            # mlp_input = tf.subtract(encoders[0].encoded, encoders[1].encoded)
            mlp = MultilayerPerceptron(
                mlp_input, layers, self.dropout_placeholder, 1,
                activation_fn=self.activation_fn)

            logits = tf.sigmoid(mlp.logits)

            assert_shape(logits, [-1, 1])

            self.predicted = logits

            # sigm =tf.sigmoid(tf.squeeze(logits, [1]))
            sigm = tf.squeeze(logits, [1])

            # clipped = tf.clip_by_value(sigm, 1.0e-15, 1.0 - 1.0e-15)

            # logloss: - GOLD * log (PREDICTED) - (1-GOLD) * log (1-PREDICTED)
            # logloss = (-tf.multiply(self.gt_inputs, tf.log(clipped))
            #            -tf.multiply((tf.subtract(1.0, self.gt_inputs)), tf.log(tf.subtract(1.0, clipped))))

            # logloss: (GOLD-1) * log (1-PREDICTED) - GOLD * log (PREDICTED)
            logloss = tf.subtract(
                        tf.multiply(tf.subtract(self.gt_inputs, 1.0),
                                    tf.log(tf.clip_by_value(tf.subtract(1.0, sigm), 1.0e-15, 1.0))),
                        tf.multiply(self.gt_inputs, tf.log(tf.clip_by_value(sigm, 1.0e-15, 1.0))))

            # logits = tf.Print(logits,
            #                 [logits],
            #                 summarize=1000)

            self.cost = tf.reduce_mean(logloss)
            # self.cost = tf.reduce_mean(tf.square(logits + self.gt_inputs))

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

    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        sentences = cast(Iterable[List[str]],
                         dataset.get_series(self.data_id, allow_none=True))

        sentences_list = list(sentences) if sentences is not None else None

        fd = {}  # type: FeedDict
        if sentences_list is not None:
            fd[self.gt_inputs] = list(zip(*sentences_list))[0]

        if train:
            fd[self.dropout_placeholder] = self.dropout_keep_prob
            fd[self.dropout_input_placeholder] = self.dropout_input
        else:
            fd[self.dropout_placeholder] = 1.0
            fd[self.dropout_input_placeholder] = 1.0

        return fd
