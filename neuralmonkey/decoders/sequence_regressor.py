from typing import Callable, Dict, List

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.nn.projection import multilayer_projection
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.model.stateful import Stateful
from neuralmonkey.decorators import tensor


class SequenceRegressor(ModelPart):
    """A simple MLP regression over encoders.

    The API pretends it is an RNN decoder which always generates a sequence of
    length exactly one.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoders: List[Stateful],
                 data_id: str,
                 layers: List[int] = None,
                 activation_fn: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
                 dropout_keep_prob: float = 1.0,
                 dimension: int = 1,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        ModelPart.__init__(self, name, reuse, save_checkpoint, load_checkpoint,
                           initializers)

        self.encoders = encoders
        self.data_id = data_id
        self.max_output_len = 1
        self.dimension = dimension

        self._layers = layers
        self._activation_fn = activation_fn
        self._dropout_keep_prob = dropout_keep_prob
    # pylint: enable=too-many-arguments

    @property
    def input_types(self) -> Dict[str, tf.DType]:
        return {self.data_id: tf.float32}

    @property
    def input_shapes(self) -> Dict[str, tf.TensorShape]:
        return {self.data_id: tf.TensorShape([None])}

    @tensor
    def train_inputs(self) -> tf.Tensor:
        return self.dataset[self.data_id]

    @tensor
    def _mlp_input(self):
        return tf.concat([enc.output for enc in self.encoders], 1)

    @tensor
    def _mlp_output(self):
        return multilayer_projection(
            self._mlp_input, self._layers, self.train_mode,
            self._activation_fn, self._dropout_keep_prob)

    @tensor
    def predictions(self):
        return tf.layers.dense(
            self._mlp_output, self.dimension, name="output_projection")

    @tensor
    def cost(self):
        cost = tf.reduce_mean(tf.square(
            self.predictions - tf.expand_dims(self.train_inputs, 1)))

        tf.summary.scalar("optimization_cost", cost,
                          collections=["summary_val", "summary_train"])

        return cost

    @property
    def train_loss(self):
        return self.cost

    @property
    def runtime_loss(self):
        return self.cost

    @property
    def decoded(self):
        return self.predictions
