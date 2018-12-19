from typing import Callable, Dict, List

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decorators import tensor
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.model.stateful import Stateful
from neuralmonkey.nn.mlp import MultilayerPerceptron
from neuralmonkey.vocabulary import Vocabulary


class Classifier(ModelPart):
    """A simple MLP classifier over encoders.

    The API pretends it is an RNN decoder which always generates a sequence of
    length exactly one.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoders: List[Stateful],
                 vocabulary: Vocabulary,
                 data_id: str,
                 layers: List[int],
                 activation_fn: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
                 dropout_keep_prob: float = 0.5,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Construct a new instance of the sequence classifier.

        Args:
            name: Name of the decoder. Should be unique accross all Neural
                Monkey objects
            encoders: Input encoders of the decoder
            vocabulary: Target vocabulary
            data_id: Target data series
            layers: List defining structure of the NN. Ini example:
                    layers=[100,20,5] ;creates classifier with hidden layers of
                                       size 100, 20, 5 and one output layer
                                       depending on the size of vocabulary
            activation_fn: activation function used on the output of each
                           hidden layer.
            dropout_keep_prob: Probability of keeping a value during dropout
        """
        check_argument_types()
        ModelPart.__init__(self, name, reuse, save_checkpoint, load_checkpoint,
                           initializers)

        self.encoders = encoders
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.layers = layers
        self.activation_fn = activation_fn
        self.dropout_keep_prob = dropout_keep_prob
        self.max_output_len = 1
    # pylint: enable=too-many-arguments

    @property
    def input_types(self) -> Dict[str, tf.DType]:
        return {self.data_id: tf.string}

    @property
    def input_shapes(self) -> Dict[str, tf.TensorShape]:
        return {self.data_id: tf.TensorShape([None])}

    @tensor
    def gt_inputs(self) -> tf.Tensor:
        return self.vocabulary.strings_to_indices(
            self.dataset[self.data_id], self.max_output_len)[:, 0]

    @tensor
    def _mlp(self) -> MultilayerPerceptron:
        mlp_input = tf.concat([enc.output for enc in self.encoders], 1)
        return MultilayerPerceptron(
            mlp_input, self.layers, self.dropout_keep_prob,
            len(self.vocabulary), activation_fn=self.activation_fn,
            train_mode=self.train_mode)

    @property
    def loss_with_decoded_ins(self) -> tf.Tensor:
        return self.loss_with_gt_ins

    @property
    def cost(self) -> tf.Tensor:
        tf.summary.scalar(
            "train_optimization_cost",
            self.loss_with_gt_ins, collections=["summary_train"])

        return self.loss_with_gt_ins

    # pylint: disable=no-member
    # this is for the _mlp attribute (pylint property bug)
    @tensor
    def loss_with_gt_ins(self) -> tf.Tensor:
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self._mlp.logits, labels=self.gt_inputs))

    @tensor
    def decoded_seq(self) -> tf.Tensor:
        return tf.expand_dims(self._mlp.classification, 0)

    @tensor
    def decoded_logits(self) -> tf.Tensor:
        return tf.expand_dims(self._mlp.logits, 0)

    @tensor
    def runtime_logprobs(self) -> tf.Tensor:
        return tf.expand_dims(tf.nn.log_softmax(self._mlp.logits), 0)
    # pylint: enable=no-member

    @property
    def train_loss(self):
        return self.loss_with_gt_ins

    @property
    def runtime_loss(self):
        return self.loss_with_decoded_ins

    @property
    def decoded(self):
        return self.decoded_seq
