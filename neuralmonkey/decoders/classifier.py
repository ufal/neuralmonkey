from typing import Callable, List

import tensorflow as tf
from tensorflow.python.framework import ops
from typeguard import check_argument_types

from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.model.model_part import ModelPart, FeedDict, InitializerSpecs
from neuralmonkey.model.stateful import Stateful
from neuralmonkey.nn.mlp import MultilayerPerceptron
from neuralmonkey.decorators import tensor


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
                 adversarial: bool = False,
                 dropout_keep_prob: float = 0.5,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Construct a new instance of the sequence classifier.

        Args:
            name: Name of the decoder. Should be unique across all Neural
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
            adversarial: Flag enabling adversarial classification that
                simultaneously trains the classifier and reverse the gradient
                to the encoders, such that worsen representation for the
                classifier.
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
        self.adversarial = adversarial
    # pylint: enable=too-many-arguments

    @tensor
    def targets(self) -> tf.Tensor:
        return [tf.placeholder(tf.int32, [None], "targets")]

    @tensor
    def mlp_input(self) -> tf.Tensor:
        mlp_input = tf.concat([enc.output for enc in self.encoders], 1)

        if self.adversarial:
            return _reverse_gradient(mlp_input)
        return mlp_input

    @tensor
    def mlp(self) -> MultilayerPerceptron:
        return MultilayerPerceptron(
            self.mlp_input, self.layers,
            self.dropout_keep_prob, len(self.vocabulary),
            activation_fn=self.activation_fn, train_mode=self.train_mode)

    @tensor
    def loss_with_gt_ins(self) -> tf.Tensor:
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.mlp.logits, labels=self.targets[0]))

    @property
    def loss_with_decoded_ins(self) -> tf.Tensor:
        return self.loss_with_gt_ins

    @property
    def cost(self) -> tf.Tensor:
        tf.summary.scalar(
            "train_optimization_cost",
            self.loss_with_gt_ins, collections=["summary_train"])

        return self.loss_with_gt_ins

    @tensor
    def decoded_seq(self) -> tf.Tensor:
        return tf.expand_dims(self.mlp.classification, 0)

    @tensor
    def decoded_logits(self) -> tf.Tensor:
        return tf.expand_dims(self.mlp.logits, 0)

    @tensor
    def runtime_logprobs(self) -> tf.Tensor:
        return tf.expand_dims(tf.nn.log_softmax(self.mlp.logits), 0)

    @property
    def train_loss(self):
        return self.loss_with_gt_ins

    @property
    def runtime_loss(self):
        return self.loss_with_decoded_ins

    @property
    def decoded(self):
        return self.decoded_seq

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = ModelPart.feed_dict(self, dataset, train)
        sentences = dataset.maybe_get_series(self.data_id)

        if sentences is not None:
            label_tensors, _ = self.vocabulary.sentences_to_tensor(
                list(sentences), self.max_output_len)
            fd[self.targets[0]] = label_tensors[0]

        return fd


def _reverse_gradient(x: tf.Tensor) -> tf.Tensor:
    """Flips the sign of the incoming gradient during training."""

    grad_name = "gradient_reversal_{}".format(x.name)

    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) ]

    from neuralmonkey.experiment import Experiment
    graph = Experiment.get_current().graph
    with graph.gradient_override_map({"Identity": grad_name}):
        y = tf.identity(x)

    return y
