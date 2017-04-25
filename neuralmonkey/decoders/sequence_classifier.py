from typing import cast, Any, Callable, Iterable, Optional, List

import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.nn.mlp import MultilayerPerceptron
from neuralmonkey.nn.projection import dropout


# pylint: disable=too-many-instance-attributes


class SequenceClassifier(ModelPart):
    """A simple MLP classifier over encoders.

    The API pretends it is an RNN decoder which always generates a sequence of
    length exactly one.
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoders: List[Any],
                 vocabulary: Vocabulary,
                 data_id: str,
                 layers: List[int],
                 activation_fn: Callable[[tf.Tensor], tf.Tensor]=tf.nn.relu,
                 dropout_keep_prob: float = 0.5,
                 dropout_input: float = 1.0,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None) -> None:
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
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        self.encoders = encoders
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.layers = layers
        self.activation_fn = activation_fn
        self.dropout_keep_prob = dropout_keep_prob
        self.dropout_input = dropout_input
        self.max_output_len = 1

        with tf.variable_scope(name):
            self.train_mode = tf.placeholder(tf.bool, name="train_mode")
            self.learning_step = tf.get_variable(
                "learning_step", [], trainable=False,
                initializer=tf.constant_initializer(0))

            self.gt_inputs = [tf.placeholder(
                tf.int32, shape=[None], name="targets")]
            mlp_input = tf.concat([enc.encoded for enc in encoders], 1)

            mlp_input = dropout(mlp_input, self.dropout_input, self.train_mode)

            mlp = MultilayerPerceptron(
                mlp_input, layers, self.dropout_keep_prob, len(vocabulary),
                activation_fn=self.activation_fn, train_mode=self.train_mode)

            self.loss_with_gt_ins = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=mlp.logits, labels=self.gt_inputs[0]))
            self.loss_with_decoded_ins = self.loss_with_gt_ins
            self.cost = self.loss_with_gt_ins

            self.decoded_seq = [mlp.classification]
            self.decoded_logits = [mlp.logits]
            self.runtime_logprobs = [tf.nn.log_softmax(mlp.logits)]

            tf.summary.scalar(
                'val_optimization_cost', self.cost,
                collections=["summary_val"])
            tf.summary.scalar(
                'train_optimization_cost',
                self.cost, collections=["summary_train"])
    # pylint: enable=too-many-arguments

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
        sentences = cast(Iterable[List[str]],
                         dataset.get_series(self.data_id, allow_none=True))

        sentences_list = list(sentences) if sentences is not None else None

        fd = {}  # type: FeedDict


        if sentences is not None:
            label_tensors, _ = self.vocabulary.sentences_to_tensor(
                sentences_list, self.max_output_len)
            fd[self.gt_inputs[0]] = label_tensors[0]

        fd[self.train_mode] = train

        return fd
