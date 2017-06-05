from typing import cast, Any, Callable, Iterable, Optional, List

import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.nn.mlp import MultilayerPerceptron
from neuralmonkey.decorators import tensor


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
        self.max_output_len = 1

        tf.summary.scalar(
            'train_optimization_cost',
            self.cost, collections=["summary_train"])
# pylint: enable=too-many-arguments

    # pylint: disable=no-self-use
    @tensor
    def train_mode(self) -> tf.Tensor:
        return tf.placeholder(tf.bool, name="train_mode")

    @tensor
    def gt_inputs(self) -> List[tf.Tensor]:
        return [tf.placeholder(tf.int32, shape=[None], name="targets")]
    # pylint: enable=no-self-use

    @tensor
    def _mlp(self) -> MultilayerPerceptron:
        mlp_input = tf.concat([enc.encoded for enc in self.encoders], 1)
        return MultilayerPerceptron(
            mlp_input, self.layers,
            self.dropout_keep_prob, len(self.vocabulary),
            activation_fn=self.activation_fn, train_mode=self.train_mode)

    @tensor
    def loss_with_gt_ins(self) -> tf.Tensor:
        # pylint: disable=no-member,unsubscriptable-object
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self._mlp.logits, labels=self.gt_inputs[0]))
        # pylint: enable=no-member,unsubscriptable-object

    @property
    def loss_with_decoded_ins(self) -> tf.Tensor:
        return self.loss_with_gt_ins

    @property
    def cost(self) -> tf.Tensor:
        return self.loss_with_gt_ins

    @tensor
    def decoded_seq(self) -> List[tf.Tensor]:
        # pylint: disable=no-member
        return [self._mlp.classification]
        # pylint: enable=no-member

    @tensor
    def decoded_logits(self) -> List[tf.Tensor]:
        # pylint: disable=no-member
        return [self._mlp.logits]
        # pylint: enable=no-member

    @tensor
    def runtime_logprobs(self) -> List[tf.Tensor]:
        # pylint: disable=no-member
        return [tf.nn.log_softmax(self._mlp.logits)]
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

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        sentences = cast(Iterable[List[str]],
                         dataset.get_series(self.data_id, allow_none=True))

        sentences_list = list(sentences) if sentences is not None else None

        fd = {}  # type: FeedDict

        if sentences is not None:
            label_tensors, _ = self.vocabulary.sentences_to_tensor(
                sentences_list, self.max_output_len)

            # pylint: disable=unsubscriptable-object
            fd[self.gt_inputs[0]] = label_tensors[0]
            # pylint: enable=unsubscriptable-object

        fd[self.train_mode] = train

        return fd
