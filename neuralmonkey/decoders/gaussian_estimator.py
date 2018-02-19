import numbers
from typing import List, cast, Optional

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.dataset import Dataset
from neuralmonkey.decoders.autoregressive import AutoregressiveDecoder
from neuralmonkey.decoders.encoder_projection import (
    EncoderProjection, linear_encoder_projection)
from neuralmonkey.decorators import tensor
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.model.stateful import Stateful, TemporalStatefulWithOutput
from neuralmonkey.nn.utils import dropout


class GaussianEstimator(ModelPart):
    """Model part estimating a scalar via a normal a distribution."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoders: List[TemporalStatefulWithOutput],
                 encoder_signal_size: Optional[int],
                 data_id: str,
                 use_gauss_density_loss: bool = True,
                 use_mse_loss: bool = True,
                 use_stddev_value_loss: bool = True,
                 fixed_stddev: float = None,
                 add_hidden_layer: bool = False,
                 encoder_projection: EncoderProjection = None,
                 dropout_keep_prob: float = 1.0,
                 oracle_estimation: bool = False,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Initialize Gaussian scalar estimator.

        Args:
          encoders: List of encoders whose states are used for the estimation.
          encoder_projection: Function projection the encoder states into a
              single vector.
          orcale_estimation: Flag whether the observed value should be used as
              a mean instead of the trained value.
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        # check_argument_types()

        # TODO: checks at least one loss is used, fixed stddev is used during
        #       oracle estimation

        self.encoders = encoders
        self.__casted_encoders = [cast(Stateful, e) for e in self.encoders]
        self.encoder_projection = encoder_projection
        self.encoder_signal_size = encoder_signal_size
        self.data_id = data_id
        self.use_gauss_density_loss = use_gauss_density_loss
        self.use_mse_loss = use_mse_loss
        self.use_stddev_value_loss = use_stddev_value_loss
        self.fixed_stddev = fixed_stddev
        self.add_hidden_layer = add_hidden_layer
        self.dropout_keep_prob = dropout_keep_prob
        self.oracle_estimation = oracle_estimation

        if self.encoder_projection is None:
            self.encoder_projection = linear_encoder_projection(
                self.dropout_keep_prob)

        self.train_mode = tf.placeholder(tf.bool, shape=[], name="train_mode")
        self.observed_value = tf.placeholder(tf.float32, shape=[None], name="observed_value")
    # pylint: enable=too-many-arguments

    @tensor
    def encoder_signal(self) -> tf.Tensor:
        """Vector which is fed to a FF net for mean and var computation."""
        with tf.variable_scope("encoder_signal"):
            encoder_signal = dropout(
                self.encoder_projection(self.train_mode,
                                        self.encoder_signal_size,
                                        self.__casted_encoders),
                self.dropout_keep_prob,
                self.train_mode)
            if self.add_hidden_layer:
                encoder_signal = dropout(tf.layers.dense(
                    encoder_signal, self.encoder_signal_size,
                    activation=tf.nn.relu), self.dropout_keep_prob, self.train_mode)

        return encoder_signal

    @tensor
    def mean(self) -> tf.Tensor:
        if self.oracle_estimation:
            return self.observed_value

        scale_and_shift = tf.layers.dense(
            self.encoder_signal, 2, name="mean_estimation")
        mean_scale = scale_and_shift[:, 0]
        mean_shift = scale_and_shift[:, 1]
        return mean_scale * tf.to_float(self.encoders[0].lengths) + mean_shift

    @tensor
    def stddev(self) -> tf.Tensor:
        if self.fixed_stddev is not None:
            return tf.constant(self.fixed_stddev)

        stddev = tf.layers.dense(
            self.encoder_signal, 1, name="stddev_estimation",
            activation=tf.nn.elu) + 1
        return tf.squeeze(stddev, axis=1)

    @tensor
    def distribution(self):
        return tf.distributions.Normal(self.mean, self.stddev)

    # pylint: disable=no-member
    @tensor
    def gauss_density_loss(self) -> tf.Tensor:
        return -tf.reduce_sum(self.distribution.log_prob(self.observed_value))

    @tensor
    def mse_loss(self) -> tf.Tensor:
        return tf.reduce_sum((self.observed_value - self.mean) ** 2)

    @tensor
    def stddev_value_loss(self) -> tf.Tensor:
        return tf.reduce_sum(self.stddev)

    @tensor
    def cost(self) -> tf.Tensor:
        costs = []

        if self.use_gauss_density_loss:
            costs.append(self.gauss_density_loss)
        if self.use_mse_loss:
            costs.append(self.mse_loss)
        if self.use_stddev_value_loss:
            costs.append(self.stddev_value_loss)

        return sum(costs)

    def probability_around(self, value: tf.Tensor, interval: float = 0.5):
        return (self.distribution.cdf(value + interval) -
                self.distribution.cdf(value - interval))

    def probability_at_least(self, value: tf.Tensor, interval: float = 0.5):
        return 1 - self.distribution.cdf(value - interval)
    # pylint: enable=no-member

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = {}  # type: FeedDict
        fd[self.train_mode] = train

        values = dataset.get_series(self.data_id, allow_none=True)

        if values is None and train:
            raise ValueError("Series '{}' must be provided in train mode."
                             .format(self.data_id))

        if values:
            values_list = list(values) if values else None
            if isinstance(values_list[0], numbers.Real):
                targets = np.array(values_list)
            elif isinstance(values_list[0], list):
                targets = np.array([len(l) for l in values_list])
            else:
                raise ValueError(
                    "Provided data series must consist either "
                    "of numbers or lists, but was '{}''."
                    .format(type(values_list[0])))
            fd[self.observed_value] = targets

        return fd


class GreedyDecderLenEstimator(ModelPart):
    def __init__(self,
                 name: str,
                 greedy_decoder: AutoregressiveDecoder,
                 fixed_stddev: float = 1.0) -> None:
        ModelPart.__init__(self, name, None, None)

        self.greedy_decoder = greedy_decoder
        self.fixed_stddev = fixed_stddev

    @tensor
    def mean(self) -> tf.Tensor:
        decoded_len = tf.reduce_sum(tf.to_float(self.greedy_decoder.runtime_mask), 0)
        return tf.to_float(decoded_len)

    @tensor
    def stddev(self) -> tf.Tensor:
        return self.fixed_stddev

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = {}  # type: FeedDict
        return fd
