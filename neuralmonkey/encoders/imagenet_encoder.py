"""Pre-trained ImageNet networks."""

from typing import Optional, Type

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as tf_slim
# pylint: disable=unused-import
# Workaround of missing slim's import
# see https://github.com/tensorflow/tensorflow/issues/6064
import tensorflow.contrib.slim.nets
# pylint: enable=unused-import

from neuralmonkey.logging import warn
from neuralmonkey.dataset import Dataset
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.decoding_function import Attention
from neuralmonkey.model.model_part import ModelPart, FeedDict

SUPPORTED_NETWORKS = {
    "AlexNet": (tf_slim.nets.alexnet.alexnet_v2_arg_scope,
                tf_slim.nets.alexnet.alexnet_v2),
    "resnet_v1_50": (tf_slim.nets.resnet_v1.resnet_arg_scope,
                     tf_slim.nets.resnet_v1.resnet_v1_50),
    "resnet_v1_101": (tf_slim.nets.resnet_v1.resnet_arg_scope,
                      tf_slim.nets.resnet_v1.resnet_v1_101),
    "resnet_v1_152": (tf_slim.nets.resnet_v1.resnet_arg_scope,
                      tf_slim.nets.resnet_v1.resnet_v1_152),
    "InceptionV1": (tf_slim.nets.inception.inception_v1_arg_scope,
                    tf_slim.nets.inception.inception_v1),
    "InceptionV2": (tf_slim.nets.inception.inception_v2_arg_scope,
                    tf_slim.nets.inception.inception_v2),
    "InceptionV3": (tf_slim.nets.inception.inception_v3_arg_scope,
                    tf_slim.nets.inception.inception_v3),
    # "inception_v4": (tf_slim.nets.inception.inception_v4_arg_scope,
    #                  tf_slim.nets.inception.inception_v4),
    "vgg_16": (tf_slim.nets.vgg.vgg_arg_scope,
               tf_slim.nets.vgg.vgg_16),
    "vgg_19": (tf_slim.nets.vgg.vgg_arg_scope,
               tf_slim.nets.vgg.vgg_19),
}


class ImageNet(ModelPart, Attentive):
    """Pre-trained ImageNet network."""

    WIDTH = 224
    HEIGHT = 224

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 data_id: str,
                 network_type: str,
                 attention_layer: Optional[str] = None,
                 attention_state_size: Optional[int] = None,
                 attention_type: Type = Attention,
                 fine_tune: bool = False,
                 encoded_layer: Optional[str] = None,
                 load_checkpoint: Optional[str] = None,
                 save_checkpoint: Optional[str] = None) -> None:
        """Initialize pre-trained ImageNet network.

        Args:
            name: Name of the model part (the ImageNet network, will be in its
                scope, independently on `name`).
            data_id: Id of series with images (list of 3D numpy arrays)
            network_type: Identifier of ImageNet network from TFSlim.
            attention_layer: String identifier of the convolutional map
                (model's endpoint) that will be used for attention. Check
                TFSlim documentation for end point specifications.
            attention_state_size: Dimensionality of state projection in
                attention computation.
            attention_type: Type of attention.
            fine_tune: Flag whether the network should be further trained with
                the rest of the model.
            encoded_layer: String id of the network layer that will be used as
                input of a decoder. `None` means averaging the convolutional
                maps.
            load_checkpoint: Checkpoint file from which the pre-trained network
                is loaded.
            save_checkpoint: Checkpoint file where the encoder is saved after
                the training. (Makes sense only if `fine_tune` is set to
                `True`).
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(self, attention_type,
                           attention_state_size=attention_state_size)

        if attention_layer is None and attention_type is not None:
            raise ValueError("Attention type is set, although "
                             "attention layer is not specified.")

        if save_checkpoint is not None and not fine_tune:
            warn("The ImageNet network is not fine-tuned and still it is set "
                 "to save after the training is finished.")

        self.data_id = data_id
        self._network_type = network_type
        self.input_plc = tf.placeholder(
            tf.float32, [None, self.HEIGHT, self.WIDTH, 3])

        if network_type not in SUPPORTED_NETWORKS:
            raise ValueError(
                "Network '{}' is not among the supoort ones ({})".format(
                    network_type, ", ".join(SUPPORTED_NETWORKS.keys())))

        scope, net_function = SUPPORTED_NETWORKS[network_type]
        with tf_slim.arg_scope(scope()):
            _, end_points = net_function(self.input_plc)

        with tf.variable_scope(self.name):
            if attention_layer is not None:

                if attention_layer not in end_points:
                    raise ValueError(
                        "Network '{}' does not contain endpoint '{}'".format(
                            network_type, attention_layer))

                net_output = end_points[attention_layer]

                if len(net_output.get_shape()) != 4:
                    raise ValueError(
                        ("Endpoint '{}' for network '{}' cannot be "
                         "a convolutional map, its dimensionality is: {}."
                        ).format(attention_layer, network_type,
                                 ", ".join([str(d.value) for d in
                                            net_output.get_shape()])))

                if not fine_tune:
                    net_output = tf.stop_gradient(net_output)
                # pylint: disable=no-member
                shape = [s.value for s in net_output.get_shape()[1:]]
                # pylint: enable=no-member
                self.__attention_tensor = tf.reshape(
                    net_output, [-1, shape[0] * shape[1], shape[2]])

            if encoded_layer is not None:
                if encoded_layer not in end_points:
                    raise ValueError(
                        "Network '{}' does not contain endpoint '{}'.".format(
                            network_type, encoded_layer))

                self.encoded = tf.squeeze(end_points[encoded_layer], [1, 2])
                if not fine_tune:
                    self.encoded = tf.stop_gradient(self.encoded)
            else:
                self.encoded = tf.reduce_mean(net_output, [1, 2])
    # pylint: enable=too-many-arguments,too-many-locals

    def _init_saver(self) -> None:
        if not self._saver:
            with tf.variable_scope(self._name, reuse=True):
                local_variables = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
                slim_variables = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self._network_type)
                self._saver = tf.train.Saver(
                    var_list=local_variables + slim_variables)

    @property
    def _attention_tensor(self) -> tf.Tensor:
        return self.__attention_tensor

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        images = np.array(dataset.get_series(self.data_id))
        assert images.shape[1:] == (self.HEIGHT, self.WIDTH, 3)

        return {self.input_plc: images}
