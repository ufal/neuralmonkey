"""Pre-trained ImageNet networks."""

from typing import Optional

from typeguard import check_argument_types
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as tf_slim
# pylint: disable=unused-import
# Workaround of missing slim's import
# see https://github.com/tensorflow/tensorflow/issues/6064
import tensorflow.contrib.slim.nets
# pylint: enable=unused-import

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.model.model_part import ModelPart, FeedDict, InitializerSpecs
from neuralmonkey.model.stateful import SpatialStatefulWithOutput


SUPPORTED_NETWORKS = {
    "AlexNet": (
        tf_slim.nets.alexnet.alexnet_v2_arg_scope, (224, 224),
        lambda image: tf_slim.nets.alexnet.alexnet_v2(image)),
    "VGG16": (
        tf_slim.nets.vgg.vgg_arg_scope, (224, 224),
        lambda image: tf_slim.nets.vgg.vgg_16(
            image, is_training=False, dropout_keep_prob=1.0)),
    "VGG19": (
        tf_slim.nets.vgg.vgg_arg_scope, (224, 224),
        lambda image: tf_slim.nets.vgg.vgg_19(
            image, is_training=False, dropout_keep_prob=1.0)),
    "ResNet_v2_50": (
        tf_slim.nets.resnet_v2.resnet_arg_scope, (229, 229),
        lambda image: tf_slim.nets.resnet_v2.resnet_v2_50(
            image, is_training=False, global_pool=False)),
    "ResNet_v2_101": (
        tf_slim.nets.resnet_v2.resnet_arg_scope, (229, 229),
        lambda image: tf_slim.nets.resnet_v2.resnet_v2_101(
            image, is_training=False, global_pool=False)),
    "ResNet_v2_152": (
        tf_slim.nets.resnet_v2.resnet_arg_scope, (229, 229),
        lambda image: tf_slim.nets.resnet_v2.resnet_v2_152(
            image, is_training=False, global_pool=False)),
}


class ImageNet(ModelPart, SpatialStatefulWithOutput):
    """Pre-trained ImageNet network."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 data_id: str,
                 network_type: str,
                 load_checkpoint: str,
                 spacial_layer: str = None,
                 encoded_layer: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Initialize pre-trained ImageNet network.

        Args:
            name: Name of the model part (the ImageNet network, will be in its
                scope, independently on `name`).
            data_id: Id of series with images (list of 3D numpy arrays)
            network_type: Identifier of ImageNet network from TFSlim.
            spacial_layer: String identifier of the convolutional map
                (model's endpoint). Check
                TFSlim documentation for end point specifications.
            encoded_layer: String id of the network layer that will be used as
                input of a decoder. `None` means averaging the convolutional
                maps.
            load_checkpoint: Checkpoint file from which the pre-trained network
                is loaded.
        """
        check_argument_types()

        ModelPart.__init__(self, name, load_checkpoint, initializers,
                           save_checkpoint=None)

        self.data_id = data_id
        self.network_type = network_type
        self.spacial_layer = spacial_layer
        self.encoded_layer = encoded_layer

        if self.network_type not in SUPPORTED_NETWORKS:
            raise ValueError(
                "Network '{}' is not among the supported ones ({})".format(
                    self.network_type, ", ".join(SUPPORTED_NETWORKS.keys())))

        (scope, (self.height, self.width),
         net_function) = SUPPORTED_NETWORKS[self.network_type]
        with tf_slim.arg_scope(scope()):
            _, self.end_points = net_function(self.input_image)

        if (self.spacial_layer is not None and
                self.spacial_layer not in self.end_points):
            raise ValueError(
                "Network '{}' does not contain endpoint '{}'".format(
                    self.network_type, self.spacial_layer))

        if spacial_layer is not None:
            net_output = self.end_points[self.spacial_layer]
            if len(net_output.get_shape()) != 4:
                raise ValueError(
                    ("Endpoint '{}' for network '{}' cannot be "
                     "a convolutional map, its dimensionality is: {}."
                    ).format(self.spacial_layer, self.network_type,
                             ", ".join([str(d.value) for d in
                                        net_output.get_shape()])))

        if (self.encoded_layer is not None
                and self.encoded_layer not in self.end_points):
            raise ValueError(
                "Network '{}' does not contain endpoint '{}'.".format(
                    self.network_type, self.encoded_layer))

    @tensor
    def input_image(self) -> tf.Tensor:
        return tf.placeholder(
            tf.float32, [None, self.height, self.width, 3])

    @tensor
    def spatial_states(self) -> Optional[tf.Tensor]:
        if self.spacial_layer is None:
            return None

        net_output = self.end_points[self.spacial_layer]
        net_output = tf.stop_gradient(net_output)
        return net_output

    @tensor
    def spatial_mask(self) -> tf.Tensor:
        if self.spacial_layer is None:
            return None
        mask = tf.ones(tf.shape(self.spatial_states)[:3])
        # pylint: disable=no-member
        mask.set_shape(self.spatial_states.get_shape()[:3])
        # pylint: enable=no-member
        return mask

    @tensor
    def output(self) -> tf.Tensor:
        if self.encoded_layer is None:
            return tf.reduce_mean(self.spatial_states, [1, 2])

        encoded = tf.squeeze(self.end_points[self.encoded_layer], [1, 2])
        encoded = tf.stop_gradient(encoded)
        return encoded

    def _init_saver(self) -> None:
        if not self._saver:
            with tf.variable_scope(self.name, reuse=True):
                local_variables = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
                slim_variables = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self.network_type)
                self._saver = tf.train.Saver(
                    var_list=local_variables + slim_variables)

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        images = np.array(dataset.get_series(self.data_id))
        assert images.shape[1:] == (self.height, self.width, 3)

        return {self.input_image: images}
