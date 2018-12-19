"""Pre-trained ImageNet networks."""

import sys
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import tensorflow as tf
import tensorflow.contrib.slim as tf_slim
# pylint: disable=unused-import
# Workaround of missing slim's import
# see https://github.com/tensorflow/tensorflow/issues/6064
import tensorflow.contrib.slim.nets
# pylint: enable=unused-import
from typeguard import check_argument_types

from neuralmonkey.decorators import tensor
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.stateful import SpatialStatefulWithOutput


class ImageNetSpec(NamedTuple(
        "ImageNetSpec",
        [("scope", Callable),
         ("image_size", Tuple[int, int]),
         ("apply_net", Callable)])):
    """Specification of the Imagenet encoder.

    Do not use this object directly, instead, use one of the ``get_*``functions
    in this module.

    Attributes:
        scope: The variable scope of the network to use.
        image_size: A tuple of two integers giving the image width and height
            in pixels.
        apply_net: The function that receives an image and applies the network.
    """


# pylint: disable=import-error
def get_alexnet() -> ImageNetSpec:
    import nets.alexnet_v2
    return ImageNetSpec(
        scope=nets.alexnet.alexnet_v2_arg_scope,
        image_size=(224, 224),
        apply_net=lambda image: nets.alexnet.alexnet_v2(
            image, is_training=False))


def get_vgg_by_type(vgg_type: str) -> Callable[[], ImageNetSpec]:
    def get_vgg() -> ImageNetSpec:
        import nets.vgg
        if vgg_type == "vgg16":
            net_fn = nets.vgg.vgg_16
        elif vgg_type == "vgg19":
            net_fn = nets.vgg.vgg_19
        else:
            raise ValueError(
                "Unknown type of VGG net: {}".format(vgg_type))

        return ImageNetSpec(
            scope=nets.vgg.vgg_arg_scope,
            image_size=(224, 224),
            apply_net=lambda image: net_fn(
                image, is_training=False, dropout_keep_prob=1.0))
    return get_vgg


def get_resnet_by_type(resnet_type: str) -> Callable[[], ImageNetSpec]:
    def get_resnet() -> ImageNetSpec:
        import nets.resnet_v2
        if resnet_type == "resnet_50":
            net_fn = nets.resnet_v2.resnet_v2_50
        elif resnet_type == "resnet_101":
            net_fn = nets.resnet_v2.resnet_v2_101
        elif resnet_type == "resnet_152":
            net_fn = nets.resnet_v2.resnet_v2_152
        else:
            raise ValueError(
                "Unknown type of ResNet: {}".format(resnet_type))

        return ImageNetSpec(
            scope=nets.resnet_v2.resnet_arg_scope,
            image_size=(229, 229),
            apply_net=lambda image: net_fn(
                image, is_training=False, global_pool=False))
    return get_resnet
# pylint: enable=import-error


SUPPORTED_NETWORKS = {
    "alexnet_v2": get_alexnet,
    "vgg_16": get_vgg_by_type("vgg16"),
    "vgg_19": get_vgg_by_type("vgg19"),
    "resnet_v2_50": get_resnet_by_type("resnet_50"),
    "resnet_v2_101": get_resnet_by_type("resnet_101"),
    "resnet_v2_152": get_resnet_by_type("resnet_152")
}


class ImageNet(ModelPart, SpatialStatefulWithOutput):
    """Pre-trained ImageNet network.

    We use the ImageNet networks as they are in the tesnorflow/models
    repository (https://github.com/tensorflow/models). In order use them, you
    need to clone the repository and configure the ImageNet object such that it
    has a full path to "research/slim" in the repository.  Visit
    https://github.com/tensorflow/models/tree/master/research/slim for
    information about checkpoints of the pre-trained models.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 data_id: str,
                 network_type: str,
                 slim_models_path: str,
                 load_checkpoint: str = None,
                 spatial_layer: str = None,
                 encoded_layer: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Initialize pre-trained ImageNet network.

        Args:
            name: Name of the model part (the ImageNet network, will be in its
                scope, independently on `name`).
            data_id: Id of series with images (list of 3D numpy arrays)
            network_type: Identifier of ImageNet network from TFSlim.
            spatial_layer: String identifier of the convolutional map
                (model's endpoint). Check
                TFSlim documentation for end point specifications.
            encoded_layer: String id of the network layer that will be used as
                input of a decoder. `None` means averaging the convolutional
                maps.
            path_to_models: Path to Slim models in tensorflow/models
                repository.
            load_checkpoint: Checkpoint file from which the pre-trained network
                is loaded.
        """
        check_argument_types()

        ModelPart.__init__(self, name, load_checkpoint=load_checkpoint,
                           initializers=initializers, save_checkpoint=None)
        sys.path.insert(0, slim_models_path)

        self.data_id = data_id
        self.network_type = network_type
        self.spatial_layer = spatial_layer
        self.encoded_layer = encoded_layer

        if self.network_type not in SUPPORTED_NETWORKS:
            raise ValueError(
                "Network '{}' is not among the supported ones ({})".format(
                    self.network_type, ", ".join(SUPPORTED_NETWORKS.keys())))

        self.net_specification = SUPPORTED_NETWORKS[self.network_type]()
        self.width, self.height = self.net_specification.image_size

    @property
    def input_types(self) -> Dict[str, tf.DType]:
        return {self.data_id: tf.float32}

    @property
    def input_shapes(self) -> Dict[str, tf.TensorShape]:
        return {
            self.data_id: tf.TensorShape([None, self.width, self.height, 3])}

    @tensor
    def input_image(self) -> tf.Tensor:
        return self.dataset[self.data_id]

    @tensor
    def end_points(self) -> Any:
        with tf_slim.arg_scope(self.net_specification.scope()):
            _, end_points = self.net_specification.apply_net(self.input_image)

        if (self.spatial_layer is not None
                and self.spatial_layer not in end_points):
            raise ValueError(
                "Network '{}' does not contain endpoint '{}'".format(
                    self.network_type, self.spatial_layer))

        if self.spatial_layer is not None:
            net_output = end_points[self.spatial_layer]
            if len(net_output.get_shape()) != 4:
                raise ValueError(
                    "Endpoint '{}' for network '{}' cannot be a convolutional "
                    " map, its dimensionality is: {}.".format(
                        self.spatial_layer, self.network_type,
                        ", ".join(
                            [str(d.value) for d in net_output.get_shape()])))

        if (self.encoded_layer is not None
                and self.encoded_layer not in end_points):
            raise ValueError(
                "Network '{}' does not contain endpoint '{}'.".format(
                    self.network_type, self.encoded_layer))

        return end_points

    @tensor
    def spatial_states(self) -> Optional[tf.Tensor]:
        if self.spatial_layer is None:
            return None

        # pylint: disable=unsubscriptable-object
        net_output = self.end_points[self.spatial_layer]
        # pylint: enable=unsubscriptable-object

        net_output = tf.stop_gradient(net_output)
        return net_output

    @tensor
    def spatial_mask(self) -> tf.Tensor:
        if self.spatial_layer is None:
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

        # pylint: disable=unsubscriptable-object
        encoded = tf.squeeze(self.end_points[self.encoded_layer], [1, 2])
        # pylint: enable=unsubscriptable-object

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
