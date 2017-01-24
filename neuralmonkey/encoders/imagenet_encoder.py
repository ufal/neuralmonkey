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

from neuralmonkey.dataset import Dataset
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.decoding_function import Attention
from neuralmonkey.model.model_part import ModelPart, FeedDict

SUPPORTED_NETWORKS = {
    "AlexNet": (tf_slim.nets.alexnet.alexnet_v2_arg_scope,
                tf_slim.nets.alexnet.alexnet_v2),
    "Resnet50": (tf_slim.nets.resnet_v1.resnet_arg_scope,
                 tf_slim.nets.resnet_v1.resnet_v1_50),
    "Resnet101": (tf_slim.nets.resnet_v1.resnet_arg_scope,
                  tf_slim.nets.resnet_v1.resnet_v1_101),
    "Resnet152": (tf_slim.nets.resnet_v1.resnet_arg_scope,
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
    """Pretrained ImageNet network."""

    WIDTH = 224
    HEIGHT = 224

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 data_id: str,
                 network_type: str,
                 output_layer: str,
                 attention_type: Type=Attention,
                 fine_tune: bool=False,
                 load_checkpoint: Optional[str]=None,
                 save_checkpoint: Optional[str]=None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(self, attention_type)

        self.data_id = data_id
        self._network_type = network_type
        self.input_plc = tf.placeholder(
            tf.float32, [None, self.HEIGHT, self.WIDTH, 3])

        if network_type not in SUPPORTED_NETWORKS:
            raise ValueError(
                "Network '{}' is not amonng the supoort ones ({})".format(
                    network_type, ", ".join(SUPPORTED_NETWORKS.keys())))

        scope, net_function = SUPPORTED_NETWORKS[network_type]
        with tf_slim.arg_scope(scope()):
            _, end_points = net_function(self.input_plc)

        with tf.variable_scope(self.name):
            net_output = end_points[output_layer]
            if not fine_tune:
                net_output = tf.stop_gradient(net_output)
            # pylint: disable=no-member
            shape = [s.value for s in net_output.get_shape()[1:]]
            # pylint: enable=no-member
            self.__attention_tensor = tf.reshape(
                net_output, [-1, shape[0] * shape[1], shape[2]])

            self.encoded = tf.reduce_mean(net_output, [1, 2])
    # pylint: enable=too-many-arguments

    def _init_saver(self) -> None:
        if not self._saver:
            with tf.variable_scope(self._name, reuse=True):
                local_variables = tf.get_collection(
                    tf.GraphKeys.VARIABLES, scope=self._name)
                slim_variables = tf.get_collection(
                    tf.GraphKeys.VARIABLES, scope=self._network_type)
                self._saver = tf.train.Saver(
                    var_list=local_variables + slim_variables)

    @property
    def _attention_tensor(self) -> tf.Tensor:
        return self.__attention_tensor

    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        images = np.array(dataset.get_series(self.data_id))
        assert images.shape[1:] == (self.HEIGHT, self.WIDTH, 3)

        return {self.input_plc: images}
