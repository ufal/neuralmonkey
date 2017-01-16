"""Pre-trained ImageNet networks."""

from typing import Optional, Type

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as tf_slim

from neuralmonkey.dataset import Dataset
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.model.model_part import ModelPart, FeedDict

SUPPORTED_NETWORKS = {
    "alexnet": (tf_slim.nets.alexnet.alexnet_v2_arg_scope,
                tf_slim.nets.alexnet.alexnet_v2),
    "resnet_50": (tf_slim.nets.resnet_v1.resnet_arg_scope,
                  tf_slim.nets.resnet_v1.resnet_v1_50),
    "resnet_101": (tf_slim.nets.resnet_v1.resnet_arg_scope,
                   tf_slim.nets.resnet_v1.resnet_v1_101),
    "resnet_152": (tf_slim.nets.resnet_v1.resnet_arg_scope,
                   tf_slim.nets.resnet_v1.resnet_v1_152),
    "inception_v1": (tf_slim.nets.inception.inception_v1_arg_scope,
                     tf_slim.nets.inception.inception_v1),
    "inception_v2": (tf_slim.nets.inception.inception_v2_arg_scope,
                     tf_slim.nets.inception.inception_v2),
    "inception_v3": (tf_slim.nets.inception.inception_v3_arg_scope,
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
                 load_checkpoint: str,
                 attention_type: Type,
                 fine_tune: bool=False,
                 save_checkpoint: Optional[str]=None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(self, attention_type)

        self.data_id = data_id
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
            self.net_output = end_points[output_layer]
            if not fine_tune:
                self.net_output = tf.stop_gradient(self.net_output)
    # pylint: enable=too-many-arguments

    @property
    def _attention_tensor(self) -> tf.Tensor:
        return self.net_output

    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        images = np.array(dataset.get_series(self.data_id))
        assert images.shape[1:] == (self.HEIGHT, self.WIDTH, 3)

        return {self.input_plc: images}
