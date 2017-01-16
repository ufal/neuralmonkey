"""Pre-trained ImageNet networks."""

from typing import Optional, Type

import tensorflow as tf
import tensorflow.contrib.slim as tf_slim
import tensorflow.contrib.slim.nets

from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.model.model_part import ModelPart

SUPPORTED_NETWORKS = {
    "resnet_50": (tf_slim.nets.resnet_v1.resnet_arg_scope,
                  tf_slim.nets.resnet_v1.resnet_v1_50),
}

class ImageNet(ModelPart, Attentive):
    """Pretrained ImageNet network."""
    def __init__(self,
                 name: str,
                 network_type: str,
                 output_layer: str,
                 load_checkpoint: str,
                 attention_type: Type,
                 fine_tune: bool=False,
                 save_checkpoint: Optional[str]=None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(self, attention_type)

        self.input_plc = None

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
