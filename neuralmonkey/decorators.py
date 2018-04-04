from functools import wraps

import tensorflow as tf

from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.tf_utils import tf_print


def tensor(func):
    @wraps(func)
    def decorate(self, *args, **kwargs):
        attribute_name = "_{}_cached_placeholder".format(func.__name__)
        if not hasattr(self, attribute_name):
            if isinstance(self, ModelPart):
                # jump out of the caller's scope and into the ModelPart's scope
                with self.use_scope():
                    value = func(self, *args, **kwargs)
                    if isinstance(value, tf.Tensor):
                        value = tf_print(
                            value, "<{}.{}>".format(self.name, func.__name__),
                            "tensorval")
            else:
                value = func(self, *args, **kwargs)
            setattr(self, attribute_name, value)

        return getattr(self, attribute_name)
    return property(decorate)
