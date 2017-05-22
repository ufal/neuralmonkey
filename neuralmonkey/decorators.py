from functools import wraps
import tensorflow as tf

from neuralmonkey.model.model_part import ModelPart


def tensor(func):
    @wraps(func)
    def decorate(self, *args, **kwargs):
        attribute_name = "_{}_cached_placeholder".format(func.__name__)
        if not hasattr(self, attribute_name):
            if isinstance(self, ModelPart):
                # jump out of the caller's scope and into the ModelPart's scope
                with self.use_scope():
                    value = func(self, *args, **kwargs)
            else:
                value = func(self, *args, **kwargs)
            assert isinstance(value, tf.Tensor)
            setattr(self, attribute_name, value)

        return getattr(self, attribute_name)
    return property(decorate)

def variable(func):
    @wraps(func)
    def decorate(self, *args, **kwargs):
        attribute_name = "_{}_cached_placeholder".format(func.__name__)
        if not hasattr(self, attribute_name):
            if isinstance(self, ModelPart):
                # jump out of the caller's scope and into the ModelPart's scope
                with self.use_scope():
                    value = func(self, *args, **kwargs)
            else:
                value = func(self, *args, **kwargs)
            assert isinstance(value, tf.Variable)
            setattr(self, attribute_name, value)

        return getattr(self, attribute_name)
    return property(decorate)


def tensortuple(func):
    @wraps(func)
    def decorate(self, *args, **kwargs):
        attribute_name = "_{}_cached_placeholder".format(func.__name__)
        if not hasattr(self, attribute_name):
            if isinstance(self, ModelPart):
                # jump out of the caller's scope and into the ModelPart's scope
                with self.use_scope():
                    value = func(self, *args, **kwargs)
            else:
                value = func(self, *args, **kwargs)

            def assert_tensortuple(val):
                if isinstance(val, tuple):
                    for child in val:
                        assert_tensortuple(child)
                else:
                    assert isinstance(val, tf.Tensor)

            assert isinstance(value, tuple)
            assert_tensortuple(value)
            setattr(self, attribute_name, value)
        return getattr(self, attribute_name)

    return property(decorate)
