import tensorflow as tf

# pylint: disable=too-few-public-methods
class Attentive(object):
    def __init__(self, attention_type, **kwargs):
        self._attention_type = attention_type
        self._attention_kwargs = kwargs

    def get_attention_object(self, runtime: bool=False):
        # pylint: disable=no-member
        if hasattr(self, "name") and self.name:
            name = self.name
        else:
            name = str(self)

        return self._attention_type(
            self._attention_tensor,
            scope="attention_{}".format(name),
            input_weights=self._attention_mask,
            runtime_mode=runtime,
            **self._attention_kwargs) if self._attention_type else None

    @property
    def _attention_tensor(self):
        """Tensor over which the attention is done."""
        raise NotImplementedError(
            "Attentive object is missing attention_tensor.")

    @property
    def _attention_mask(self):
        return tf.ones(tf.shape(self._attention_tensor))
