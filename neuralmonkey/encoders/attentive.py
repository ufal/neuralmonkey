from abc import ABCMeta, abstractproperty
import tensorflow as tf

from neuralmonkey.decoding_function import Attention


# pylint: disable=too-few-public-methods
class Attentive(metaclass=ABCMeta):
    """A base class fro an attentive part of graph (typically encoder).

    Objects inheriting this class are able to generate an attention object that
    allows a decoder to perform attention over an attention_object provided by
    the encoder (e.g., input word representations in case of MT or
    convolutional maps in case of image captioning).
    """

    def __init__(self, attention_type, **kwargs):
        self._attention_type = attention_type
        self._attention_kwargs = kwargs

        if attention_type is not None and not issubclass(attention_type,
                                                         Attention):
            raise ValueError("Attention type is not subclass of the "
                             "Attention class")

    def create_attention_object(self):
        """Attention object that can be used in decoder."""
        # pylint: disable=no-member
        if hasattr(self, "name") and self.name:  # type: ignore
            name = self.name  # type: ignore
        else:
            name = str(self)

        return self._attention_type(
            self._attention_tensor,
            scope="attention_{}".format(name),
            input_weights=self._attention_mask,
            **self._attention_kwargs) if self._attention_type else None

    @abstractproperty
    def _attention_tensor(self):
        """Tensor over which the attention is done."""
        raise NotImplementedError(
            "Attentive object is missing attention_tensor.")

    @property
    def _attention_mask(self):
        """Zero/one masking the attention logits."""
        return tf.ones(tf.shape(self._attention_tensor)[:-1])
