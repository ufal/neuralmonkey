# pylint: disable=too-few-public-methods
class Attentive(object):
    def __init__(self, attention_type, **kwargs):
        self._attention_type = attention_type
        self._attention_kwargs = kwargs

        assert hasattr(self, "name")
        assert hasattr(self, "_padding")
        assert hasattr(self, "_attention_tensor")

    def get_attention_object(self, runtime: bool=False):
        # pylint: disable=no-member
        if self._attention_type and self._attention_tensor is None:
            raise Exception("Can't get attention: missing attention tensor.")
        if self._attention_type and self.name is None:
            raise Exception("Can't get attention: missing encoder's name.")
        if self._attention_type and self._padding is None:
            raise Exception("Can't get attention: missing input padding.")

        return self._attention_type(
            self._attention_tensor,
            scope="attention_{}".format(self.name),
            input_weights=self._padding,
            runtime_mode=runtime,
            **self._attention_kwargs) if self._attention_type else None
