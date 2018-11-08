"""Basic functionality of all model parts."""
from abc import ABCMeta
from typing import Set

# pylint: disable=unused-import
# TODO feed dict and initializer specs are imported for convenience
# TODO this is because the codebase import these things from this module
from neuralmonkey.model.parameterized import Parameterized, InitializerSpecs
from neuralmonkey.model.feedable import Feedable, FeedDict
# pylint: enable=unused-import


# pylint: disable=too-few-public-methods
# TODO add some public methods or think of something else
class GenericModelPart(metaclass=ABCMeta):

    def get_dependencies(self) -> Set["GenericModelPart"]:
        """Collect recusively all encoders and decoders."""
        to_return = set([self])

        if hasattr(self, "attentions"):
            to_return = to_return.union(
                *(enc.get_dependencies()
                  for enc in getattr(self, "attentions")
                  if isinstance(enc, GenericModelPart)))

        if hasattr(self, "encoders"):
            to_return = to_return.union(
                *(enc.get_dependencies()
                  for enc in getattr(self, "encoders")
                  if isinstance(enc, GenericModelPart)))

        if hasattr(self, "encoder"):
            enc = getattr(self, "encoder")
            if isinstance(enc, GenericModelPart):
                to_return = to_return.union(enc.get_dependencies())

        if hasattr(self, "input_sequence"):
            inpseq = getattr(self, "input_sequence")
            if isinstance(inpseq, GenericModelPart):
                to_return = to_return.union(inpseq.get_dependencies())

        if hasattr(self, "parent_decoder"):
            dec = getattr(self, "parent_decoder")
            if isinstance(dec, GenericModelPart):
                to_return = to_return.union(dec.get_dependencies())

        return to_return
# pylint: enable=too-few-public-methods


class ModelPart(Parameterized, GenericModelPart, Feedable):
    """Base class of all parametric feedable model parts."""

    def __init__(self,
                 name: str,
                 reuse: "ModelPart" = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        Parameterized.__init__(self, name, reuse, save_checkpoint,
                               load_checkpoint, initializers)
        GenericModelPart.__init__(self)
        with self.use_scope():
            Feedable.__init__(self)
