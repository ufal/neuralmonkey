"""Basic functionality of all model parts."""
from typing import Set

# pylint: disable=unused-import
# TODO feed dict and initializer specs are imported for convenience
# TODO this is because the codebase import these things from this module
from neuralmonkey.model.parameterized import Parameterized, InitializerSpecs
from neuralmonkey.model.feedable import Feedable, FeedDict
# pylint: enable=unused-import


class ModelPart(Parameterized, Feedable):
    """Base class of all model parts."""

    def __init__(self,
                 name: str,
                 reuse: "ModelPart" = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        Parameterized.__init__(self, name, reuse, save_checkpoint,
                               load_checkpoint, initializers)
        with self.use_scope():
            Feedable.__init__(self)

    def get_dependencies(self) -> Set["ModelPart"]:
        """Collect recusively all encoders and decoders."""
        to_return = set([self])

        if hasattr(self, "attentions"):
            to_return = to_return.union(
                *(enc.get_dependencies()
                  for enc in getattr(self, "attentions")
                  if isinstance(enc, ModelPart)))

        if hasattr(self, "encoders"):
            to_return = to_return.union(
                *(enc.get_dependencies()
                  for enc in getattr(self, "encoders")
                  if isinstance(enc, ModelPart)))

        if hasattr(self, "encoder"):
            enc = getattr(self, "encoder")
            if isinstance(enc, ModelPart):
                to_return = to_return.union(enc.get_dependencies())

        if hasattr(self, "input_sequence"):
            inpseq = getattr(self, "input_sequence")
            if isinstance(inpseq, ModelPart):
                to_return = to_return.union(inpseq.get_dependencies())

        if hasattr(self, "parent_decoder"):
            dec = getattr(self, "parent_decoder")
            if isinstance(dec, ModelPart):
                to_return = to_return.union(dec.get_dependencies())

        return to_return
