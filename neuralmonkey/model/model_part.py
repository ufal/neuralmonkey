"""Basic functionality of all model parts."""
from abc import ABCMeta
from typing import MutableSet, Set, List, Tuple

# pylint: disable=unused-import
# TODO feed dict and initializer specs are imported for convenience
# TODO this is because the codebase import these things from this module
from neuralmonkey.model.parameterized import Parameterized, InitializerSpecs
from neuralmonkey.model.feedable import Feedable, FeedDict
# pylint: enable=unused-import


# pylint: disable=too-few-public-methods
# TODO add some public methods or think of something else
class GenericModelPart(metaclass=ABCMeta):

    @property
    def _list_dependencies(self) -> List[str]:
        return ["attentions", "encoders"]

    @property
    def _singleton_dependencies(self) -> List[str]:
        return ["encoder", "parent_decoder", "input_sequence"]

    def __get_deps_from_list(
            self,
            attr: str,
            feedables: MutableSet[Feedable],
            parameterizeds: MutableSet[Parameterized]) -> None:

        if hasattr(self, attr):
            for enc in getattr(self, attr):
                if isinstance(enc, GenericModelPart):
                    feeds, params = enc.get_dependencies()
                    feedables |= feeds
                    parameterizeds |= params

    def __get_deps(
            self,
            attr: str,
            feedables: MutableSet[Feedable],
            parameterizeds: MutableSet[Parameterized]) -> None:

        if hasattr(self, attr):
            enc = getattr(self, attr)
            if isinstance(enc, GenericModelPart):
                feeds, params = enc.get_dependencies()
                feedables |= feeds
                parameterizeds |= params

    def get_dependencies(self) -> Tuple[Set[Feedable], Set[Parameterized]]:
        """Collect recusively all encoders and decoders."""
        feedables = set()  # type: Set[Feedable]
        parameterizeds = set()  # type: Set[Parameterized]

        if isinstance(self, Feedable):
            feedables |= {self}
        if isinstance(self, Parameterized):
            parameterizeds |= {self}

        for attr in self._list_dependencies:
            self.__get_deps_from_list(attr, feedables, parameterizeds)

        for attr in self._singleton_dependencies:
            self.__get_deps(attr, feedables, parameterizeds)

        return feedables, parameterizeds
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
