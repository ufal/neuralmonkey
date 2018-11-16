"""Basic functionality of all model parts."""
from abc import ABCMeta
from typing import MutableSet, Set, List, Tuple

# pylint: disable=unused-import
# TODO feed dict and initializer specs are imported for convenience
# TODO this is because the codebase import these things from this module
from neuralmonkey.model.parameterized import Parameterized, InitializerSpecs
from neuralmonkey.model.feedable import Feedable, FeedDict
# pylint: enable=unused-import


class GenericModelPart(metaclass=ABCMeta):
    """Base class for Neural Monkey model parts.

    Neural Monkey dynamically decides which model parts are in use when using a
    specific trainer or a runner. Each trainer/runner holds a reference to a
    top-level model part, which is then responsible for collecting references
    to all `Parameterized` and `Feedable` objects that contribute to the
    computation of its Tensors. This behavior is implemented using the
    `get_dependencies` method, which is called recursively on all instances of
    `GenericModelPart` class that are references from within a model part.

    Apart from the `get_dependencies` method, this class also provides two
    properties: `list_dependencies` and `singleton_dependencies` which store
    the names of the Python class attributes that are regarded as potential
    dependents of the `GenericModelPart` object. These dependents are
    automatically checked for type and when they are instances of the
    `GenericModelPart` class, results of their `get_dependencies` are united
    and returned as dependencies of the current object.
    """

    @property
    def list_dependencies(self) -> List[str]:
        """Return a list of attributes regarded as lists of dependents."""
        return ["attentions", "encoders"]

    @property
    def singleton_dependencies(self) -> List[str]:
        """Return a list of attributes regarded as dependents."""
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
        """Collect all dependents of this object recursively.

        The dependents are collected using the `list_dependencies` and
        `singleton_dependencies` properties. Each stores a potential dependent
        object. If the object exsits and is an instance of `GenericModelPart`,
        dependents are collected recursively by calling its `get_dependencies`
        method.

        If the object itself is instance of `Feedable` or `Parameterized`
        class, it is added among the respective sets returned.

        Returns:
            A `Tuple` of `Set`s of `Feedable` and `Parameterized` objects.
        """
        feedables = set()  # type: Set[Feedable]
        parameterizeds = set()  # type: Set[Parameterized]

        if isinstance(self, Feedable):
            feedables |= {self}
        if isinstance(self, Parameterized):
            parameterizeds |= {self}

        for attr in self.list_dependencies:
            self.__get_deps_from_list(attr, feedables, parameterizeds)

        for attr in self.singleton_dependencies:
            self.__get_deps(attr, feedables, parameterizeds)

        return feedables, parameterizeds


class ModelPart(Parameterized, GenericModelPart, Feedable):
    """Base class of all parametric feedable model parts.

    Serves as a syntactic sugar for labeling `Feedable`, `Parameterized`, and
    `GenericModelPart` objects.
    """

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
