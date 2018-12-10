from abc import abstractmethod, abstractproperty
from typing import (Dict, Tuple, List, NamedTuple, Union, Set, TypeVar,
                    Generic, Optional)
import numpy as np
import tensorflow as tf

from neuralmonkey.model.model_part import GenericModelPart
from neuralmonkey.model.feedable import Feedable
from neuralmonkey.model.parameterized import Parameterized

# pylint: disable=invalid-name
FeedDict = Dict[tf.Tensor, Union[int, float, np.ndarray]]
NextExecute = Tuple[Union[Dict, List], List[FeedDict]]
MP = TypeVar("MP", bound=GenericModelPart)
Executor = TypeVar("Executor", bound="GraphExecutor")
Runner = TypeVar("Runner", bound="BaseRunner")
OutputSeries = Union[List, np.ndarray]
# pylint: enable=invalid-name


class ExecutionResult(NamedTuple(
        "ExecutionResult",
        [("outputs", Dict[str, OutputSeries]),
         ("losses", Dict[str, float]),
         ("size", int),
         ("summaries", List[tf.Summary])])):
    """A data structure that represents the result of a graph execution.

    The goal of each graph executor is to populate this structure using its
    ``set_result`` function.

    Attributes:
        outputs: A dictionary mapping an output series to the batch of
            outputs of the graph executor.
        losses: A (possibly empty) list of loss values computed during the run.
        size: The length of the output batch.
        summaries: A list of TensorFlow summary objects fetched by the graph
            executor
    """


class GraphExecutor(GenericModelPart):
    """The abstract parent class of all graph executors.

    In Neural Monkey, a graph executor is an object that retrieves tensors
    from the computational graph. The two major groups of graph executors are
    trainers and runners.

    Each graph executor is an instance of `GenericModelPart` class, which means
    it has parameterized and feedable dependencies which reference the model
    part objects needed to be created in order to compute the tensors of
    interest (called "fetches").

    Every graph executor has a method called `get_executable`, which returns
    an `GraphExecutor.Executable` instance, which specifies what tensors to
    execute and collects results from the session execution.
    """

    class Executable(Generic[Executor]):
        """Abstract base class for executables.

        Executables are objects associated with the graph executors. Each
        executable has two main functions: `next_to_execute` and
        `collect_results`. These functions are called in a loop, until
        the executable's result has been set.

        To make use of Mypy's type checking, the executables are generic and
        are parameterized by the type of their graph executor. Since Python
        does not know the concept of nested classes, each executable receives
        the instance of the graph executor through its constructor.

        When subclassing `GraphExecutor`, it is also necessary to subclass
        the `Executable` class and name it `Executable`, so it overrides the
        definition of this class. Following this guideline, the default
        implementation of the `get_executable` function on the graph executor
        will work without the need of overriding it.
        """

        def __init__(self,
                     executor: Executor,
                     compute_losses: bool,
                     summaries: bool,
                     num_sessions: int) -> None:
            self._executor = executor
            self.compute_losses = compute_losses
            self.summaries = summaries
            self.num_sessions = num_sessions

            self._result = None  # type: Optional[ExecutionResult]

        def set_result(self,
                       outputs: Dict[str, OutputSeries],
                       losses: Dict[str, float],
                       size: int,
                       summaries: List[tf.Summary]) -> None:
            self._result = ExecutionResult(outputs, losses, size, summaries)

        @property
        def result(self) -> Optional[ExecutionResult]:
            return self._result

        @property
        def executor(self) -> Executor:
            return self._executor

        def next_to_execute(self) -> NextExecute:
            """Get the tensors and additional feed dicts for execution."""
            return self.executor.fetches, []

        @abstractmethod
        def collect_results(self, results: List[Dict]) -> None:
            return None

    def __init__(self,
                 dependencies: Set[GenericModelPart]) -> None:
        self._dependencies = dependencies
        self._feedables, self._parameterizeds = self.get_dependencies()

    def get_executable(self,
                       compute_losses: bool,
                       summaries: bool,
                       num_sessions: int) -> "GraphExecutor.Executable":
        # Since the executable is always subclassed, we can instantiate it
        return self.Executable(  # type: ignore
            self, compute_losses, summaries, num_sessions)

    @abstractproperty
    def fetches(self) -> Dict[str, tf.Tensor]:
        raise NotImplementedError()

    @property
    def dependencies(self) -> List[str]:
        return ["_dependencies"]

    @property
    def feedables(self) -> Set[Feedable]:
        return self._feedables

    @property
    def parameterizeds(self) -> Set[Parameterized]:
        return self._parameterizeds


class BaseRunner(GraphExecutor, Generic[MP]):
    """Base class for runners.

    Runners are graph executors that retrieve tensors from the model without
    changing the model parameters. Each runner has a top-level model part it
    relates to.
    """

    # pylint: disable=too-few-public-methods
    # Pylint issue here: https://github.com/PyCQA/pylint/issues/2607
    class Executable(GraphExecutor.Executable[Runner]):

        def next_to_execute(self) -> NextExecute:
            fetches = self.executor.fetches

            if not self.compute_losses:
                for loss in self.executor.loss_names:
                    fetches[loss] = tf.zeros([])

            return fetches, []

        def set_runner_result(self, outputs: OutputSeries,
                              losses: List[float], size: int = None,
                              summaries: List[tf.Summary] = None) -> None:
            if summaries is None:
                summaries = []

            if size is None:
                size = len(outputs)

            loss_names = ["{}/{}".format(self.executor.output_series, loss)
                          for loss in self.executor.loss_names]

            self.set_result({self.executor.output_series: outputs},
                            dict(zip(loss_names, losses)), size, summaries)
    # pylint: enable=too-few-public-methods

    def __init__(self,
                 output_series: str,
                 decoder: MP) -> None:
        GraphExecutor.__init__(self, {decoder})
        self.output_series = output_series
        # TODO(tf-data) rename decoder to something more general
        self.decoder = decoder

    @property
    def decoder_data_id(self) -> Optional[str]:
        return getattr(self.decoder, "data_id", None)

    @property
    def loss_names(self) -> List[str]:
        raise NotImplementedError()
