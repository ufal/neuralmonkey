from typing import Any, Dict, Tuple, List, NamedTuple, Union, Set
import numpy as np
import tensorflow as tf

from neuralmonkey.model.model_part import ModelPart

# pylint: disable=invalid-name
FeedDict = Dict[tf.Tensor, Union[int, float, np.ndarray]]
NextExecute = Tuple[Set[ModelPart], Union[Dict, List], FeedDict]
ExecutionResult = NamedTuple('ExecutionResult',
                             [('outputs', List[Any]),
                              ('losses', List[float]),
                              ('scalar_summaries', tf.Summary),
                              ('histogram_summaries', tf.Summary),
                              ('image_summaries', tf.Summary)])


class Executable(object):
    def next_to_execute(self) -> NextExecute:
        raise NotImplementedError()

    def collect_results(self, results: List[Dict]) -> None:
        raise NotImplementedError()


class BaseRunner(object):
    def __init__(self, output_series: str, decoder: ModelPart) -> None:
        self.output_series = output_series
        self._decoder = decoder
        self.all_coders = decoder.get_dependencies()

    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True) -> Executable:
        raise NotImplementedError()

    @property
    def decoder_data_id(self) -> str:
        if not hasattr(self._decoder, "data_id"):
            raise ValueError(
                "Top-level decoder {} does not have the 'data_id' attribute"
                .format(self._decoder.name))

        return getattr(self._decoder, "data_id")

    @property
    def loss_names(self) -> List[str]:
        raise NotImplementedError()


def reduce_execution_results(
        execution_results: List[ExecutionResult]) -> ExecutionResult:
    """Aggregate execution results into one."""
    outputs = []  # type: List[Any]
    losses_sum = [0. for _ in execution_results[0].losses]
    for result in execution_results:
        outputs.extend(result.outputs)
        for i, loss in enumerate(result.losses):
            losses_sum[i] += loss
        # TODO aggregate TensorBoard summaries
    if outputs and isinstance(outputs[0], np.ndarray):
        outputs = np.array(outputs)
    losses = [l / max(len(outputs), 1) for l in losses_sum]
    return ExecutionResult(outputs, losses,
                           execution_results[0].scalar_summaries,
                           execution_results[0].histogram_summaries,
                           execution_results[0].image_summaries)
