from typing import Any, Dict, Tuple, List, NamedTuple, Union
import numpy as np
import tensorflow as tf


# pylint: disable=invalid-name
FeedDict = Dict[tf.Tensor, Union[int, float, np.ndarray]]
NextExecute = Tuple[List[Any], Union[Dict, List], FeedDict]
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


def collect_encoders(coder):
    """Collect recusively all encoders and decoders."""
    if hasattr(coder, "encoders"):
        return set([coder]).union(*(collect_encoders(enc)
                                    for enc in coder.encoders))
    if hasattr(coder, "encoder"):
        return set([coder]).union(collect_encoders(coder.encoder))
    else:
        return set([coder])


class BaseRunner(object):
    def __init__(self, output_series: str, decoder) -> None:
        self.output_series = output_series
        self._decoder = decoder
        self.all_coders = collect_encoders(decoder)

    def get_executable(self, compute_losses=False,
                       summaries=True) -> Executable:
        raise NotImplementedError()

    @property
    def decoder_data_id(self) -> str:
        return self._decoder.data_id

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
    if len(outputs) > 0 and isinstance(outputs[0], np.ndarray):
        outputs = np.array(outputs)
    losses = [l / max(len(outputs), 1) for l in losses_sum]
    return ExecutionResult(outputs, losses,
                           execution_results[0].scalar_summaries,
                           execution_results[0].histogram_summaries,
                           execution_results[0].image_summaries)
