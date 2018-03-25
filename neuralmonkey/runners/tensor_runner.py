from typing import Dict, List

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.logging import log
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, ExecutionResult, NextExecute)


class TensorExecutable(Executable):

    def __init__(self, all_coders, fetches, batch_dims):
        self._all_coders = all_coders
        self._fetches = fetches
        self._batch_dims = batch_dims
        self.result = None

    def next_to_execute(self) -> NextExecute:
        return self._all_coders, self._fetches, None

    def collect_results(self, results: List[Dict]) -> None:
        if len(results) != 1:
            raise ValueError("TensorRunner needs exactly 1 execution result, "
                             "got {}".format(len(results)))

        transposed = {}
        for name, val in results[0].items():
            batch_dim = self._batch_dims[name]

            perm = [batch_dim]
            for dim in range(len(val.shape)):
                if dim != batch_dim:
                    perm.append(dim)

            transposed_val = np.transpose(val, perm)
            transposed[name] = transposed_val

        # now we have dict of tensors in batch. we need
        # to have a batch of dicts with the batch dim removed
        batched = [dict(zip(transposed, col))
                   for col in zip(*transposed.values())]

        self.result = ExecutionResult(
            outputs=batched,
            losses=[],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class TensorRunner(BaseRunner[ModelPart]):

    def __init__(self,
                 toplevel_modelpart: ModelPart,
                 toplevel_tensors: List[tf.Tensor],
                 tensors_by_name: List[str],
                 tensors_by_ref: List[tf.Tensor],
                 batch_dims_by_name: List[int],
                 batch_dims_by_ref: List[int],
                 output_series: str) -> None:
        check_argument_types()
        BaseRunner[ModelPart].__init__(self, output_series, toplevel_modelpart)

        self._names = tensors_by_name
        self._tensors = tensors_by_ref
        self._batch_dims_name = batch_dims_by_name
        self._batch_dims_ref = batch_dims_by_ref

        log("Blessing toplevel tensors for tensor runner:")
        for tensor in toplevel_tensors:
            log("Toplevel tensor: {}".format(tensor))

    # pylint: disable=unused-argument
    def get_executable(self, *args, **kwargs) -> TensorExecutable:
        fetches = {}
        batch_ids = {}

        for name, bid in zip(self._names, self._batch_dims_name):
            fetches[name] = tf.get_default_graph().get_tensor_by_name(name)
            batch_ids[name] = bid

        for tensor, bid in zip(self._tensors, self._batch_dims_ref):
            fetches[tensor.name] = tensor
            batch_ids[tensor.name] = bid

        return TensorExecutable(self.all_coders, fetches, batch_ids)
    # pylint: enable=unused-argument

    @property
    def loss_names(self) -> List[str]:
        return []
