from typing import Dict, List

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.logging import log, warn
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, ExecutionResult, NextExecute)


class TensorExecutable(Executable):

    def __init__(self, all_coders, fetches, batch_dims, select_session):
        self._all_coders = all_coders
        self._fetches = fetches
        self._batch_dims = batch_dims
        self._select_session = select_session
        self.result = None

    def next_to_execute(self) -> NextExecute:
        return self._all_coders, self._fetches, None

    def collect_results(self, results: List[Dict]) -> None:
        if len(results) > 1 and self._select_session is None:
            sessions = []
            for res_dict in results:
                sessions.append(self._fetch_values_from_session(res_dict))

                # one call returns a list of dicts. we need to add another list
                # dimension in between, so it'll become a 2D list of dicts
                # with dimensions (batch, session, tensor_name)
                # the ``sessions`` structure is of 'shape'
                # (session, batch, tensor_name) so it should be sufficient to
                # transpose it:
                batched = list(zip(*sessions))
        else:
            batched = self._fetch_values_from_session(results[0])

        self.result = ExecutionResult(
            outputs=batched,
            losses=[],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)

    def _fetch_values_from_session(self, sess_results: Dict) -> List:

        transposed = {}
        for name, val in sess_results.items():
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

        return batched


class TensorRunner(BaseRunner[ModelPart]):
    """Runner class for printing tensors from a model.

    Use this runner if you want to retrieve a specific tensor from the model
    using a given dataset. The runner generates an output data series which
    will contain the tensors in a dictionary of numpy arrays.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 output_series: str,
                 toplevel_modelpart: ModelPart,
                 toplevel_tensors: List[tf.Tensor],
                 tensors_by_name: List[str],
                 tensors_by_ref: List[tf.Tensor],
                 batch_dims_by_name: List[int],
                 batch_dims_by_ref: List[int],
                 select_session: int = None) -> None:
        """Construct a new ``TensorRunner`` object.

        Note that at this time, one must specify the toplevel objects so that
        it is ensured that the graph is built. The reason for this behavior is
        that the graph is constructed lazily and therefore if the tensors to
        store are provided by indirect reference (name), the system does not
        know early enough that it needs to create them.

        Args:
            output_series: The name of the generated output data series.
            toplevel_modelpart: A ``ModelPart`` object that is used as the
                top-level component of the model. This object should depend on
                values of all the wanted tensors.
            toplevel_tensors: A list of tensors that should be constructed. Use
                this when the toplevel model part does not depend on this
                tensor. The tensors are constructed during running this
                constructor method which prints them out.
            tensors_by_name: A list of tensor names to fetch. If a tensor
                is not in the graph, a warning is generated and the tensor is
                ignored.
            tensors_by_ref: A list of tensor objects to fetch.
            batch_dims_by_name: A list of integers that correspond to the
                batch dimension in each wanted tensor specified by name.
            batch_dims_by_ref: A list of integers that correspond to the
                batch dimension in each wanted tensor specified by reference.
            select_session: An optional integer specifying the session to use
                in case of ensembling. When not used, tensors from all sessions
                are stored. In case of a single session, this option has no
                effect.
        """
        check_argument_types()
        BaseRunner[ModelPart].__init__(self, output_series, toplevel_modelpart)

        self._names = tensors_by_name
        self._tensors = tensors_by_ref
        self._batch_dims_name = batch_dims_by_name
        self._batch_dims_ref = batch_dims_by_ref
        self._select_session = select_session

        log("Blessing toplevel tensors for tensor runner:")
        for tensor in toplevel_tensors:
            log("Toplevel tensor: {}".format(tensor))
    # pylint: enable=too-many-arguments

    # pylint: disable=unused-argument
    def get_executable(self,
                       compute_losses: bool,
                       summaries: bool,
                       num_sessions: int) -> TensorExecutable:
        fetches = {}
        batch_ids = {}

        for name, bid in zip(self._names, self._batch_dims_name):
            try:
                fetches[name] = tf.get_default_graph().get_tensor_by_name(name)
                batch_ids[name] = bid
            except KeyError:
                warn(("The tensor of name '{}' is not present in the "
                      "graph.").format(name))

        for tensor, bid in zip(self._tensors, self._batch_dims_ref):
            fetches[tensor.name] = tensor
            batch_ids[tensor.name] = bid

        return TensorExecutable(
            self.all_coders, fetches, batch_ids, self._select_session)
    # pylint: enable=unused-argument

    @property
    def loss_names(self) -> List[str]:
        return []


class RepresentationRunner(TensorRunner):
    """Runner printing out representation from an encoder.

    Use this runner to get input / other data representation out from one of
    Neural Monkey encoders.
    """

    def __init__(self,
                 output_series: str,
                 encoder: ModelPart,
                 attribute: str = "output",
                 select_session: int = None) -> None:
        """Initialize the representation runner.

        Args:
            output_series: Name of the output series with vectors.
            encoder: The encoder to use. This can be any ``ModelPart`` object.
            attribute: The name of the encoder attribute that contains the
                data.
            used_session: Id of the TensorFlow session used in case of model
                ensembles.
        """
        check_argument_types()

        if not hasattr(encoder, attribute):
            raise TypeError("The encoder '{}' does not have the specified "
                            "attribute '{}'".format(encoder.name, attribute))

        tensor_to_get = getattr(encoder, attribute)

        TensorRunner.__init__(
            self,
            output_series,
            toplevel_modelpart=encoder,
            toplevel_tensors=[],
            tensors_by_name=[],
            tensors_by_ref=[tensor_to_get],
            batch_dims_by_name=[],
            batch_dims_by_ref=[0],
            select_session=select_session)
