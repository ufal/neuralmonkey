from typing import Dict, List

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decorators import tensor
from neuralmonkey.logging import warn
from neuralmonkey.model.model_part import GenericModelPart
from neuralmonkey.runners.base_runner import BaseRunner


class TensorRunner(BaseRunner[GenericModelPart]):
    """Runner class for printing tensors from a model.

    Use this runner if you want to retrieve a specific tensor from the model
    using a given dataset. The runner generates an output data series which
    will contain the tensors in a dictionary of numpy arrays.
    """

    # pylint: disable=too-few-public-methods
    # Pylint issue here: https://github.com/PyCQA/pylint/issues/2607
    class Executable(BaseRunner.Executable["TensorRunner"]):

        def collect_results(self, results: List[Dict]) -> None:
            if len(results) > 1 and self.executor.select_session is None:
                sessions = []
                for res_dict in results:
                    sessions.append(self._fetch_values_from_session(res_dict))

                    # one call returns a list of dicts. we need to add another
                    # list dimension in between, so it'll become a 2D list of
                    # dicts with dimensions (batch, session, tensor_name) the
                    # ``sessions`` structure is of 'shape' (session, batch,
                    # tensor_name) so it should be sufficient to transpose it:
                    batched = list(zip(*sessions))
            else:
                batched = self._fetch_values_from_session(results[0])

            self.set_runner_result(outputs=batched, losses=[])

        def _fetch_values_from_session(self, sess_results: Dict) -> List:

            transposed = {}
            for name, val in sess_results.items():
                batch_dim = self.executor.batch_ids[name]

                perm = ([batch_dim]
                        + [x for x in range(val.ndim) if x != batch_dim])

                transposed_val = np.transpose(val, perm)
                transposed[name] = transposed_val

            # now we have dict of tensors in batch. we need
            # to have a batch of dicts with the batch dim removed
            batched = [dict(zip(transposed, col))
                       for col in zip(*transposed.values())]
            return batched
    # pylint: enable=too-few-public-methods

    # pylint: disable=too-many-arguments
    def __init__(self,
                 output_series: str,
                 modelpart: GenericModelPart,
                 tensors: List[tf.Tensor],
                 batch_dims: List[int],
                 select_session: int = None):
        """Construct a new ``TensorRunner`` object.

        Note that at this time, one must specify the toplevel objects so that
        it is ensured that the graph is built. The reason for this behavior is
        that the graph is constructed lazily and therefore if the tensors to
        store are provided by indirect reference (name), the system does not
        know early enough that it needs to create them.

        Args:
            output_series: The name of the generated output data series.
            modelparts: ``GenericModelPart`` object that hold the
                tensors that will be retrieved.
            tensors: A list of names of tensors that should be retrieved.
            batch_dims_by_ref: A list of integers that correspond to the
                batch dimension in each wanted tensor.
            select_session: An optional integer specifying the session to use
                in case of ensembling. When not used, tensors from all sessions
                are stored. In case of a single session, this option has no
                effect.
        """
        check_argument_types()

        # TODO: remove ``modelpart'' altogether or use some dummy modelpart
        # here
        BaseRunner[GenericModelPart].__init__(
            self, output_series, modelpart)

        if len(tensors) != len(batch_dims):
            raise ValueError("TODO")

        self.modelpart = modelpart
        self.tensors = tensors
        self.batch_dims = batch_dims
        self.select_session = select_session

        self.batch_ids = {}  # type: Dict[str, int]
    # pylint: enable=too-many-arguments

    @tensor
    def fetches(self) -> Dict[str, tf.Tensor]:

        fetches = {}  # type: Dict[str, tf.Tensor]
        for tensor, bid in zip(self.tensors, self.batch_dims):
            try:
                fetches[tensor.name] = tensor
                self.batch_ids[tensor.name] = bid
            except KeyError:
                warn(("The tensor of name '{}' is not present in the "
                      "graph.").format(tensor.name))
        return fetches

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
                 encoder: GenericModelPart,
                 attribute: str = "output",
                 select_session: int = None) -> None:
        """Initialize the representation runner.

        Args:
            output_series: Name of the output series with vectors.
            encoder: The encoder to use. This can be any ``GenericModelPart``
                object.
            attribute: The name of the encoder attribute that contains the
                data.
            used_session: Id of the TensorFlow session used in case of model
                ensembles.
        """
        check_argument_types()

        if attribute not in dir(encoder):
            warn("The encoder '{}' seems not to have the specified "
                 "attribute '{}'".format(encoder, attribute))

        TensorRunner.__init__(
            self,
            output_series,
            modelparts=[encoder],
            tensors=[attribute],
            batch_dims=[0],
            tensors_by_name=[],
            batch_dims_by_name=[],
            select_session=select_session,
            single_tensor=True)
