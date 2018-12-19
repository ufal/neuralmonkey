from typing import List, Dict
import tensorflow as tf

from neuralmonkey.decorators import tensor
from neuralmonkey.model.feedable import Feedable
from neuralmonkey.runners.base_runner import GraphExecutor


class DatasetRunner(GraphExecutor, Feedable):

    # pylint: disable=too-few-public-methods
    # Pylint issue here: https://github.com/PyCQA/pylint/issues/2607
    class Executable(GraphExecutor.Executable["DatasetRunner"]):

        def collect_results(self, results: List[Dict]) -> None:
            res = results[0]

            # Assuming the size of all dataset tensors is the same and also
            # assuming that all of the dataset series are present, even if in
            # the form of dummy series.
            size = len(res[next(iter(res))])
            self.set_result(res, {}, size, [])
    # pylint: enable=too-few-public-methods

    def __init__(self) -> None:
        GraphExecutor.__init__(self, set())
        Feedable.__init__(self)

        self.string_series = []  # type: List[str]

    def register_input(self, dataset: tf.data.Dataset) -> None:
        super().register_input(dataset)
        self.string_series = [
            key for key in dataset if hasattr(
                dataset[key], "dtype") and dataset[key].dtype == tf.string]

    @tensor
    def fetches(self) -> Dict[str, tf.Tensor]:
        assert self.dataset is not None
        return self.dataset
