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
            size = res["batch"]
            self.set_result(res, {}, size, [])
    # pylint: enable=too-few-public-methods

    def __init__(self) -> None:
        GraphExecutor.__init__(self, set())
        Feedable.__init__(self)

    @tensor
    def fetches(self) -> Dict[str, tf.Tensor]:
        assert self.dataset is not None
        # TODO(tf-data) this will change to fetch real data
        return {"batch": self.batch_size}
