"""A runner outputing logits or normalized distriution from a decoder."""

from typing import Dict, List, Optional
from typeguard import check_argument_types

import numpy as np
import tensorflow as tf

from neuralmonkey.decoders.classifier import Classifier
from neuralmonkey.decorators import tensor
from neuralmonkey.runners.base_runner import BaseRunner


# pylint: disable=too-few-public-methods
class LogitsRunner(BaseRunner[Classifier]):
    """A runner which takes the output from decoder.decoded_logits.

    The logits / normalized probabilities are outputted as tab-separates string
    values. If the decoder produces a list of logits (as the recurrent
    decoder), the tab separated arrays are separated with commas.
    Alternatively, we may be interested in a single distribution dimension.
    """

    class Executable(BaseRunner.Executable["LogitsRunner"]):

        def collect_results(self, results: List[Dict]) -> None:
            if len(results) != 1:
                raise ValueError("LogitsRunner needs exactly 1 execution "
                                 "result, got {}".format(len(results)))

            train_loss = results[0]["train_loss"]
            runtime_loss = results[0]["runtime_loss"]

            # logits_list in shape (time, batch, vocab)
            logits_list = results[0]["logits"]

            # outputs are lists of strings (batch, time)
            outputs = [[] for _ in logits_list[0]]  # type: List[List[str]]

            for time_step in logits_list:
                for logits, output_list in zip(time_step, outputs):

                    if self.executor.normalize:
                        logits = np.exp(logits) / np.sum(np.exp(logits),
                                                         axis=0)
                    if self.executor.pick_index:
                        instance_logits = str(logits[self.executor.pick_index])
                    else:
                        instance_logits = ",".join(str(l) for l in logits)

                    output_list.append(instance_logits)

            str_outputs = [["\t".join(l)] for l in outputs]

            self.set_runner_result(outputs=str_outputs,
                                   losses=[train_loss, runtime_loss])

    def __init__(self,
                 output_series: str,
                 decoder: Classifier,
                 normalize: bool = True,
                 pick_index: int = None,
                 pick_value: str = None) -> None:
        """Initialize the logits runner.

        Args:
            output_series: Name of the series produced by the runner.
            decoder: A decoder having logits.
            normalize: Flag whether the logits should be normalized with
                softmax.
            pick_index: If not None, it specifies the index of the logit or the
                probability that should be on output.
            pick_value: If not None, it specifies a value from the decoder's
                vocabulary whose logit or probability should be on output.
        """
        check_argument_types()
        BaseRunner[Classifier].__init__(self, output_series, decoder)

        if pick_index is not None and pick_value is not None:
            raise ValueError("Either a pick index or a vocabulary value can "
                             "be specified, not both at the same time.")

        self.pick_index = None  # type: Optional[int]

        self.normalize = normalize
        if pick_value is not None:
            if pick_value in self.decoder.vocabulary:
                self.pick_index = self.decoder.vocabulary.index_to_word.index(
                    pick_value)
            else:
                raise ValueError(
                    "Value '{}' is not in vocabulary of decoder '{}'".format(
                        pick_value, decoder.name))
        else:
            self.pick_index = pick_index

    @tensor
    def fetches(self) -> Dict[str, tf.Tensor]:
        return {"logits": self.decoder.decoded_logits,
                "train_loss": self.decoder.train_loss,
                "runtime_loss": self.decoder.runtime_loss}

    @property
    def loss_names(self) -> List[str]:
        return ["train_loss", "runtime_loss"]
