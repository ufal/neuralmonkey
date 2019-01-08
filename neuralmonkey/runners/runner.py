from typing import Dict, List, Callable, Union

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.runners.base_runner import BaseRunner, NextExecute
from neuralmonkey.decoders.autoregressive import AutoregressiveDecoder
from neuralmonkey.decoders.classifier import Classifier
from neuralmonkey.decorators import tensor

# pylint: disable=invalid-name
SupportedDecoder = Union[AutoregressiveDecoder, Classifier]
Postprocessor = Callable[[List[List[str]]], List[List[str]]]
# pylint: enable=invalid-name


class GreedyRunner(BaseRunner[SupportedDecoder]):

    class Executable(BaseRunner.Executable["GreedyRunner"]):

        def next_to_execute(self) -> NextExecute:
            """Get the tensors and additional feed dicts for execution."""
            fetches = self.executor.fetches

            if not self.summaries:
                fetches["image_summaries"] = None

            if not self.compute_losses:
                fetches["train_xent"] = tf.zeros([])
                fetches["runtime_xent"] = tf.zeros([])

            return fetches, []

        def collect_results(self, results: List[Dict]) -> None:
            train_loss = 0.
            runtime_loss = 0.
            summed_logprobs = [-np.inf for _ in range(
                results[0]["decoded_logprobs"].shape[0])]

            for sess_result in results:
                train_loss += sess_result["train_xent"]
                runtime_loss += sess_result["runtime_xent"]

                for i, logprob in enumerate(sess_result["decoded_logprobs"]):
                    summed_logprobs[i] = np.logaddexp(
                        summed_logprobs[i], logprob)

            argmaxes = [np.argmax(l, axis=1) for l in summed_logprobs]

            decoded_tokens = self.executor.vocabulary.vectors_to_sentences(
                argmaxes)

            if self.executor.postprocess is not None:
                decoded_tokens = self.executor.postprocess(decoded_tokens)

            summaries = None
            if "image_summaries" in results[0]:
                summaries = [results[0]["image_summaries"]]

            self.set_runner_result(
                outputs=decoded_tokens, losses=[train_loss, runtime_loss],
                summaries=summaries)

    def __init__(self,
                 output_series: str,
                 decoder: SupportedDecoder,
                 postprocess: Postprocessor = None) -> None:
        check_argument_types()
        BaseRunner[AutoregressiveDecoder].__init__(
            self, output_series, decoder)

        self.postprocess = postprocess
        self.vocabulary = self.decoder.vocabulary

    @tensor
    def fetches(self) -> Dict[str, tf.Tensor]:

        fetches = {"decoded_logprobs": self.decoder.runtime_logprobs,
                   "train_xent": self.decoder.train_loss,
                   "runtime_xent": self.decoder.runtime_loss}

        att_plot_summaries = tf.get_collection("summary_att_plots")
        if att_plot_summaries:
            fetches["image_summaries"] = tf.summary.merge(att_plot_summaries)

        return fetches

    @property
    def loss_names(self) -> List[str]:
        return ["train_xent", "runtime_xent"]
