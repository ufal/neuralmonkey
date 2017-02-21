

class LabelRunner(BaseRunner):

    def __init__(self,
                 output_series: str,
                 decoder: Any,
                 postprocess: Callable[[List[str]], List[str]]=None) -> None:
        super(GreedyRunner, self).__init__(output_series, decoder)
        self._postprocess = postprocess
        self.

    def get_executable(self, compute_losses=False, summaries=True):
        if compute_losses:
            fetches = {"loss": self._decoder.cost}

        fetches["label_logprobs"] = self._decoder.logprobs

        return GreedyRunExecutable(self.all_coders, fetches,
                                   self._decoder.vocabulary,
                                   self._postprocess)

    @property
    def loss_names(self) -> List[str]:
        return ["loss"]


class LabelRunExecutable(Executable):

    def __init__(self, all_coders, fetches, vocabulary, postprocess):
        self.all_coders = all_coders
        self._fetches = fetches
        self._vocabulary = vocabulary
        self._postprocess = postprocess

        self.decoded_labels = []
        self.result = None  # type: Option[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self.all_coders, self._fetches, {}

    def collect_results(self, results: List[Dict]) -> None:

        loss = 0.
        summed_logprobs = [-np.inf for _ in self._fetches["label_logprobs"]]

        for sess_result in results:
            loss += sess_result["loss"]

            for i, logprob in enumerate(sess_result["label_logprobs"]):
                summed_logprobs[i] = np.logaddexp(summed_logprobs[i], logprob)

        argmaxes = [np.argmax(l, axis=1) for l in summed_logprobs]

        decoded_labels = self._vocabulary.vectors_to_sentences(argmaxes)

        if self._postprocess is not None:
            decoded_labels = self._postprocess(decoded_labels)

        self.result = ExecutionResult(
            outputs=decoded_labels,
            losses=[loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)
