import tensorflow as tf

# tests: mypy,pylint

# pylint: disable=too-few-public-methods


class GreedyRunner(object):

    def __init__(self, decoder, evaluators):
        self.decoder = decoder
        self.evaluators = evaluators
        # TODO this needs to be done recursively - encoders can have encoders
        self.all_coders = set([decoder] + decoder.encoders)
        self.loss_names = ["train_loss", "runtime_loss"]

    def get_executable(self, train=False):
        if train:
            losses = [self.decoder.train_loss,
                      self.decoder.runtime_loss]
        else:
            losses = [tf.zeros([]), tf.zeros([])]
        to_run = losses + self.decoder.decoded
        return GreedyRunExecutable(self.all_coders, to_run, self.decoder.vocabulary)

    def collect_finished(self, execution_results):
        outputs = []
        losses_sum = [0. for _ in self.loss_names]
        for result in execution_results:
            outputs.extend(result.outputs)
            for i, loss in enumerate(result.losses):
                losses_sum[i] += loss
        losses = [l / len(outputs) for l in losses_sum]
        return outputs, losses

    def evaluate(self):
        pass

# pylint: disable=too-few-public-methods


class GreedyRunExecutable(object):

    def __init__(self, all_coders, to_run, vocabulary):
        self.all_coders = all_coders
        self.to_run = to_run
        self.vocabulary = vocabulary

        self.loss_with_gt_ins = 0.0
        self.loss_with_decoded_ins = 0.0
        self.decoded_sentences = []
        self.result = None

    def next_to_execute(self):
        """Get the feedables and tensors to run."""
        return self.all_coders, self.to_run

    def collect_results(self, results):
        decoded_sentences_batch = \
            self.vocabulary.vectors_to_sentences(results[2:])
        self.result = ExecutionResult(decoded_sentences_batch,
                                      results[:2])


class ExecutionResult(object):

    def __init__(self, outputs, losses):
        self.outputs = outputs
        self.losses = losses
