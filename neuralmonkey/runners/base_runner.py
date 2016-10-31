class BaseRunner(object):
    def __init__(self, decoder, evaluators):
        self.decoder = decoder
        self.evaluators = evaluators
        # TODO this needs to be done recursively - encoders can have encoders
        self.all_coders = set([decoder] + decoder.encoders)
        self.loss_names = []

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
