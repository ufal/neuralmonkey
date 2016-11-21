import tensorflow as tf
import numpy as np

from neuralmonkey.logging import log, debug
from neuralmonkey.learning_utils import feed_dicts

class EnsembleRunner(object):

    def __init__(self, decoder, batch_size):
        self.decoder = decoder
        self.batch_size = batch_size
        self.vocabulary = decoder.vocabulary


    def __call__(self, sessions, dataset, coders):

        batched_dataset = dataset.batch_dataset(self.batch_size)
        decoded_sentences_dataset = []

        train_loss = 0.0
        runtime_loss = 0.0
        batch_count = 0

        for batch in batched_dataset:
            batch_feed_dict = feed_dicts(batch, coders, train=False)
            batch_count += 1

            # if is a target sentence, compute also the losses
            # otherwise, just compute zero
            if dataset.has_series(self.decoder.data_id):
                losses = [self.decoder.train_loss,
                          self.decoder.runtime_loss]
            else:
                losses = [tf.zeros([]), tf.zeros([])]

            logprobs = []

            for sess in sessions:
                computation = sess.run(losses + self.decoder.runtime_logprobs,
                                       feed_dict=batch_feed_dict)

                train_loss += computation[0]
                runtime_loss += computation[1]
                logprobs.append(computation[len(losses):])


            ## logprobs is a list with shape ensemble x num_steps of items with
            ## shape batch x vocabulary

            ## First, we need to sum the logprobs along the ensemble axis
            ## (for each step).

            summed_logprobs = []

            for step in range(len(self.decoder.runtime_logprobs)):

                ## [l[step] for l in logprobs] is a list of len(ensembles)
                ## of numpy arrays of logprobs in the given step

                summed_logprobs.append(sum([l[step] for l in logprobs]))

            ## summed logprobs is a list of num_steps items of shape batch x
            ## vocabulary

            ## Next, we need to take argmaxes over the vocabulary in each step
            decoded = [np.argmax(l[:, 1:], 1) + 1 for l in summed_logprobs]
            decoded_sentences = self.vocabulary.vectors_to_sentences(decoded)
            decoded_sentences_dataset += decoded_sentences

        train_loss /= batch_count
        runtime_loss /= batch_count

        return decoded_sentences_dataset, train_loss, runtime_loss
