"""This module implements single-sentence-wise beam search."""
# tests: lint
import numpy as np

from neuralmonkey.learning_utils import feed_dicts
from neuralmonkey.encoders.sentence_encoder import SentenceEncoder
#from neuralmonkey.vocabulary import START_TOKEN, END_TOKEN

START_TOKEN_INDEX = 1
END_TOKEN_INDEX = 2

def sort_hypotheses(hyps, normalize_by_length=True):
    """Sort hypotheses based on log probs and length.

    Args:
        hyps: A list of hypothesis.
        normalize_by_length: Whether to normalize by length
    Returns:
        hyps: A list of sorted hypothesis in reverse log_prob order.
    """
    if normalize_by_length:
        return sorted(hyps, key=lambda h: h.log_prob/len(h.tokens),
                      reverse=True)
    else:
        return sorted(hyps, key=lambda h: h.log_prob, reverse=True)


class Hypothesis(object):
    """A class that represents a single hypothesis in a beam."""

    # pylint: disable=too-many-arguments
    # Maybe the logits can be refactored out (they serve only to compute loss)
    def __init__(self, tokens, log_prob, state, rnn_output=None, logits=None):
        """Construct a new hypothesis object

        Arguments:
            tokens: The list of already decoded tokens
            log_prob: The log probability of the decoded tokens given the model
            state: The last state of the decoder
            rnn_output: The last rnn_output of the decoder
            logits: The list of logits over the vocabulary (in time)
        """
        self.tokens = tokens
        self.log_prob = log_prob
        self.state = state
        self._rnn_output = rnn_output
        self._logits = [] if logits is None else logits

    # pylint: disable=too-many-arguments
    # Maybe the logits can be refactored out (they serve only to compute loss)
    def extend(self, token, log_prob, new_state, new_rnn_output, new_logits):
        """Return an extended version of the hypothesis.

        Arguments:
            token: The token to attach to the hypothesis
            log_prob: The log probability of emiting this token
            new_state: The RNN state of the decoder after decoding the token
            new_rnn_output: The RNN output tensor that emitted the token
            new_logits: The logits made from the RNN output
        """
        return Hypothesis(self.tokens + [token],
                          self.log_prob + log_prob,
                          new_state, new_rnn_output,
                          self._logits + [new_logits])

    @property
    def latest_token(self):
        """Get the last token from the hypothesis."""
        return self.tokens[-1]

    @property
    def rnn_output(self):
        """Get the last RNN output"""
        if self._rnn_output is None:
            raise Exception("Getting rnn_output before specifying it")
        return self._rnn_output

    @property
    def runtime_logits(self):
        """Get the sequence of logits (for computing loss)"""
        if self._logits is None:
            raise Exception("Getting logit sequence from empty hypothesis")
        return self._logits


# pylint: disable=too-few-public-methods
# Subject to issue #9
class BeamSearchRunner(object):
    """A runner that does beam search decoding. The beam search decoding is
    computed separately for each sentence in a batch so it's not as efficient
    as the batch solution. The implementation, however, is rather simple."""

    def __init__(self, decoder, beam_size):
        """Construct a new instance of the runner.

        Arguments:
            decoder: The decoder to use for decoding
            beam_size: How many alternative hypotheses to run
        """
        self.decoder = decoder
        self.vocabulary = decoder.vocabulary
        self.beam_size = beam_size

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    # Long function, whoever wants to refactor it, be my guest
    def __call__(self, sess, dataset, coders):
        """The caller method for the runner.

        Arguments:
            sess: The tensorflow session to use for computation
            dataset: The dataset to run the model on
            coders: The list of all encoders and decoders in the model
        """
        sentence_datasets = dataset.batch_dataset(1)
        decoded_sentences = []

        train_loss = 0.0
        runtime_loss = 0.0
        sentence_count = 0

        for sentence_ds in sentence_datasets:
            feed_dict = feed_dicts(sentence_ds, coders, train=False)
            sentence_count += 1

            # Run the encoders.
            # We want to fetch all the RNN states as well as outputs for each
            # encoder, and also the initial state of the decoder
            fetches = [self.decoder.runtime_rnn_states[0]]
            fetches += [e.encoded for e in coders
                        if isinstance(e, SentenceEncoder)]

            for encoder in coders:
                if isinstance(encoder, SentenceEncoder):
                    fetches += encoder.outputs_bidi

            computation = sess.run(fetches, feed_dict=feed_dict)

            # Use the fetched values to create the continuation feed dict
            init_feed_dict = {tensor: value for tensor, value in
                              zip(fetches, computation)}

            # Create empty hypotheses for beam search and prepare the list of
            # results
            initial_state = computation[0]
            hyps = [Hypothesis([START_TOKEN_INDEX], 0.0, initial_state)]
            results = []

            # Collect tensors of top K best tokens from the decoder
            topk_in_time = self.decoder.top_k_runtime_logprobs(self.beam_size)

            # Loop in time until we have enough results or we reached the
            # maximum output
            time_step = 0
            while (time_step < self.decoder.max_output
                   and len(results) < self.beam_size):

                # Gather the latest tokens and states from existing hypotheses,
                # prepare the list of candidate hypotheses
                #latest_tokens = [h.latest_token for h in hyps]
                #states = [h.state for h in hyps]
                candidate_hyps = []

                #prev_feed_dict = {}

                # Run one decoder step for each of the hypotheses
                for hyp in hyps:

                    # Fetch the next state and rnn output
                    # Note that there is one more states than outputs because
                    # the initial state is included in the list.
                    s_fetches = {
                        "state": self.decoder.runtime_rnn_states[time_step + 1],
                        "output": self.decoder.runtime_rnn_outputs[time_step],
                        "topk_val": topk_in_time[time_step][0],
                        "topk_ids": topk_in_time[time_step][1],
                        "logits": self.decoder.runtime_logits[time_step]
                    }

                    # For computation of the next step we use the feed dict
                    # created in the last time step and add the output from the
                    # encoder.
                    s_feed_dict = {
                        self.decoder.runtime_rnn_states[time_step]:
                        np.expand_dims(hyp.state, 0)}

                    s_feed_dict.update(init_feed_dict)
                    s_feed_dict.update(feed_dict)

                    if time_step > 0:
                        s_feed_dict[
                            self.decoder.runtime_rnn_outputs[time_step - 1]
                        ] = np.expand_dims(hyp.rnn_output, 0)

                    res = sess.run(s_fetches, s_feed_dict)

                    # Shapes and description of the tensors from "res":
                    # state:       (1, rnn_size)  [intermediate RNN state]
                    # output:      (1, rnn_size)  [projected cell_output]
                    # topk_values: (1, beam_size) [logprobs of words]
                    # topk_ids:    (1, beam_size) [indices to vocab]
                    # logits:      (1, vocabulary_sice) [logits to vocab]
                    state = res["state"][0]
                    output = res["output"][0]
                    topk_values = res["topk_val"][0]
                    topk_ids = res["topk_ids"][0]
                    logits = res["logits"][0]

                    # For each value and index from the topk list, create a
                    # candidate hypothesis that extends "hyp".
                    for val, index in zip(topk_values, topk_ids):
                        candidate_hyps.append(
                            hyp.extend(index, val, state, output, logits))

                # After the first time step, we should have exactly beam_size
                # hypotheses. After each next step, we should have beam_size
                # extension to each of that hypothesis, i.e. beam_size ^ 2.
                # The candidates are cropped to contain only beam_size
                # hypotheses after each step.
                if time_step > 0:
                    assert len(candidate_hyps) == self.beam_size ** 2
                else:
                    assert len(candidate_hyps) == self.beam_size

                # In order to select the best candidates, we need to sort the
                # hypotheses according to their log probabilities.  We don't
                # need to normalize by length because all the hypotheses has the
                # same length
                candidates_sorted = sort_hypotheses(candidate_hyps,
                                                    normalize_by_length=False)

                # Create a new list of hypotheses that contains the best K
                # candidate hypotheses.
                hyps = []

                for candidate in candidates_sorted:

                    # If the sentence has ended, put the hypothesis to the list
                    # of results. Otherwise, put the hypothesis to the pool.
                    if candidate.latest_token == END_TOKEN_INDEX:
                        results.append(candidate)
                    else:
                        hyps.append(candidate)

                    # We are done with this time step either when we have
                    # enough hypotheses in the beam for the next step,
                    # or if all the best hypotheses have ended.
                    if len(hyps) == self.beam_size:
                        break
                    if len(results) == self.beam_size:
                        break

                # Increment the time step and continue the loop
                time_step += 1

            # If we reach maximum time step, we need to add also unfinished
            # hypotheses.
            if time_step == self.decoder.max_output:
                results.extend(hyps)

            # Sort the results according tho the log probability. Now, the
            # normalization is necessary, because different result hypotheses
            # can have different lengths
            results_sorted = sort_hypotheses(results, normalize_by_length=True)
            best_hyp = results_sorted[0]

            # Decode the sentence from indices to vocabulary
            sent = self.vocabulary.vectors_to_sentences(
                [np.array(best_hyp.tokens)])[0]
            decoded_sentences.append(sent)

            # Now, compute the loss for the best hypothesis (if we have the
            # target data)
            if dataset.has_series(self.decoder.data_id):
                losses = {
                    "train_loss": self.decoder.train_loss,
                    "runtime_loss": self.decoder.runtime_loss}

                # The feed dict for losses needs train and runtime logits, train
                # targets and padding weights. The targets and paddings are in
                # the feed dict constructed in the beginning of this function.
                loss_feed_dict = {
                    tensor: np.expand_dims(state, 0) for tensor, state in zip(
                        self.decoder.runtime_logits, best_hyp.runtime_logits)}

                loss_feed_dict.update(feed_dict)
                loss_comp = sess.run(losses, feed_dict=loss_feed_dict)

                train_loss += loss_comp["train_loss"]
                runtime_loss += loss_comp["runtime_loss"]

        # The train and runtime loss is an average over the whole dataset
        train_loss /= sentence_count
        runtime_loss /= sentence_count

        return decoded_sentences, train_loss, runtime_loss
