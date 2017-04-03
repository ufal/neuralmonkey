
from typing import NamedTuple, Tuple, Optional

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decorators import tensor
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.decoders.decoder import Decoder

# pylint: disable=invalid-name
SearchState = NamedTuple("SearchState",
                         [("logprob_sum", tf.Tensor),  # beam x 1
                          ("lengths", tf.Tensor),  # beam x 1
                          ("finished", tf.Tensor),  # beam x 1
                          ("last_word_ids", tf.Tensor),  # beam x 1
                          ("last_state", tf.Tensor),  # beam x rnn_size
                          ("last_attns", tf.Tensor)])  # beam x ???

SearchStepOutput = NamedTuple("SearchStepOutput",
                              [("scores", tf.Tensor),
                               ("parent_ids", tf.Tensor),
                               ("predicted_ids", tf.Tensor)])
# pylint: enable=invalid-name




class BeamSearchDecoder(ModelPart):

    def __init__(self,
                 name: str,
                 parent_decoder: Decoder,
                 beam_size: int,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        assert check_argument_types()

        self._parent_decoder = parent_decoder
        self._beam_size = beam_size

        self._cell = self._parent_decoder.get_rnn_cell()



    def step(self, bs_state: SearchState) -> Tuple[SearchState,
                                                   SearchStepOutput]:

        # collect attention objects
        att_objects = [self._parent_decoder.get_attention_object(e, False)
                       for e in self._parent_decoder.encoders]
        att_objects = [a for a in att_objects if a is not None]

        # embed the previously decoded word
        input_ = self._parent_decoder._embed_and_dropout(
            bs_state.last_word_ids)

        # don't want to use this decoder with uninitialized parent
        assert self._parent_decoder.step_scope.reuse

        # run the parent decoder decoding step
        # shapes:
        # logits: beam x vocabulary
        # state: beam x rnn_size
        # attns: encoder x beam x context vector size
        logits, state, attns = self._parent_decoder.step(
            att_objects, _input, bs_state.last_state, bs_state.last_attns)

        # mask the probabilities
        # shape(logprobs) = beam x vocabulary
        logprobs = (1 - bs_state.finished) * tf.nn.log_softmax(logits)

        # update hypothesis scores
        # shape(hyp_probs) = beam x vocabulary
        hyp_probs = bs_state.logprob_sum + logprobs

        # update hypothesis lengths
        hyp_lengths = bs_state.lengths + bs_state.unfinished

        # TODO do this the google way
        # shape(scores) = beam x vocabulary
        scores = hyp_probs / hyp_lengths

        # flatten so we can use top_k
        scores_flat = tf.reshape(scores, [-1])

        # shape(both) = beam
        topk_scores, topk_indices = tf.nn.top_k(scores_flat, self._beam_size)

        topk_scores.set_shape([self._beam_size])
        topk_indices.set_shape([self._beam_size])

        # flatten the hypothesis probabilities
        hyp_probs_flat = tf.reshape(hyp_probs, [-1])

        # select logprobs of the best hyps (disregard lenghts)
        next_logprob_sum = tf.gather(hyp_probs_flat, topk_indices)
        next_logprob_sum.set_shape([self._beam_size])

        next_word_ids = tf.mod(topk_indices,
                               len(self._parent_decoder.vocabulary))

        next_beam_ids = tf.div(topk_indices,
                               len(self._parent_decoder.vocabulary))

        next_beam_prev_state = tf.gather(state, next_beam_ids)
        next_beam_prev_attns = [tf.gather(a, next_beam_ids) for a in attns]

        # update finished flags
        has_just_finished = tf.equal(next_word_ids, END_TOKEN_INDEX)
        next_finished = tf.logical_or(
            tf.gather(bs_state.finished, next_beam_ids),
            has_just_finished)

        output = SearchStepOutput(
            scores=topk_scores,
            parent_ids=next_beam_ids,
            predicted_ids=next_word_ids)

        search_state = SearchState(
            logprob_sum=next_logprob_sum,
            lengths=hyp_lengths,
            finished=next_finished,
            last_word_ids=next_word_ids,
            last_state=next_beam_prev_state,
            last_attns=next_beam_prev_attns)

        return search_state, output



    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        """Populate the feed dictionary for the decoder object

        Arguments:
            dataset: The dataset to use for the decoder.
            train: Boolean flag, telling whether this is a training run
        """
        assert not train
        assert len(dataset) == 1

        # sentences = cast(Iterable[List[str]],
        #                  dataset.get_series(self.data_id, allow_none=True))

        # if sentences is None and train:
        #     raise ValueError("When training, you must feed "
        #                      "reference sentences")

        # sentences_list = list(sentences) if sentences is not None else None

        # TODO assert ze je to jen jedna veta

        fd = {}  # type: FeedDict

        # if sentences is not None:
        #     # train_mode=False, since we don't want to <unk>ize target words!
        #     inputs, weights = self.vocabulary.sentences_to_tensor(
        #         sentences_list, self.max_output_len, train_mode=False,
        #         add_start_symbol=False, add_end_symbol=True)

        #     assert inputs.shape == (self.max_output_len, len(sentences_list))
        #     assert weights.shape == (self.max_output_len, len(sentences_list))

        #     fd[self.train_inputs] = inputs
        #     fd[self.train_padding] = weights

        return fd
