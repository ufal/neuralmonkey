#tests: lint

import tensorflow as tf

from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.nn.projection import maxout

class BahdanauDecoder(Decoder):
    """This class implements the decoder as described in
    Bahdanau et al., 2015.

    The BahdanauDecoder brings the deep output with a hidden maxout layer
    to the computation of the RNN output (formulas for t and t_tilde on page
    14 of the article.
    """
    def __init__(self, encoders, vocabulary, data_id, maxout_size, **kwargs):
        """Create a new instance of the decoder.

        For the description of the arguments except maxout_size, see
        docs for neuralmonkey.decoders.decoder.Decoder.

        Arguments:
            maxout_size: The size of the hidden maxout layer (denoted as l in
                         the article.
        """
        self.maxout_size = maxout_size
        super().__init__(encoders, vocabulary, data_id, **kwargs)


    def _rnn_output_proj_params(self):
        """Create parameters for projection of RNN outputs to vocabulary
        indices.

        This method provides the projection parameters betweeen the maxout
        layer and the final logits.
        """
        weights = tf.get_variable(
            "maxout_to_word_W", [self.maxout_size, self.vocabulary_size])
        biases = tf.get_variable(
            "maxout_to_word_b",
            initializer=tf.zeros_initializer([self.vocabulary_size]))

        return weights, biases


    def _get_rnn_output(self, prev_state, prev_output, ctx_tensors):
        """Compute RNN output out of the previous state and output, and the
        context tensors returned from attention mechanisms, as described
        in the article

        This function corresponds to the equations for computation the
        t_tilde in the Bahdanau et al. (2015) paper, on page 14,
        with the maxout projection, before the last linear projection.

        Arguments:
            prev_state: Previous decoder RNN state.
            prev_output: Embedded output of the previous step.
            ctx_tensors: Context tensors computed by the attentions.

        Returns:
            Returns the maxout projection of the concatenated inputs
        """
        return maxout([prev_state, prev_output] + ctx_tensors,
                      self.maxout_size)
