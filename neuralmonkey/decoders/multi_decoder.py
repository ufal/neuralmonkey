import tensorflow as tf
import numpy as np

from neuralmonkey.vocabulary import PAD_TOKEN
from neuralmonkey.logging import log
from neuralmonkey.dataset import Dataset


class MultiDecoder(object):
    """The MultiDecoder class wraps a several child decoders into
    one parent encoder. The Neural Monkey architecture requires the model to
    have only one decoder, so this class can be used when more than one output
    sequence should be generated (i.e. multi-task learning).

    The multi decoder object composes of one main decoder and an arbitrary
    number of additional decoders, called 'regularization' decoders.

    The reason for this division is that during validation, we need to report
    a single score of the model as a whole, and based on this score, the
    training process decides whether to save the model variables or not.

    So if the task is translation with POS tagging of the source sentence, the
    main decoder should be the decoder that generates the target sentence,
    whereas the sequence labeler used for POS tagging should be included in the
    regularization decoders list.

    During training, the multi decoder works in the following way: According to
    the value of the ``_input_selector`` placeholder, the loss corresponds to
    one of the child decoders (so in multi-task learning, the weights in each
    batch are updated with respect only to one sub-task). It is therefore a
    good practice to alternate between batches of different task. This is
    because we often do not have the training data that cover all tasks in one
    corpus.

    """

    def __init__(self, main_decoder, regularization_decoders):
        """Create a new instance of the multi-decoder.

        Arguments:
            main_decoder: The decoder that corresponds to the output which
                we want at runtime.

            regularization_decoders: A list of the decoders among which the
                multidecoder will switch.
        """
        self.main_decoder = main_decoder

        self.regularization_decoders = regularization_decoders

        self._training_decoders = [main_decoder] + regularization_decoders
        self._decoder_costs = tf.concat([tf.expand_dims(d.cost, 0)
                                         for d in self._training_decoders], 0)

        self._scheduled_decoder = 0
        self._input_selector = tf.placeholder(tf.float32,
                                              [len(self._training_decoders)],
                                              name="input_decoder_selector")

        log("MultiDecoder initialized.")

    def all_decoded(self):
        return [d.decoded for d in self._training_decoders]

    @property
    def cost(self):
        # Without specifying dimension, returns a scalar.
        return tf.reduce_sum(self._decoder_costs * self._input_selector)

    @property
    def train_loss(self):
        return self.cost

    # The other @properties transparently point to self.main_encoder, because
    # they are only used when we want to get the decoded outputs.

    @property
    def vocabulary_size(self):
        return self.main_decoder.vocabulary_size

    @property
    def learning_step(self):
        # Maybe this should come from the current training decoder?
        # return self._training_decoders[self._scheduled_decoder].learning_step
        return self.main_decoder.learning_step

    @property
    def runtime_loss(self):
        return self.main_decoder.runtime_loss

    @property
    def decoded(self):
        return self.main_decoder.decoded

    @property
    def vocabulary(self):
        return self.main_decoder.vocabulary

    @property
    def data_id(self):
        return self.main_decoder.data_id

    def feed_dict(self, dataset, train=False):
        """Populate the feed dictionary for the decoder object

        Decoder placeholders:
            ``input_selector``: the index of the child decoder used for
                                computing the loss
        """
        # Two options:
        #  - call feed_dict only on the currently selected child decoder
        #  - call feed_dict preventatively everywhere (with dummy data)

        # First option:
        # fd = self.decoders[self._scheduled_decoder].feed_dict(
        #     dataset, train=train)
        # First option does not seem to work, so the second option
        # will have to do.

        # Second option:
        # (This is a fallback plan in canse TensorFlow requires us to fill in
        #  the data placeholders for the nodes that are hidden behind a zero
        #  through self.input_selector.)

        # pylint: disable=invalid-name
        # fd stands for feed_dict
        fd = {}
        for i, decoder in enumerate(self._training_decoders):
            if i == self._scheduled_decoder:
                fd_i = decoder.feed_dict(dataset, train=train)
            else:
                # serie is a generator of lists of words (i.e. sentences)
                serie = [[PAD_TOKEN] for _ in range(len(dataset))]
                dummy_dataset = Dataset("dummy", {decoder.data_id: serie}, {})
                fd_i = decoder.feed_dict(dummy_dataset, train=train)
            fd.update(fd_i)

        # We now need to set the value of our input_selector placeholder
        # as well.
        input_selector_value = np.zeros(len(self._training_decoders))
        input_selector_value[self._scheduled_decoder] = 1
        fd[self._input_selector] = input_selector_value

        # Schedule update
        self._scheduled_decoder = ((self._scheduled_decoder + 1)
                                   % len(self._training_decoders))

        return fd
