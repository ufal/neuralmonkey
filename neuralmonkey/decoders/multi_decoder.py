import math
import tensorflow as tf
import numpy as np

from neuralmonkey.vocabulary import START_TOKEN
from neuralmonkey.decoding_function import attention_decoder
from neuralmonkey.logging import log
from neuralmonkey.dataset import Dataset


class MultiDecoder(object):
    """A class that manages parts of the computation graph that are
    used for the decoding.
    """

    # pylint: disable=too-many-instance-attributes,too-many-locals
    # Big decoder cannot be simpler. Not sure if refactoring
    # it into smaller units would be helpful
    # Some locals may be turned to attributes

    def __init__(self, main_decoder, regularization_decoders, **kwargs):
        """Create a new instance of the multi-decoder.

        Arguments:
            main_decoder: The decoder that corresponds to the output which
                we want at runtime.

            additional_decoders: A list of the decoders among which the multidecoder
                will switch.

        Keyword arguments:

        """
        self.main_decoder = main_decoder

        self.regularization_decoders = regularization_decoders

        self._training_decoders = [main_decoder] + regularization_decoders
        self._decoder_costs = tf.concat(0, [tf.expand_dims(d.cost, 0)
                                            for d in self._training_decoders])

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
        """TODO: rewrite for multidecoder

        Populate the feed dictionary for the decoder object

        Decoder placeholders:
            ``input_selector``
        """

        # Two options:
        #  - call feed_dict only on the currently selected child decoder
        #  - call feed_dict preventatively everywhere (with dummy data)

        # First option:
        # fd = self.decoders[self.scheduled_decoder].feed_dict(dataset, train=train)

        # Second option:
        # (This is a fallback plan in canse TensorFlow requires us to fill in
        #  the data placeholders for the nodes that are hidden behind a zero
        #  through self.input_selector.)
        #
        fd = {}
        for i, d in enumerate(self._training_decoders):
            if i == self.scheduled_decoder:
                fd_i = d.feed_dict(dataset, train=train)
            else:
                # serie je generator seznamu slov (vet)
                # <pad> is PAD_TOKEN
                serie = [["<pad>"] for _ in range(len(dataset))]
                dummy_dataset = Dataset("dummy", {d.data_id: serie}, {})
                fd_i = d.feed_dict(dummy_dataset, train=train)
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
