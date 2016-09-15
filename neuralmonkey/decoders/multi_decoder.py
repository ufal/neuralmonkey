#tests: lint

import math
import tensorflow as tf
import numpy as np

from neuralmonkey.vocabulary import START_TOKEN
from neuralmonkey.decoding_function import attention_decoder
from neuralmonkey.logging import log


class MultiDecoder(object):
    """A class that manages parts of the computation graph that are
    used for the decoding.
    """

    # pylint: disable=too-many-instance-attributes,too-many-locals
    # Big decoder cannot be simpler. Not sure if refactoring
    # it into smaller units would be helpful
    # Some locals may be turned to attributes

    def __init__(self, decoders, **kwargs):
        """Create a new instance of the multi-decoder.

        Arguments:
            decoders: A list of the decoders among which the multidecoder
                will switch.

        Keyword arguments:

        """

        self.decoders = decoders
        self.decoder_costs = tf.concat(0, [tf.expand_dims(d.cost, 0) for d in self.decoders])

        self.scheduled_decoder = 0
        self.input_selector = tf.placeholder(tf.float32,
                                             [len(decoders)],
                                             name="input_decoder_selector")

        log("MultiDecoder initialized.")

    def all_decoded(self):
        return [d.decoded for d in self.decoders]

    @property
    def vocabulary_size(self):
        raise NotImplementedError('MultiDecoder does not have a vocabulary.'
                                  ' If needed, implement looking inside the list'
                                  ' of decoders according to self.input_selector.')
        # return len(self.vocabulary)

    @property
    def learning_step(self):
        return self.decoders[self.scheduled_decoder].learning_step

    @property
    def cost(self):
        # Without specifying dimension, returns a scalar.
        return tf.reduce_sum(self.decoder_costs * self.input_selector)


    @property
    def vocabulary(self):
        return self.decoders[self.scheduled_decoder].vocabulary


    @property
    def data_id(self):
        return self.decoders[self.scheduled_decoder].data_id


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
        fd = self.decoders[self.scheduled_decoder].feed_dict(dataset, train=train)

        # Second option:
        # (This is a fallback plan in canse TensorFlow requires us to fill in
        #  the data placeholders for the nodes that are hidden behind a zero
        #  through self.input_selector.)
        #
        # fd = {}
        # for i, d in enumerate(self.decoders):
        #     if i == self.scheduled_decoder:
        #         fd_i = self.decoders[self.scheduled_decoder].feed_dict(dataset, train=train)
        #     else:
        #         fd_i = d.feed_dict([], train=train)
        #     fd.update(fd)

        # We now need to set the value of our input_selector placeholder
        # as well.
        input_selector_value = np.zeros(len(self.decoders))
        input_selector_value[self.scheduled_decoder] = 1
        fd[self.input_selector] = input_selector_value

        # Schedule update
        self.scheduled_decoder = ((self.scheduled_decoder + 1)
                                  % len(self.decoders))

        return fd
