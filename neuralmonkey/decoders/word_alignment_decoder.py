import numpy as np
import tensorflow as tf

from neuralmonkey.logging import log

# tests: mypy


class WordAlignmentDecoder(object):
    """
    A decoder that computes soft alignment from an attentive encoder. Loss is
    computed as cross-entropy against a reference alignment.
    """

    def __init__(self, encoder, decoder, data_id, name):
        self.encoder = encoder
        self.decoder = decoder
        self.data_id = data_id
        self.name = name

        self.ref_alignment = tf.placeholder(
            tf.float32,
            [None, self.decoder.max_output_len, self.encoder.max_input_len],
            name="ref_alignment")

        # shape will be [max_output_len, batch_size, max_input_len]
        self.alignment_target = tf.transpose(self.ref_alignment,
                                             perm=[1, 0, 2])

        _, self.train_loss = self._make_decoder(runtime_mode=False)
        self.decoded, self.runtime_loss = self._make_decoder(runtime_mode=True)

        tf.scalar_summary("alignment_train_xent", self.train_loss,
                          collections=["summary_train"])

    def _make_decoder(self, runtime_mode=False):
        attn_obj = self.decoder.get_attention_object(self.encoder,
                                                     runtime_mode)

        alignment_logits = tf.pack(attn_obj.logits_in_time,
                                   name="alignment_logits")

        if runtime_mode:
            # make batch_size the first dimension
            alignment = tf.transpose(tf.nn.softmax(alignment_logits),
                                     perm=[1, 0, 2])
            loss = tf.constant(0)
        else:
            alignment = None

            xent = tf.nn.softmax_cross_entropy_with_logits(
                alignment_logits, self.alignment_target)
            loss = tf.reduce_sum(xent * self.decoder.train_padding)

        return alignment, loss

    @property
    def cost(self):
        return self.train_loss

    def feed_dict(self, dataset, train=False):
        fd = {}

        alignment = dataset.get_series(self.data_id, allow_none=True)
        if alignment is None:
            if train:
                log("Warning: training alignment not present", color="red")

            alignment = np.zeros((len(dataset),
                                  self.decoder.max_output_len,
                                  self.encoder.max_input_len), np.float32)

        fd[self.ref_alignment] = alignment

        return fd
