import numpy as np
import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.encoders.recurrent import RecurrentEncoder
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.logging import warn
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.decorators import tensor


class WordAlignmentDecoder(ModelPart):
    """
    A decoder that computes soft alignment from an attentive encoder. Loss is
    computed as cross-entropy against a reference alignment.
    """

    def __init__(self,
                 encoder: RecurrentEncoder,
                 decoder: Decoder,
                 data_id: str,
                 name: str) -> None:
        ModelPart.__init__(self, name, None, None)

        self.encoder = encoder
        self.decoder = decoder
        self.data_id = data_id

        # TODO this is here to call the lazy properties which create
        # the list of attention distribbutions
        # pylint: disable=pointless-statement
        self.decoder.runtime_logits
        self.decoder.train_logits
        # pylint: enable=pointless-statement

        _, self.train_loss = self._make_decoder(runtime_mode=False)
        self.decoded, self.runtime_loss = self._make_decoder(runtime_mode=True)

        tf.summary.scalar("alignment_train_xent", self.train_loss,
                          collections=["summary_train"])

    @tensor
    def ref_alignment(self) -> tf.Tensor:
        return tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.decoder.max_output_len,
                   self.encoder.input_sequence.max_length],
            name="ref_alignment")

    @tensor
    def alignment_target(self) -> tf.Tensor:
        # shape will be [max_output_len, batch_size, max_input_len]
        return tf.transpose(self.ref_alignment, perm=[1, 0, 2])

    def _make_decoder(self, runtime_mode=False):
        attn_obj = self.decoder.get_attention_object(self.encoder,
                                                     not runtime_mode)
        if runtime_mode:
            alignment_logits = tf.stack(
                attn_obj.histories["{}_run".format(
                    self.decoder.name)],
                name="alignment_logits")
            # make batch_size the first dimension
            alignment = tf.transpose(tf.nn.softmax(alignment_logits),
                                     perm=[1, 0, 2])
            loss = tf.constant(0)
        else:
            alignment_logits = tf.stack(
                attn_obj.histories["{}_train".format(
                    self.decoder.name)],
                name="alignment_logits")
            alignment = None

            xent = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.alignment_target, logits=alignment_logits)
            loss = tf.reduce_sum(xent * self.decoder.train_padding)

        return alignment, loss

    @property
    def cost(self) -> tf.Tensor:
        return self.train_loss

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = {}

        alignment = dataset.get_series(self.data_id, allow_none=True)
        if alignment is None:
            if train:
                warn("Training alignment not present!")

            alignment = np.zeros((len(dataset),
                                  self.decoder.max_output_len,
                                  self.encoder.input_sequence.max_length),
                                 np.float32)

        fd[self.ref_alignment] = alignment

        return fd
