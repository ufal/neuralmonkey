# TODO untested module
from typing import cast, Dict, Tuple

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.encoders.recurrent import RecurrentEncoder
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.model.sequence import Sequence
from neuralmonkey.decorators import tensor


class WordAlignmentDecoder(ModelPart):
    """A decoder that computes soft alignment from an attentive encoder.

    Loss is computed as cross-entropy against a reference alignment.
    """

    def __init__(self,
                 encoder: RecurrentEncoder,
                 decoder: Decoder,
                 data_id: str,
                 name: str,
                 reuse: ModelPart = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        ModelPart.__init__(self, name, reuse, None, None, initializers)

        self.encoder = encoder
        self.decoder = decoder
        self.data_id = data_id

    @property
    def enc_input(self) -> Sequence:
        if not isinstance(self.encoder.input_sequence, Sequence):
            raise TypeError("Expected Sequence type in encoder.input_sequence")

        return cast(Sequence, self.encoder.input_sequence)

    @property
    def input_types(self) -> Dict[str, tf.DType]:
        return {self.data_id: tf.float32}

    @property
    def input_shapes(self) -> Dict[str, tf.TensorShape]:
        return {self.data_id: tf.TensorShape(
            [None, self.decoder.max_output_len, self.enc_input.max_length])}

    @tensor
    def ref_alignment(self) -> tf.Tensor:
        return self.dataset[self.data_id]

    @tensor
    def alignment_target(self) -> tf.Tensor:
        # shape will be [max_output_len, batch_size, max_input_len]
        return tf.transpose(self.ref_alignment, perm=[1, 0, 2])

    @tensor
    def train_loss(self) -> tf.Tensor:
        loss = self._make_decoder(runtime_mode=False)
        tf.summary.scalar(
            "alignment_train_xent", loss, collections=["summary_train"])

        return loss

    # pylint: disable=unsubscriptable-object
    # Bug in pylint
    @tensor
    def decoded(self) -> tf.Tensor:
        return self.runtime_outputs[0]

    @tensor
    def runtime_loss(self) -> tf.Tensor:
        return self.runtime_outputs[1]
    # pylint: enable=unsubscriptable-object

    @tensor
    def runtime_outputs(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self._make_decoder(runtime_mode=True)

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
