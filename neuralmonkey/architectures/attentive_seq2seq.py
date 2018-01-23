from typeguard import check_argument_types

from neuralmonkey.attention import Attention
from neuralmonkey.decoders import Decoder
from neuralmonkey.decoders.encoder_projection import EncoderProjection
from neuralmonkey.decoders.output_projection import OutputProjectionSpec
from neuralmonkey.encoders import SentenceEncoder
from neuralmonkey.vocabulary import Vocabulary

class AttentiveSeq2Seq(object):

    def __init__(
            self,
            name: str,
            encoder_vocabulary: Vocabulary,
            decoder_vocabulary: Vocabulary,
            encoder_data_id: str,
            decoder_data_id: str,
            rnn_size: int,
            embedding_size: int,
            max_output_len: int,
            max_input_len: int = None,
            dropout_keep_prob: float = 1.0,
            output_projection: OutputProjectionSpec = None,
            encoder_projection: EncoderProjection = None,
            attention_on_input: bool = False,
            rnn_cell: str = "GRU",
            conditional_gru: bool = False,
            save_checkpoint: str = None,
            load_checkpoint: str = None) -> None:
        check_argument_types()

        enc_name = "{}_encoder".format(name)
        e_sckp = "enc_{}".format(save_checkpoint) if save_checkpoint else None
        e_lckp = "enc_{}".format(load_checkpoint) if load_checkpoint else None

        dec_name = "{}_decoder".format(name)
        d_sckp = "dec_{}".format(save_checkpoint) if save_checkpoint else None
        d_lckp = "dec_{}".format(load_checkpoint) if load_checkpoint else None

        att_name = "{}_attention".format(name)
        a_sckp = "att_{}".format(save_checkpoint) if save_checkpoint else None
        a_lckp = "att_{}".format(load_checkpoint) if load_checkpoint else None

        self.encoder = SentenceEncoder(
            name=enc_name,
            vocabulary=encoder_vocabulary,
            data_id=encoder_data_id,
            embedding_size=embedding_size,
            rnn_size=rnn_size,
            rnn_cell=rnn_cell,
            max_input_len=max_input_len,
            dropout_keep_prob=dropout_keep_prob,
            save_checkpoint=e_sckp,
            load_checkpoint=e_lckp)

        self.attention = Attention(
            name=att_name,
            encoder=self.encoder,
            dropout_keep_prob=dropout_keep_prob,
            save_checkpoint=a_sckp,
            load_checkpoint=a_lckp)

        self.decoder = Decoder(
            encoders=[self.encoder],
            vocabulary=decoder_vocabulary,
            data_id=decoder_data_id,
            name=dec_name,
            max_output_len=max_output_len,
            dropout_keep_prob=dropout_keep_prob,
            rnn_size=rnn_size,
            embedding_size=embedding_size,
            output_projection=output_projection,
            encoder_projection=encoder_projection,
            attentions=[self.attention],
            attention_on_input=attention_on_input,
            rnn_cell=rnn_cell,
            conditional_gru=conditional_gru,
            save_checkpoint=d_sckp,
            load_checkpoint=d_lckp)
