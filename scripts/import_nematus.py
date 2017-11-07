#!/usr/bin/env python3
"""Imports nematus model file and convert it into a neural monkey experiment
given a neural monkey configuration file.
"""
from typing import Dict, Tuple
import argparse
import json
import os
import numpy as np

from neuralmonkey.config.parsing import parse_file
from neuralmonkey.config.builder import build_config
from neuralmonkey.attention.feed_forward import Attention
from neuralmonkey.encoders.recurrent import SentenceEncoder
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.vocabulary import from_nematus_json


ENCODER_NAME = "encoder"
DECODER_NAME = "decoder"
ATTENTION_NAME = "attention"


def load_nematus_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f_json:
        contents = json.load(f_json)

    prefix = os.path.realpath(os.path.dirname(path))
    config = {
        "encoder_type": contents["encoder"],
        "decoder_type": contents["decoder"],
        "n_words_src": contents["n_words_src"],
        "n_words_tgt": contents["n_words"],
        "variables_file": contents["saveto"],
        "rnn_size": contents["dim"],
        "embedding_size": contents["dim_word"],
        "src_vocabulary": os.path.join(
            prefix, contents["dictionaries"][0]),
        "tgt_vocabulary": os.path.join(
            prefix, contents["dictionaries"][1]),
        "max_length": contents["maxlen"]
    }

    if config["encoder_type"] != "gru":
        raise ValueError("Unsupported encoder type: {}"
                         .format(config["encoder_type"]))

    if config["decoder_type"] != "gru_cond":
        raise ValueError("Unsupported decoder type: {}"
                         .format(config["decoder_type"]))

    if not os.path.isfile(config["src_vocabulary"]):
        raise FileNotFoundError("Vocabulary file not found: {}"
                                .format(config["src_vocabulary"]))

    if not os.path.isfile(config["tgt_vocabulary"]):
        raise FileNotFoundError("Vocabulary file not found: {}"
                                .format(config["tgt_vocabulary"]))

    return config


VOCABULARY_TEMPLATE = """
[vocabulary_{}]
class=vocabulary.from_nematus_json
path="{}"
max_size={}
"""

ENCODER_TEMPLATE = """
[encoder]
class=encoders.SentenceEncoder
name="{}"
vocabulary=<vocabulary_src>
data_id="source"
embedding_size={}
rnn_size={}
max_input_len={}
"""


def build_encoder(config: Dict) -> Tuple[SentenceEncoder, str]:
    vocabulary = from_nematus_json(
        config["src_vocabulary"],
        max_size=config["n_words_src"])

    vocabulary_ini = VOCABULARY_TEMPLATE.format(
        "src", config["src_vocabulary"], config["n_words_src"])

    encoder = SentenceEncoder(
        name=ENCODER_NAME,
        vocabulary=vocabulary,
        data_id="source",
        embedding_size=config["embedding_size"],
        rnn_size=config["rnn_size"])

    encoder_ini = ENCODER_TEMPLATE.format(
        ENCODER_NAME, config["embedding_size"],
        config["rnn_size"], config["max_length"])

    return encoder, "\n".join([vocabulary_ini, encoder_ini])


ATTENTION_TEMPLATE = """
[attention]
class=attention.Attention
name="{}"
encoder=<encoder>
"""


def build_attention(config: Dict,
                    encoder: SentenceEncoder) -> Tuple[Attention, str]:
    attention = Attention(
        name=ATTENTION_NAME,
        encoder=encoder)

    attention_ini = ATTENTION_TEMPLATE.format(ATTENTION_NAME)

    return attention, attention_ini


DECODER_TEMPLATE = """
[decoder]
class=decoders.Decoder
name="{}"
vocabulary=<vocabulary_tgt>
data_id="target"
embedding_size={}
rnn_size={}
max_output_len={}
encoders=[<encoder>]
attentions=[<attention>]
attention_on_input=False
conditional_gru=True
"""


def build_decoder(config: Dict,
                  attention: Attention,
                  encoder: SentenceEncoder) -> Tuple[Decoder, str]:
    vocabulary = from_nematus_json(
        config["tgt_vocabulary"],
        max_size=config["n_words_tgt"])

    vocabulary_ini = VOCABULARY_TEMPLATE.format(
        "tgt", config["tgt_vocabulary"], config["n_words_tgt"])

    decoder = Decoder(
        name=DECODER_NAME,
        vocabulary=vocabulary,
        data_id="target",
        embedding_size=config["embedding_size"],
        rnn_size=config["rnn_size"],
        encoders=[encoder],
        attentions=[attention],
        attention_on_input=False,
        conditional_gru=True)

    decoder_ini = DECODER_TEMPLATE.format(
        DECODER_NAME, config["embedding_size"],
        config["rnn_size"], config["max_length"])

    return decoder, "\n".join([vocabulary_ini, decoder_ini])


def build_model(config: Dict) -> Tuple[
        SentenceEncoder, Attention, Decoder, str]:
    encoder, encoder_cfg = build_encoder(config)
    attention, attention_cfg = build_attention(config, encoder)
    decoder, decoder_cfg = build_decoder(config, attention, encoder)

    ini = "\n".join([encoder_cfg, attention_cfg, decoder_cfg])

    return encoder, attention, decoder, ini


def load_nematus_file(path: str) -> Dict[str, np.ndarray]:
    contents = np.load(path)
    cnt_dict = dict(contents)
    contents.close()
    return cnt_dict


def assign_encoder_vars(
        encoder: SentenceEncoder,
        variables: Dict[str, np.ndarray]) -> List[tf.Tensor]:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nematus-json", metavar="NEMATUS-JSON",
                        help="nematus json file")
    parser.add_argument("nematus-variables", metavar="NEMATUS-FILE",
                        help="nematus variable file")
    parser.add_argument("config", metavar="INI-FILE",
                        help="a configuration file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        _, config_dict = parse_file(f)

    build_config(config_dict, ignore_names=set())


if __name__ == "__main__":
    main()
