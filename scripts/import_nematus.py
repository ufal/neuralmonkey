#!/usr/bin/env python3
"""Imports nematus model file and convert it into a neural monkey experiment
given a neural monkey configuration file.
"""
from typing import Dict, Tuple, List
import argparse
import json
import os
import numpy as np
import tensorflow as tf

from neuralmonkey.config.parsing import parse_file
from neuralmonkey.config.builder import build_config
from neuralmonkey.attention.feed_forward import Attention
from neuralmonkey.encoders.recurrent import SentenceEncoder
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.decoders.output_projection import nonlinear_output
from neuralmonkey.vocabulary import from_nematus_json
from neuralmonkey.logging import log as _log


def log(message: str, color: str = "blue") -> None:
    _log(message, color)


def check_shape(var1_tf: tf.Variable, var2_np: np.ndarray):
    if var1_tf.get_shape().as_list() != list(var2_np.shape):
        log("Shapes do not match! Exception will follow.", color="red")


def emb_fix_flat(variables: List[np.ndarray]) -> np.ndarray:
    return emb_fix(variables)


def emb_fix_dim1(variables: List[np.ndarray]) -> np.ndarray:
    return emb_fix(variables, dim=1)


def emb_fix(variables: List[np.ndarray], dim: int = 0) -> np.ndarray:
    if len(variables) != 1:
        raise ValueError("VocabFix only works with single vars. {} given."
                         .format(len(variables)))

    if dim != 0 and dim != 1:
        raise ValueError("dim can only be 0 or 1. is: {}".format(dim))

    variable = variables[0]
    shape = variable.shape

    # insert start token (hack from nematus - last from vocab - does it work?)
    to_insert = np.squeeze(variable[-1] if dim == 0 else variable[:, -1])
    variable = np.insert(variable, 0, to_insert, axis=dim)

    # insert padding token
    to_insert = np.zeros(shape[1 - dim]) if len(shape) > 1 else 0.
    variable = np.insert(variable, 0, to_insert, axis=dim)

    return variable

def sum_vars(variables: List[np.ndarray]) -> np.ndarray:
    return sum(variables)


def concat_vars(variables: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(variables)


def squeeze(variables: List[np.ndarray]) -> np.ndarray:
    if len(variables) != 1:
        raise ValueError("Squeeze only works with single vars. {} given."
                         .format(len(variables)))
    return np.squeeze(variables[0])

# pylint: disable=line-too-long
# No point in line wrapping
VARIABLE_MAP = {
    "encoder_input/embedding_matrix_0": (["Wemb"], emb_fix),
    "decoder/word_embeddings": (["Wemb_dec"], emb_fix),
    "decoder/state_to_word_W": (["ff_logit_W"], emb_fix_dim1),
    "decoder/state_to_word_b": (["ff_logit_b"], emb_fix_flat),
    "encoder/bidirectional_rnn/fw/OrthoGRUCell/gates/kernel": (["encoder_W", "encoder_U"], concat_vars),
    "encoder/bidirectional_rnn/fw/OrthoGRUCell/gates/bias": (["encoder_b"], None),
    "encoder/bidirectional_rnn/fw/OrthoGRUCell/candidate/kernel": (["encoder_Wx", "encoder_Ux"], concat_vars),
    "encoder/bidirectional_rnn/fw/OrthoGRUCell/candidate/bias": (["encoder_bx"], None),
    "encoder/bidirectional_rnn/bw/OrthoGRUCell/gates/kernel": (["encoder_r_W", "encoder_r_U"], concat_vars),
    "encoder/bidirectional_rnn/bw/OrthoGRUCell/gates/bias": (["encoder_r_b"], None),
    "encoder/bidirectional_rnn/bw/OrthoGRUCell/candidate/kernel": (["encoder_r_Wx", "encoder_r_Ux"], concat_vars),
    "encoder/bidirectional_rnn/bw/OrthoGRUCell/candidate/bias": (["encoder_r_bx"], None),
    "decoder/initial_state/encoders_projection/kernel": (["ff_state_W"], None),
    "decoder/initial_state/encoders_projection/bias": (["ff_state_b"], None),
    "decoder/attention_decoder/OrthoGRUCell/gates/kernel": (["decoder_W", "decoder_U"], concat_vars),
    "decoder/attention_decoder/OrthoGRUCell/gates/bias": (["decoder_b"], None),
    "decoder/attention_decoder/OrthoGRUCell/candidate/kernel": (["decoder_Wx", "decoder_Ux"], concat_vars),
    "decoder/attention_decoder/OrthoGRUCell/candidate/bias": (["decoder_bx"], None),
    "attention/attn_key_projection": (["decoder_Wc_att"], None),
    "attention/attn_projection_bias": (["decoder_b_att"], None),
    "attention/Attention/attn_query_projection": (["decoder_W_comb_att"], None),
    "attention/attn_similarity_v": (["decoder_U_att"], squeeze),
    "attention/attn_bias": (["decoder_c_tt"], squeeze),
    "decoder/attention_decoder/cond_gru_2_cell/gates/kernel": (["decoder_Wc", "decoder_U_nl"], concat_vars),
    "decoder/attention_decoder/cond_gru_2_cell/gates/bias": (["decoder_b_nl"], None),
    "decoder/attention_decoder/cond_gru_2_cell/candidate/kernel": (["decoder_Wcx", "decoder_Ux_nl"], concat_vars),
    "decoder/attention_decoder/cond_gru_2_cell/candidate/bias": (["decoder_bx_nl"], None),
    "decoder/attention_decoder/dense/kernel": (["ff_logit_lstm_W", "ff_logit_prev_W", "ff_logit_ctx_W"], concat_vars),
    "decoder/attention_decoder/dense/bias": (["ff_logit_lstm_b", "ff_logit_prev_b", "ff_logit_ctx_b"], sum_vars)
}
# pylint: enable=line-too-long

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


VOCABULARY_TEMPLATE = """\
[vocabulary_{}]
class=vocabulary.from_nematus_json
path="{}"
max_size={}
pad_to_max_size=True
"""

ENCODER_TEMPLATE = """\
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
        config["src_vocabulary"], max_size=config["n_words_src"],
        pad_to_max_size=True)

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


ATTENTION_TEMPLATE = """\
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


DECODER_TEMPLATE = """\
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
output_projection=<nonlinear>

[nonlinear]
class=decoders.output_projection.nonlinear_output
output_size={}
"""


def build_decoder(config: Dict,
                  attention: Attention,
                  encoder: SentenceEncoder) -> Tuple[Decoder, str]:
    vocabulary = from_nematus_json(
        config["tgt_vocabulary"],
        max_size=config["n_words_tgt"],
        pad_to_max_size=True)

    vocabulary_ini = VOCABULARY_TEMPLATE.format(
        "tgt", config["tgt_vocabulary"], config["n_words_tgt"])

    decoder = Decoder(
        name=DECODER_NAME,
        vocabulary=vocabulary,
        data_id="target",
        max_output_len=config["max_length"],
        embedding_size=config["embedding_size"],
        rnn_size=config["rnn_size"],
        encoders=[encoder],
        attentions=[attention],
        attention_on_input=False,
        conditional_gru=True,
        output_projection=nonlinear_output(config["embedding_size"]))

    decoder_ini = DECODER_TEMPLATE.format(
        DECODER_NAME, config["embedding_size"], config["rnn_size"],
        config["max_length"], config["embedding_size"])

    return decoder, "\n".join([vocabulary_ini, decoder_ini])


def build_model(config: Dict) -> Tuple[
        SentenceEncoder, Attention, Decoder, str]:
    encoder, encoder_cfg = build_encoder(config)
    attention, attention_cfg = build_attention(config, encoder)
    decoder, decoder_cfg = build_decoder(config, attention, encoder)

    ini = "\n".join([encoder_cfg, attention_cfg, decoder_cfg])

    return ini


def load_nematus_file(path: str) -> Dict[str, np.ndarray]:
    contents = np.load(path)
    cnt_dict = dict(contents)
    contents.close()
    return cnt_dict


def assign_vars(variables: Dict[str, np.ndarray]) -> List[tf.Tensor]:
    """For each variable in the map, assign the value from the dict"""

    trainable_vars = tf.trainable_variables()
    assign_ops = []

    for var in trainable_vars:
        map_key = var.op.name

        if map_key not in VARIABLE_MAP:
            raise ValueError("Map key {} not in variable map".format(map_key))

        nem_var_list, fun = VARIABLE_MAP[map_key]

        for nem_var in nem_var_list:
            if nem_var not in variables:
                raise ValueError("Alleged nematus var {} not found in loaded "
                                 "nematus vars.".format(nem_var))

        if fun is None:
            if len(nem_var_list) != 1:
                raise ValueError(
                    "Var list for map key {} must have length 1. "
                    "Length {} found instead."
                    .format(map_key, len(nem_var_list)))

            to_assign = variables[nem_var_list[0]]
        else:
            to_assign = fun([variables[v] for v in nem_var_list])

        check_shape(var, to_assign)
        assign_ops.append(tf.assign(var, to_assign))

    return assign_ops


INI_HEADER = """\
; This is an automatically generated configuration file
; for running imported nematus model
; For further training, set the configuration as appropriate

[main]
name="nematus imported translation"
tf_manager=<tf_manager>
output="{}"
runners=[<runner>]
postprocess=None
evaluation=[("target", evaluators.bleu.BLEU)]
runners_batch_size=1

; TODO Set these additional attributes for further training
; batch_size=80
; epochs=10
; train_dataset=<train_data>
; val_dataset=<val_data>
; trainer=<trainer>
; logging_period=20
; validation_period=60
; random_seed=1234

; [train_data]
; class=dataset.load_dataset_from_files
; s_source="PATH/TO/DATA" ; TODO do not forget to fill this out!
; s_target="PATH/TO/DATA" ; TODO do not forget to fill this out!
; lazy=True

; [val_data]
; class=dataset.load_dataset_from_files
; s_source="PATH/TO/DATA" ; TODO do not forget to fill this out!
; s_target="PATH/TO/DATA" ; TODO do not forget to fill this out!

; [trainer]
; class=trainers.cross_entropy_trainer.CrossEntropyTrainer
; decoders=[<decoder>]
; l2_weight=1.0e-8
; clip_norm=1.0

[tf_manager]
class=tf_manager.TensorFlowManager
num_threads=4
num_sessions=1

[runner]
class=runners.runner.GreedyRunner
decoder=<decoder>
output_series="target"
"""


def write_config(experiment_dir: str, ini: str) -> None:
    experiment_file = os.path.join(experiment_dir, "experiment.ini")
    with open(experiment_file, "w", encoding="utf-8") as f_out:
        f_out.write(INI_HEADER.format(experiment_dir))
        f_out.write(ini)


def prepare_output_dir(output_dir: str) -> bool:
    if os.path.isdir(output_dir):
        log("Directory {} already exists. Choose a nonexistent one.".
            format(output_dir))
        exit(1)

    os.mkdir(output_dir)


def main() -> None:
    log("Script started.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nematus_json", metavar="NEMATUS-JSON",
                        help="nematus json file")
    parser.add_argument("nematus_variables", metavar="NEMATUS-FILE",
                        help="nematus variable file")
    parser.add_argument("output_dir", metavar="INI-FILE",
                        help="output directory")
    args = parser.parse_args()

    log("Loading nematus variables from {}.".format(args.nematus_variables))
    nematus_vars = load_nematus_file(args.nematus_variables)

    log("Loading nematus JSON config from {}.".format(args.nematus_json))
    nematus_json_cfg = load_nematus_json(args.nematus_json)

    log("Bulding model.")
    ini = build_model(nematus_json_cfg)

    log("Defining assign Ops.")
    assign_ops = assign_vars(nematus_vars)

    log("Preparing output directory {}".format(args.output_dir))
    prepare_output_dir(args.output_dir)

    log("Writing configuration file to {}/experiment.ini."
        .format(args.output_dir))
    write_config(args.output_dir, ini)

    log("Creating TF session.")
    s = tf.Session()

    log("Running session to assign to Neural Monkey variables.")
    s.run(assign_ops)

    log("Initializing saver.")
    saver = tf.train.Saver()

    variables_file = os.path.join(args.output_dir, "variables.data")
    log("Saving variables to {}".format(variables_file))
    saver.save(s, variables_file)

    log("Finished.")


if __name__ == "__main__":
    main()
