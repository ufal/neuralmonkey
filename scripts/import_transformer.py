#!/usr/bin/env python3
"""Imports transformer model checkpoint file and convert it into a neural monkey experiment
given a neural monkey configuration file.
"""
from typing import Dict, Tuple, List
import argparse
import os
import numpy as np
import tensorflow as tf
import json

from neuralmonkey.logging import log as _log
from neuralmonkey.vocabulary import from_t2t_vocabulary, Vocabulary
from neuralmonkey.encoders.transformer import TransformerEncoder
from neuralmonkey.decoders.transformer import TransformerDecoder
from neuralmonkey.model.sequence import EmbeddedSequence

ENCODER_NAME = "encoder"
DECODER_NAME = "decoder"

# pylint: disable=line-too-long
# No point in line wrapping
def create_variable_map(hparams: Dict, np_vars) -> Dict:

    # Always present
    VARIABLE_MAP = {
        # This one is added to the source embeddings
        "TODO: Not Implemented": (["body/target_space_embedding/kernel"], None),
        "encoder_input/embedding_matrix_0": (get_shared_emb_vars(np_vars), emb_fix),
        "encoder/LayerNorm/beta": (["transformer/body/encoder/layer_prepostprocess/layer_norm/layer_norm_bias"], None),
        "encoder/LayerNorm/gamma": (["transformer/body/encoder/layer_prepostprocess/layer_norm/layer_norm_scale"], None),
        "decoder/LayerNorm/beta": (["transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_bias"], None),
        "decoder/LayerNorm/gamma": (["transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_scale"], None)
    }

    

    for i in range(hparams["depth"]):
        # Encoder
        VARIABLE_MAP.update({
            "encoder/layer_{}/self_attention/query_proj/kernel".format(i): (["transformer/body/encoder/layer_{}/self_attention/multihead_attention/q/kernel".format(i)], None),
            "encoder/layer_{}/self_attention/keys_proj/kernel".format(i): (["transformer/body/encoder/layer_{}/self_attention/multihead_attention/k/kernel".format(i)], None),
            "encoder/layer_{}/self_attention/vals_proj/kernel".format(i): (["transformer/body/encoder/layer_{}/self_attention/multihead_attention/v/kernel".format(i)], None),
            "encoder/layer_{}/self_attention/output_proj/kernel".format(i): (["transformer/body/encoder/layer_{}/self_attention/multihead_attention/output_transform/kernel".format(i)], None),
            "encoder/layer_{}/self_attention/LayerNorm/beta".format(i): (["transformer/body/encoder/layer_{}/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias".format(i)], None),
            "encoder/layer_{}/self_attention/LayerNorm/gamma".format(i): (["transformer/body/encoder/layer_{}/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale".format(i)], None),
            "encoder/layer_{}/feedforward/hidden_state/kernel".format(i): (["transformer/body/encoder/layer_{}/ffn/conv1/kernel".format(i)], None),
            "encoder/layer_{}/feedforward/hidden_state/bias".format(i): (["transformer/body/encoder/layer_{}/ffn/conv1/bias".format(i)], None),
            "encoder/layer_{}/feedforward/output/kernel".format(i): (["transformer/body/encoder/layer_{}/ffn/conv2/kernel".format(i)], None),
            "encoder/layer_{}/feedforward/output/bias".format(i): (["transformer/body/encoder/layer_{}/ffn/conv2/bias".format(i)], None),
            "encoder/layer_{}/feedforward/LayerNorm/beta".format(i): (["transformer/body/encoder/layer_{}/ffn/layer_prepostprocess/layer_norm/layer_norm_bias".format(i)], None),
            "encoder/layer_{}/feedforward/LayerNorm/gamma".format(i): (["transformer/body/encoder/layer_{}/ffn/layer_prepostprocess/layer_norm/layer_norm_scale".format(i)], None)})

        # Decoder
        VARIABLE_MAP.update({
            "decoder/layer_{}/encdec_attention/query_proj/kernel".format(i): (["transformer/body/decoder/layer_{}/encdec_attention/multihead_attention/q/kernel".format(i)], None),
            "decoder/layer_{}/encdec_attention/keys_proj/kernel".format(i): (["transformer/body/decoder/layer_{}/encdec_attention/multihead_attention/k/kernel".format(i)], None),
            "decoder/layer_{}/encdec_attention/vals_proj/kernel".format(i): (["transformer/body/decoder/layer_{}/encdec_attention/multihead_attention/v/kernel".format(i)], None),
            "decoder/layer_{}/encdec_attention/output_proj/kernel".format(i): (["transformer/body/decoder/layer_{}/encdec_attention/multihead_attention/output_transform/kernel".format(i)], None),
            "decoder/layer_{}/encdec_attention/LayerNorm/beta".format(i): (["transformer/body/decoder/layer_{}/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_bias".format(i)], None),
            "decoder/layer_{}/encdec_attention/LayerNorm/gamma".format(i): (["transformer/body/decoder/layer_{}/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_scale".format(i)], None),
            "decoder/layer_{}/self_attention/query_proj/kernel".format(i): (["transformer/body/decoder/layer_{}/self_attention/multihead_attention/q/kernel".format(i)], None),
            "decoder/layer_{}/self_attention/keys_proj/kernel".format(i): (["transformer/body/decoder/layer_{}/self_attention/multihead_attention/k/kernel".format(i)], None),
            "decoder/layer_{}/self_attention/vals_proj/kernel".format(i): (["transformer/body/decoder/layer_{}/self_attention/multihead_attention/v/kernel".format(i)], None),
            "decoder/layer_{}/self_attention/output_proj/kernel".format(i): (["transformer/body/decoder/layer_{}/self_attention/multihead_attention/output_transform/kernel".format(i)], None),
            "decoder/layer_{}/self_attention/LayerNorm/beta".format(i): (["transformer/body/decoder/layer_{}/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias".format(i)], None),
            "decoder/layer_{}/self_attention/LayerNorm/gamma".format(i): (["transformer/body/decoder/layer_{}/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale".format(i)], None),
            "decoder/layer_{}/feedforward/hidden_state/kernel".format(i): (["transformer/body/decoder/layer_{}/ffn/conv1/kernel".format(i)], None),
            "decoder/layer_{}/feedforward/hidden_state/bias".format(i): (["transformer/body/decoder/layer_{}/ffn/conv1/bias".format(i)], None),
            "decoder/layer_{}/feedforward/output/kernel".format(i): (["transformer/body/decoder/layer_{}/ffn/conv2/kernel".format(i)], None),
            "decoder/layer_{}/feedforward/output/bias".format(i): (["transformer/body/decoder/layer_{}/ffn/conv2/bias".format(i)], None),
            "decoder/layer_{}/feedforward/LayerNorm/beta".format(i): (["transformer/body/decoder/layer_{}/ffn/layer_prepostprocess/layer_norm/layer_norm_bias".format(i)], None),
            "decoder/layer_{}/feedforward/LayerNorm/gamma".format(i): (["transformer/body/decoder/layer_{}/ffn/layer_prepostprocess/layer_norm/layer_norm_scale".format(i)], None)})

    return VARIABLE_MAP
# pylint: enable=line-too-long


def log(message: str, color: str = "blue") -> None:
    _log(message, color)

def check_shape(var1_tf: tf.Variable, var2_np: np.ndarray):
    if var1_tf.get_shape().as_list() != list(var2_np.shape):
        log("Shapes do not match! Exception will follow.", color="red")

def get_shared_emb_vars(np_vars: Dict) -> List[str]:
    modality_vars = [var for var in np_vars if "symbol_modality" in var]
    modality_prefix = modality_vars[0].split("/")[:-1]
    sorted_vars = []
    for i in range(len(modality_vars)):
        modality_arr = modality_prefix + ["weights_{}".format(i)]
        sorted_vars.append("/".join(modality_arr))
    return sorted_vars
    

def emb_fix(variables: List[tf.Tensor]) -> tf.Tensor:
    """Concat sharded embedding matrix and include NMonkey special symbols.

    We need to include embeddings for 
    """
    concat = np.concatenate(variables, axis=0)
    concat_split = np.split(concat, [1, 2], axis=0)

    emb_shape = concat.shape[-1]
    return np.concatenate([concat_split[0],
                           concat_split[0],
                           concat_split[1],
                           np.zeros([1, emb_shape]),
                           concat_split[2]], axis=0)


VOCABULARY_TEMPLATE = """\
[vocabulary]
class=vocabulary.from_t2t_vocabulary
path="{}"
"""

ENCODER_TEMPLATE = """\
[input_sequence]
class=model.sequence.EmbeddedSequence
name="{}"
vocabulary=<vocabulary>
data_id="source_wp"
embedding_size={}
multiply_embedding_mode="{}"
max_length={}

[encoder]
class=encoders.transformer.TransformerEncoder
name="{}"
input_sequence=<input_sequence>
ff_hidden_size={}
depth={}
n_heads={}
dropout_keep_prob=1.0
attention_dropout_keep_prob=1.0
"""


def build_encoder(hparams: Dict,
                  vocab_path: str) -> Tuple[TransformerEncoder, str]:
    vocabulary = from_t2t_vocabulary(vocab_path)
    vocabulary_ini = VOCABULARY_TEMPLATE.format(vocab_path)

    inp_seq_name = "{}_input".format(ENCODER_NAME)
    inp_seq = EmbeddedSequence(
        name=inp_seq_name,
        vocabulary=vocabulary,
        data_id="source_wp",
        embedding_size=hparams["embedding_size"],
        multiply_embedding_mode=hparams["multiply_embedding_mode"])

    encoder = TransformerEncoder(
        name=ENCODER_NAME,
        input_sequence=inp_seq,
        ff_hidden_size=hparams["ff_hidden_size"],
        depth=hparams["depth"],
        n_heads=hparams["n_heads"])

    encoder_ini = ENCODER_TEMPLATE.format(
        inp_seq_name, hparams["embedding_size"],
        hparams["multiply_embedding_mode"], hparams["max_length"],
        ENCODER_NAME, hparams["ff_hidden_size"], hparams["depth"],
        hparams["n_heads"])

    return encoder, vocabulary, "\n".join([vocabulary_ini, encoder_ini])


DECODER_TEMPLATE = """\
[decoder]
class=decoders.transformer.TransformerDecoder
name="{}"
vocabulary=<vocabulary>
data_id="target"
encoder=<encoder>
ff_hidden_size={}
n_heads_self={}
n_heads_enc={}
depth={}
embedding_size={}
embeddings_source=<input_sequence>
max_output_len=50
dropout_keep_prob=1.0
attention_dropout_keep_prob=1.0
"""


def build_decoder(hparams: Dict,
                  encoder: TransformerEncoder,
                  vocab: Vocabulary) -> Tuple[
        TransformerDecoder, str]:
    decoder = TransformerDecoder(
        name=DECODER_NAME,
        encoder=encoder,
        vocabulary=vocab,
        data_id="target_wp",
        ff_hidden_size=hparams["ff_hidden_size"],
        n_heads_self=hparams["n_heads"],
        n_heads_enc=hparams["n_heads"],
        depth=hparams["depth"],
        embeddings_source=encoder.input_sequence,
        embedding_size=hparams["embedding_size"],
        max_output_len=50)

    decoder_ini = DECODER_TEMPLATE.format(
        DECODER_NAME, hparams["ff_hidden_size"], hparams["n_heads"],
        hparams["n_heads"], hparams["depth"], hparams["embedding_size"])

    return decoder, "\n".join([decoder_ini])


def build_model(hparams: Dict,
                vocab_path: str) -> Tuple[
        TransformerEncoder, TransformerDecoder, str]:
    encoder, vocab, encoder_cfg = build_encoder(hparams, vocab_path)
    decocer, decoder_cfg = build_decoder(hparams, encoder, vocab)

    ini = "\n".join([encoder_cfg, decoder_cfg])

    return ini


def load_hparams(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f_json:
        contents = json.load(f_json)

    hparams = {
        "n_heads": contents["num_heads"],
        "ff_hidden_size": contents["filter_size"],
        "embedding_size": contents["hidden_size"],
        "max_length": contents["max_length"],
        "label_smoothing": contents["label_smoothing"],
        "depth": contents["num_hidden_layers"],
        "multiply_embedding_mode": contents["multiply_embedding_mode"]
    }

    # TODO: check, whether the hparams that we do not set in NeuralMonkey
    # are set to correct values

    return hparams

def assign_vars(hparams: Dict, np_vars: Dict) -> List[tf.Tensor]:
    trainable_vars = tf.trainable_variables()
    assign_ops = []

    var_map = create_variable_map(hparams, np_vars)
    for var in trainable_vars:
        map_key = var.op.name

        if map_key not in var_map:
            raise ValueError("Map key {} not in variable map".format(map_key))

        t2t_var_list, fun = var_map[map_key]

        for t2t_var in t2t_var_list:
            if t2t_var not in np_vars:
                raise ValueError("Alleged transformer var {} not found "
                                 "in loaded transformer vars. For neuralmonkey"
                                 " var {}.".format(t2t_var, map_key))

        if fun is None:
            if len(t2t_var_list) != 1:
                raise ValueError(
                    "Var list for map key {} must have length 1. "
                    "Length {} found instead."
                    .format(map_key, len(t2t_var_list)))
            to_assign = np_vars[t2t_var_list[0]]
        else:
            to_assign = fun([np_vars[v] for v in t2t_var_list])

        check_shape(var, to_assign)
        assign_ops.append(tf.assign(var, to_assign))

    return assign_ops


INI_HEADER = """\
; This is an automatically generated configuration file
; for running imported nematus model
; For further training, set the configuration as appropriate

[main]
name="t2t-transformer imported translation"
tf_manager=<tf_manager>
output="{}"
runners=[<runner>]
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

[wp_preprocess]
class=processors.wordpiece.WordpiecePreprocessor
vocabulary=<vocabulary>

[wp_postprocess]
class=processors.wordpiece.WordpiecePostprocessor

; [train_data]
; class=dataset.load_dataset_from_files
; s_source="PATH/TO/DATA" ; TODO do not forget to fill this out!
; s_target="PATH/TO/DATA" ; TODO do not forget to fill this out!
; preprocessors=[("source", "source_wp", <wp_preprocess>), ("target", "target_wp", <wp_postprocess>)]
; lazy=True

; [val_data]
; class=dataset.load_dataset_from_files
; s_source="PATH/TO/DATA" ; TODO do not forget to fill this out!
; s_target="PATH/TO/DATA" ; TODO do not forget to fill this out!
; preprocessors=[("source", "source_wp", <wp_preprocess>), ("target", "target_wp", <wp_postprocess>)]

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
postprocess=<wp_postprocess>
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
    parser.add_argument("--t2t_checkpoint", metavar="T2T-CHECKPOINT",
                        help="t2t checkpoint file")
    parser.add_argument("--t2t_hparams", metavar="T2T-HPARAMS",
                        help="t2t hparams json file")
    parser.add_argument("--vocabulary", metavar="VOCABULARY",
                        help="vocabulary file")
    parser.add_argument("--output_dir", metavar="OUTPUT-DIR",
                        help="output directory")
    args = parser.parse_args()

    ckpt = args.t2t_checkpoint

    log("Loading transformer hparams JSON from {}.".format(args.t2t_hparams))
    hparams = load_hparams(args.t2t_hparams)

    log("Bulding model.")
    ini = build_model(hparams, args.vocabulary)

    log("Read from checkpoint {}.".format(ckpt))
    t2t_var_list = tf.contrib.framework.list_variables(ckpt)
    t2t_reader = tf.contrib.framework.load_checkpoint(ckpt)
    t2t_var_values = {}
    for (name, shape) in t2t_var_list:
        if name.startswith("training"):
            continue
        t2t_var_values[name] = t2t_reader.get_tensor(name)

    log("Defining assign_ops.")
    assign_ops = assign_vars(hparams, t2t_var_values)

    log("Preparing output directory {}.".format(args.output_dir))
    prepare_output_dir(args.output_dir)

    log("Writing configuration file to {}/experiment.ini."
        .format(args.output_dir))
    write_config(args.output_dir, ini)

    log("Creating TF session.")
    s =  tf.Session()

    log("Running session to assign to Neural Monkey variables.")
    s.run(assign_ops)

    log("Initializing saver.")
    saver = tf.train.Saver()

    variables_file = os.path.join(args.output_dir, "variables.data")
    log("Saving variables to {}.".format(variables_file))
    saver.save(s, variables_file)

    log("Finished.")

if __name__ == "__main__":
    main()
