#!/bin/bash

set -ex

export NEURALMONKEY_STRICT=1
export PYTHONFAULTHANDLER=1

bin/neuralmonkey-train tests/vocab.ini
bin/neuralmonkey-train tests/bahdanau.ini
NEURALMONKEY_STRICT= bin/neuralmonkey-train tests/bpe.ini
bin/neuralmonkey-train tests/bpe.ini -s 'decoder.encoders=[<encoder_output_frozen>]' -s 'attention.encoder=<encoder_states_frozen>' -s 'main.initial_variables=["tests/outputs/bpe/variables.data"]'
#bin/neuralmonkey-train tests/alignment.ini
bin/neuralmonkey-train tests/post-edit.ini
bin/neuralmonkey-train tests/factored.ini
bin/neuralmonkey-train tests/classifier.ini
bin/neuralmonkey-train tests/labeler.ini
bin/neuralmonkey-train tests/regressor.ini
bin/neuralmonkey-train tests/language-model.ini
bin/neuralmonkey-train tests/audio-classifier.ini
bin/neuralmonkey-train tests/ctc.ini
bin/neuralmonkey-train tests/beamsearch.ini
bin/neuralmonkey-train tests/self-critical.ini
bin/neuralmonkey-train tests/rl.ini
bin/neuralmonkey-train tests/transformer.ini

# Testing environment variable substitution in config file
NM_EXPERIMENT_NAME=small bin/neuralmonkey-train tests/small.ini
export NM_EXPERIMENT_NAME='"small"'
bin/neuralmonkey-run tests/small.ini tests/test_data.ini
bin/neuralmonkey-run tests/small.ini tests/test_data.ini --json /dev/stdout \
    | python -c 'import sys,json; print(json.load(sys.stdin)[0]["target/bleu"])'
unset NM_EXPERIMENT_NAME

bin/neuralmonkey-train tests/small_sent_cnn.ini

# Ensembles testing
score_single=$(bin/neuralmonkey-run tests/beamsearch.ini tests/test_data_ensembles_single.ini --json /dev/stdout | python -c 'import sys,json;print(json.load(sys.stdin)[0]["target_beam.rank001/beam_search_score"])')
score_ensemble=$(bin/neuralmonkey-run tests/beamsearch_ensembles.ini tests/test_data_ensembles_duplicate.ini --json /dev/stdout | python -c 'import sys,json;print(json.load(sys.stdin)[0]["target_beam.rank001/beam_search_score"])')

echo "SINGLE SCORE: $score_single"
echo "ENSEMBLE OF THE SAME VARS SCORE: $score_ensemble"

if [ "${score_single:0:8}" != "${score_ensemble:0:8}" ] ; then
    echo "SCORES DO NOT MATCH" >&2
    exit 1
fi

bin/neuralmonkey-run tests/beamsearch_ensembles.ini tests/test_data_ensembles_all.ini

NM_EXPERIMENT_NAME=small bin/neuralmonkey-server --configuration=tests/small.ini --port=5000 &
SERVER_PID=$!
sleep 20

curl 127.0.0.1:5000/run -H "Content-Type: application/json" -X POST -d '{"source": ["I am the eggman.", "I am the walrus ."]}'
kill $SERVER_PID

bin/neuralmonkey-train tests/str.ini

# git clone https://github.com/tensorflow/models tests/tensorflow-models
# bin/neuralmonkey-train tests/captioning.ini

bin/neuralmonkey-train tests/flat-multiattention.ini
bin/neuralmonkey-train tests/hier-multiattention.ini

rm -rf tests/tmp-test-output
echo Tests OK.
