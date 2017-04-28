#!/bin/bash

set -ex

export NEURALMONKEY_STRICT=1
export PYTHONFAULTHANDLER=1

bin/neuralmonkey-train tests/vocab.ini
bin/neuralmonkey-train tests/bahdanau.ini
bin/neuralmonkey-train tests/bpe.ini
bin/neuralmonkey-train tests/alignment.ini
bin/neuralmonkey-train tests/post-edit.ini
bin/neuralmonkey-train tests/factored.ini
bin/neuralmonkey-train tests/classifier.ini
bin/neuralmonkey-train tests/labeler.ini
bin/neuralmonkey-train tests/language-model.ini
bin/neuralmonkey-train tests/audio-classifier.ini
bin/neuralmonkey-train tests/ctc.ini
bin/neuralmonkey-train tests/beamsearch.ini
bin/neuralmonkey-train tests/self-critical.ini

bin/neuralmonkey-train tests/small.ini
bin/neuralmonkey-train tests/small_sent_cnn.ini
bin/neuralmonkey-run tests/small.ini tests/test_data.ini
bin/neuralmonkey-server --configuration=tests/small.ini --port=5000 &
SERVER_PID=$!
sleep 20

curl 127.0.0.1:5000 -H "Content-Type: application/json" -X POST -d '{"source": ["I am the eggman.", "I am the walrus ."]}'
kill $SERVER_PID

bin/neuralmonkey-train tests/str.ini

rm -rf tests/tmp-test-output
echo Tests OK.
