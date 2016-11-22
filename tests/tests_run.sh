#!/bin/bash

set -ex

bin/neuralmonkey-train tests/vocab.ini
bin/neuralmonkey-train tests/bahdanau.ini

bin/neuralmonkey-train tests/small.ini
bin/neuralmonkey-run tests/small.ini tests/test_data.ini
bin/neuralmonkey-server --configuration=tests/small.ini --port=5000 &
SERVER_PID=$!
sleep 20

curl 127.0.0.1:5000 -H "Content-Type: application/json" -X POST -d '{"source": ["I am the eggman.", "I am the walrus ."]}'
kill $SERVER_PID

rm -rf tests/tmp-test-output
