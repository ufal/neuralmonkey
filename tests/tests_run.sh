#!/bin/bash

set -ex

if [ ! -f tests/data/train.tc.en ]; then
	wget -P tests/data http://ufallab.ms.mff.cuni.cz/~musil/{train,val}.tc.{en,de}
fi

bin/neuralmonkey train tests/vocab.ini
bin/neuralmonkey train tests/small.ini

mkdir -p tests/tmp-vars
mv tests/tmp-test-output/variables.data.cont-1.0 tests/tmp-vars/variables.data.0
mv tests/tmp-test-output/variables.data.cont-1.1 tests/tmp-vars/variables.data.1


bin/neuralmonkey run tests/small.ini tests/test_data.ini
bin/neuralmonkey run tests/ensemble.ini tests/test_data.ini

#bin/neuralmonkey-server --configuration=tests/small.ini --port=5000 &
#SERVER_PID=$!
#sleep 20

#curl 127.0.0.1:5000 -H "Content-Type: application/json" -X POST -d '{"source": ["I am the eggman.", "I am the walrus ."]}'
#kill $SERVER_PID
