#!/bin/bash

set -ex

if [ ! -f tests/data/train.tc.en ]; then
	wget -P tests/data http://ufallab.ms.mff.cuni.cz/~musil/{train,val}.tc.{en,de}
fi

bin/neuralmonkey run tests/vocab.ini

bin/neuralmonkey train tests/small.ini
bin/neuralmonkey run tests/small.ini tests/test_data.ini

#bin/neuralmonkey-server --configuration=tests/small.ini --port=5000 &
#SERVER_PID=$!
#sleep 20

#curl 127.0.0.1:5000 -H "Content-Type: application/json" -X POST -d '{"source": ["I am the eggman.", "I am the walrus ."]}'
#kill $SERVER_PID
