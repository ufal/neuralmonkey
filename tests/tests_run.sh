#!/bin/bash

cd ..
if [ ! -f tests/data/train.tc.en ]; then
	wget -P tests/data http://ufallab.ms.mff.cuni.cz/~musil/{train,val}.tc.{en,de}
fi

bin/neuralmonkey-train tests/small.ini
