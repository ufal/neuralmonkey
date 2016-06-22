#!/bin/bash

if [ ! -f tests/train.tc.en ]; then
	cd tests
	wget http://ufallab.ms.mff.cuni.cz/~musil/{train,val}.tc.{en,de}
	cd ..
fi

python -m neuralmonkey.train tests/small.ini
