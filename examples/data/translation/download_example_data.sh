#!/bin/bash

for file in bpe_merges train.en train.de val.en val.de; do
    wget http://ufallab.ms.mff.cuni.cz/~helcl/neuralmonkey-example-data/translation/$file
done
