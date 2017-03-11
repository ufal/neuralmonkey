#!/bin/bash

for file in train.forms-cs train.tags-cs.subpos val.forms-cs val.tags-cs.subpos LICENSE; do
    wget http://ufallab.ms.mff.cuni.cz/~helcl/neuralmonkey-example-data/tagging/$file
done
