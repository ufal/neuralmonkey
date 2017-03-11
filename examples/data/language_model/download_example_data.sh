#!/bin/bash

for file in train val test LICENSE; do
    wget http://ufallab.ms.mff.cuni.cz/~helcl/neuralmonkey-example-data/language_model/$file
done
