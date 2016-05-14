#!/bin/bash

./train_translation.py --train-source-sentences=tests/train.tc.en --val-source-sentences=tests/val.tc.en --train-target-sentences=tests/train.tc.de --val-target-sentences=tests/val.tc.de --dropout=0.5 --epochs=1 --target-german=True --test-run=True --batch-size=16 --maximum-output=10 $@
