#!/bin/bash

./train.py ./tests/small.ini && rm -rf test_configuration
./train.py ./tests/small-beam.ini && rm -rf test_configuration
