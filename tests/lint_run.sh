#!/bin/bash

set -e

pylint -j 4 $(find neuralmonkey -name '*.py')

pycodestyle $(find neuralmonkey -name '*.py')
