#!/bin/bash

set -e

pylint -j 4 $(find neuralmonkey -name '*.py')
