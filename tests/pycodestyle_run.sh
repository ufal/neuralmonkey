#!/bin/bash

set -e

pycodestyle -j 4 $(find neuralmonkey -name '*.py')
