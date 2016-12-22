#!/bin/bash

set -Ex
trap 'echo -e "\033[1;31mPycodestyle spotted errors!\033[0m"' ERR

pycodestyle $(find neuralmonkey -name '*.py')
