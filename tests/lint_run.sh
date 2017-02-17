#!/bin/bash

set -Ex
trap 'echo -e "\033[1;31mPylint spotted errors!\033[0m"' ERR

pylint --output-format=colorized -j 4 neuralmonkey
