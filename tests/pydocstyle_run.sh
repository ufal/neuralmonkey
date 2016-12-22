#!/bin/bash

set -Ex
trap 'echo -e "\033[1;31mPydocstyle spotted errors!\033[0m"' ERR

pydocstyle neuralmonkey
