#!/bin/bash

set -ex
trap 'echo -e "\033[1;31mSome unit tests have failed!\033[0m"' ERR

python3 -m unittest discover -v
