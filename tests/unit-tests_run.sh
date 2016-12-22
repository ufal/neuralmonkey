#!/bin/bash

set -Ex
trap 'echo -e "\033[1;31mSome unit tests have failed!\033[0m"' ERR

for file in $(cd neuralmonkey/tests && echo *.py); do
    python3 -m neuralmonkey.tests.${file%.py}
done
