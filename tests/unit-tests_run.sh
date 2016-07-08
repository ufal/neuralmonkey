#!/bin/bash

set -ex

for file in $(cd neuralmonkey/tests && echo *.py); do
	python3 -m neuralmonkey.tests.${file%.py}
done
