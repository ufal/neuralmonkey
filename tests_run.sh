#!/bin/bash

set -ex

for file in $(cd neuralmonkey/tests && echo *.py); do
	python -m neuralmonkey.tests.${file%.py}
done
