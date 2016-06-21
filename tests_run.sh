#!/bin/bash

set -ex

for file in $(cd tests/python && echo *.py); do
	python -m tests.python.${file%.py}
done
