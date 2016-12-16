#!/bin/bash

pylint -j 4 $(find neuralmonkey -name '*.py')

EXIT_CODE=$?

if (( $EXIT_CODE )); then
    echo Pylint spotted errors!
    exit $EXIT_CODE
else
    echo Pylint OK.
fi
