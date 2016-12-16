#!/bin/bash

pycodestyle $(find neuralmonkey -name '*.py')

EXIT_CODE=$?

if (( $EXIT_CODE )); then
    exit $EXIT_CODE
else
    echo Pycodestyle OK.
fi
