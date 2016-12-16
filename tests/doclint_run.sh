#!/bin/bash

mdl *.md $(find neuralmonkey -name '*.md') $(find tests -name '*.md') $(find examples -name '*.md')

EXIT_CODE=$?

if (( $EXIT_CODE )); then
    exit $EXIT_CODE
else
    echo Markdownlint OK.
fi
