#!/bin/bash

set -e

mdl *.md $(find neuralmonkey -name '*.md') $(find tests -name '*.md') $(find examples -name '*.md')
