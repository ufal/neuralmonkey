#!/bin/bash

set -e

pydocstyle $(find neuralmonkey -name '*.py')
