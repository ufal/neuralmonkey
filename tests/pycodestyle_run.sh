#!/bin/bash

set -e

pycodestyle $(find neuralmonkey -name '*.py')
