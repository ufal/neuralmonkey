#!/bin/bash

set -e

pylint $(find neuralmonkey -name '*.py')
