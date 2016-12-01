#!/bin/bash

set -e

pylint -j 4 $(grep --include='*.py' --exclude='__init__.py' -rl neuralmonkey -e "^# *tests:.*lint")
