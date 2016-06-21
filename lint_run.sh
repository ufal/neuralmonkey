#!/bin/bash

pylint -j 4 $(grep --include='*.py' -rl . -e "^# *tests:.*lint")
