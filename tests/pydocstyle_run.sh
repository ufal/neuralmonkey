#!/bin/bash

set -Ex
trap 'echo -e "\033[1;31mPydocstyle spotted errors!\033[0m"' ERR

# The basic ignore codes.
IGNORED="D202,D203,D213,D406,D407,D408,D409,D413"

# These are currently turned off on master branch
# because of the missing docstrings. However, they should
# be switched on in the future.
IGNORED="D100,D101,D102,D103,D104,D107,$IGNORED"

pydocstyle --ignore=$IGNORED neuralmonkey
