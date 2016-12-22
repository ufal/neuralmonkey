#!/bin/bash

set -Ex
trap 'echo -e "\033[1;31mMypy spotted errors!\033[0m"' ERR

mypy -s $(grep --include='*.py' --exclude='__init__.py' -rl neuralmonkey -e "^# *tests:.*mypy")
#mypy -s -p neuralmonkey
