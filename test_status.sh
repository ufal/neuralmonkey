#!/bin/bash

echo "Mypy runs on:"
grep --include='*.py' -rl neuralmonkey -e "^# *tests:.*mypy"
echo ""

echo "Files with (some) type annotations:"
grep --include='*.py' -rl neuralmonkey -e "# *type:"
echo ""

echo "Mypy does not run on:"
grep --include='*.py' --exclude='__init__.py' -rL neuralmonkey -e "^# *tests:.*mypy"
echo ""
