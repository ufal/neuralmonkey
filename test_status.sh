#!/bin/bash

echo "Pylint runs on:"
grep --include='*.py' -rl . -e "^# *tests:.*lint"
echo ""

echo "Mypy runs on:"
grep --include='*.py' -rl . -e "^# *tests:.*mypy"
echo ""

echo "Files with (some) type annotations:"
grep --include='*.py' -rl . -e "# *type:"
echo ""

echo "Pylint does not run on:"
grep --include='*.py' -rL . -e "^# *tests:.*lint"
echo ""
