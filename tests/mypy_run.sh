#!/bin/bash

mypy -s $(grep --include='*.py' --exclude='__init__.py' -rl neuralmonkey -e "^# *tests:.*mypy")
#mypy -s $(find neuralmonkey -name '*.py')

r=$?

if (( $r ))
then
	exit $r
else
	echo "Typecheck OK."
fi
