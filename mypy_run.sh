#!/bin/bash

mypy --py2 -s $(grep --include='*.py' --exclude='__init__.py' -rl neuralmonkey -e "^# *tests:.*mypy")

if (( $? ))
then
	exit $?
else
	echo "Typecheck OK."
fi
