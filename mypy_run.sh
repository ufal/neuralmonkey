#!/bin/bash

mypy --py2 -s $(grep --include='*.py' -rl 'tests/..' -e "^# *tests:.*mypy")

if (( $? ))
then
	exit $?
else
	echo "Typecheck OK."
fi
