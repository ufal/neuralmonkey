#!/bin/bash

mypy --py2 -s \
       dataset.py \
       vocabulary.py \
       learning_utils.py \
       config_loader.py

if (( $? ))
then
	exit $?
else
	echo "Typecheck OK."
fi
