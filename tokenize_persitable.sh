#!/usr/bin/bash

LANGUAGE=$1
SOURCE=$2
TARGET=$3

CONTINUE=1

rm $TARGET
touch $TARGET

while [ $CONTINUE == 1 ]; do
    START=$(bc <<< "`wc -l < $TARGET` + 1")
    echo starting script on line $START of the corpus
    tail -n +$START $SOURCE | python -u tokenize_data.py --language=$LANGUAGE >> $TARGET
    CONTINUE=$?
done
