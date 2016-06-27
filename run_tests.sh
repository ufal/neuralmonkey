#!/bin/bash

set -ex

./lint_run.sh
./mypy_run.sh
./tests_run.sh
