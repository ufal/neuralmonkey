#!/bin/bash

set -ex

tests/lint_run.sh
tests/mypy_run.sh
tests/doclint_run.sh
tests/unit-tests_run.sh
tests/tests_run.sh
