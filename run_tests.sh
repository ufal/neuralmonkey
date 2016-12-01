#!/bin/bash

set -ex

tests/lint_run.sh
tests/mypy_run.sh
tests/mdl_run.sh
tests/unit-tests_run.sh
tests/tests_run.sh
