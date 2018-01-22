#!/bin/bash

set -Ex
trap 'echo -e "\033[1;31mPylint spotted errors!\033[0m"' ERR

pylint --output-format=colorized -j 4 --reports=no --load-plugins pylint_quotes,pylint.extensions.bad_builtin,pylint.extensions.check_elif,pylint.extensions.emptystring,pylint.extensions.redefined_variable_type neuralmonkey
