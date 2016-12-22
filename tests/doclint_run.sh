#!/bin/bash

set -Ex
trap 'echo -e "\033[1;31mMarkdownlint spotted errors!\033[0m"' ERR

mdl *.md scripts examples tests docs neuralmonkey
