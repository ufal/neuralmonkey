#!/usr/bin/env python3

import sys
import tensorflow as tf





def main():

    if len(sys.args) < 3:
        print("Usage: neuralmonkey <ini_file> COMMAND [OPTION] ...")
        exit(1)
