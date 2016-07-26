#!/usr/bin/env python3

import sys
import tensorflow as tf

from neuralmonkey.logging import log



def main():

    if len(sys.argv) < 3:
        print("Usage: neuralmonkey <ini_file> COMMAND [OPTION] ...")
        exit(1)


    config_file = sys.argv[1]
    command = sys.argv[2]

    config = Configuration()
    args = config.load_file(config_file)

    if not hasattr(args, command):
        log("Unknown command: {}".format(command), color="red")
        exit(1)

    if args
