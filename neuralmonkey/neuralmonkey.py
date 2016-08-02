#!/usr/bin/env python3

import sys
import tensorflow as tf

from neuralmonkey.logging import log
from neuralmonkey.experiment import EntryPoint


def main():
    if len(sys.argv) < 3:
        print("Usage: neuralmonkey COMMAND <model-config> [OPTION] ...")
        exit(1)

    command = sys.argv[1]
    config_file = sys.argv[2]

    config = Configuration()
    args = config.load_file(config_file)

    if not hasattr(args, command):
        log("Unknown command: {}".format(command), color="red")
        exit(1)

    if args.command is not EntryPoint:
        log("Command {} must be an instance of EntryPoint class"
            .format(command), color="red")
        exit(1)



    args.command.execute(sys.argv[3:])
