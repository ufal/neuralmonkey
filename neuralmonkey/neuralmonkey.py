#!/usr/bin/env python3

import sys

from neuralmonkey.logging import log
from neuralmonkey.entrypoints.entrypoint import EntryPoint

import neuralmonkey.config.parsing as parsing
from neuralmonkey.config.config_loader import build_object

def main():
    if len(sys.argv) < 3:
        print("Usage: neuralmonkey COMMAND <model-config> [OPTION] ...")
        exit(1)

    command = sys.argv[1]
    config_file = sys.argv[2]
    config_dicts = parsing.parse_file(config_file)

    log("INI file is parsed.")

    if command not in config_dicts:
        log("Unknown command: {}".format(command), color="red")
        exit(1)

    # build config from given entry point
    # - can omit some unused configurations - which we want!

    if "class" not in config_dicts[command]:
        log("Command {} has undefined 'class' attribute"
            .format(command), color="red")
        exit(1)

    if not issubclass(config_dicts[command]['class'], EntryPoint):
        log("Command {} must be an EntryPoint. Got: {} instead."
            .format(command, str(config_dicts[command]['class'])), color="red")
        exit(1)

    entrypoint = build_object("object:{}".format(command), config_dicts,
                              dict(), 0)
    entrypoint.execute(*sys.argv[3:])

if __name__ == '__main__':
    main()
