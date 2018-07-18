"""Training script for sequence to sequence learning."""

# pylint: disable=unused-import, wrong-import-order
import neuralmonkey.checkpython
# pylint: enable=unused-import, wrong-import-order

import argparse
import os
import shlex
from shutil import copyfile
import sys
import traceback

from neuralmonkey.logging import log, debug
from neuralmonkey.experiment import Experiment


# pylint: disable=too-many-statements, too-many-locals, too-many-branches
def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", metavar="INI-FILE",
                        help="the configuration file for the experiment")
    parser.add_argument("-s", "--set", type=str, metavar="SETTING",
                        action="append", dest="config_changes", default=[],
                        help="override an option in the configuration; the "
                        "syntax is [section.]option=value")
    parser.add_argument("-v", "--var", type=str, metavar="VAR", default=[],
                        action="append", dest="config_vars",
                        help="set a variable in the configuration; the syntax "
                        "is var=value (shorthand for -s vars.var=value)")
    parser.add_argument("-i", "--init", dest="init_only", action="store_true",
                        help="initialize the experiment directory and exit "
                        "without building the model")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="force overwriting the output directory; can be "
                        "used to start an experiment created with --init")
    args = parser.parse_args()

    args.config_changes.extend("vars.{}".format(s) for s in args.config_vars)

    exp = Experiment(config_path=args.config,
                     config_changes=args.config_changes,
                     train_mode=True,
                     overwrite_output_dir=args.overwrite)

    with open(exp.get_path("args", exp.cont_index + 1), "w") as file:
        print(" ".join(shlex.quote(a) for a in sys.argv), file=file)

    if args.init_only:
        if exp.cont_index >= 0:
            log("The experiment directory already exists.", color="red")
            exit(1)

        exp.config.save_file(exp.get_path("experiment.ini", 0))
        copyfile(args.config, exp.get_path("original.ini", 0))

        log("Experiment directory initialized.")

        cmd = [os.path.basename(sys.argv[0]), "-f",
               exp.get_path("experiment.ini", 0)]
        log("To start experiment, run: {}".format(" ".join(shlex.quote(a)
                                                           for a in cmd)))
        exit(0)

    try:
        exp.train()
    except KeyboardInterrupt:  # pylint: disable=try-except-raise
        raise
    except Exception:  # pylint: disable=broad-except
        log(traceback.format_exc(), color="red")
        exit(1)


def main() -> None:
    try:
        _main()
    except KeyboardInterrupt:
        log("Training interrupted by user.")
        debug(traceback.format_exc())
        exit(1)
