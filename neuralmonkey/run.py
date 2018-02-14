import os
import argparse

from neuralmonkey.config.configuration import Configuration
from neuralmonkey.experiment import Experiment
from neuralmonkey.logging import log


def main() -> None:
    # pylint: disable=no-member,broad-except
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", metavar="INI-FILE",
                        help="the configuration file of the experiment")
    parser.add_argument("datasets", metavar="INI-TEST-DATASETS",
                        help="the configuration of the test datasets")
    parser.add_argument("-g", "--grid", dest="grid", action="store_true",
                        help="look at the SGE variables for slicing the data")
    parser.add_argument("-s", "--set", type=str, metavar="SETTING",
                        action="append", dest="config_changes", default=[],
                        help="override an option in the configuration; the "
                        "syntax is [section.]option=value")
    parser.add_argument("-v", "--var", type=str, metavar="VAR", default=[],
                        action="append", dest="config_vars",
                        help="set a variable in the configuration; the syntax "
                        "is var=value (shorthand for -s vars.var=value)")
    args = parser.parse_args()
    args.config_changes.extend("vars.{}".format(s) for s in args.config_vars)

    test_datasets = Configuration()
    test_datasets.add_argument("test_datasets")
    test_datasets.add_argument("variables", cond=lambda x: isinstance(x, list))

    test_datasets.load_file(args.datasets)
    test_datasets.build_model()
    datasets_model = test_datasets.model

    exp = Experiment(config_path=args.config,
                     config_changes=args.config_changes)
    exp.build_model()
    exp.load_variables(datasets_model.variables)

    if args.grid and len(datasets_model.test_datasets) > 1:
        raise ValueError("Only one test dataset supported when using --grid")

    for dataset in datasets_model.test_datasets:
        if args.grid:
            if ("SGE_TASK_FIRST" not in os.environ
                    or "SGE_TASK_LAST" not in os.environ
                    or "SGE_TASK_STEPSIZE" not in os.environ
                    or "SGE_TASK_ID" not in os.environ):
                raise EnvironmentError(
                    "Some SGE environment variables are missing")

            length = int(os.environ["SGE_TASK_STEPSIZE"])
            start = int(os.environ["SGE_TASK_ID"]) - 1
            end = int(os.environ["SGE_TASK_LAST"]) - 1

            if start + length > end:
                length = end - start + 1

            log("Running grid task {} starting at {} with step {}"
                .format(start // length, start, length))

            dataset = dataset.subset(start, length)

        if exp.config.args.evaluation is None:
            exp.run_model(dataset, write_out=True)
        else:
            exp.evaluate(dataset, write_out=True)

    for session in exp.config.model.tf_manager.sessions:
        session.close()
