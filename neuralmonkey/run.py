# pylint: disable=unused-import, wrong-import-order
import neuralmonkey.checkpython
# pylint: enable=unused-import, wrong-import-order

import argparse
import json
import os

from neuralmonkey.config.configuration import Configuration
from neuralmonkey.experiment import Experiment
from neuralmonkey.logging import log


def load_runtime_config(config_path: str) -> argparse.Namespace:
    """Load a runtime configuration file."""
    cfg = Configuration()
    cfg.add_argument("test_datasets")
    cfg.add_argument("variables", cond=lambda x: isinstance(x, list))

    cfg.load_file(config_path)
    cfg.build_model()
    return cfg.model


def main() -> None:
    # pylint: disable=no-member
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", metavar="INI-FILE",
                        help="the configuration file of the experiment")
    parser.add_argument("datasets", metavar="INI-TEST-DATASETS",
                        help="the configuration of the test datasets")
    parser.add_argument("--json", type=str, help="write the evaluation "
                        "results to this file in JSON format")
    parser.add_argument("-g", "--grid", dest="grid", action="store_true",
                        help="look at the SGE variables for slicing the data")
    args = parser.parse_args()

    datasets_model = load_runtime_config(args.datasets)

    exp = Experiment(config_path=args.config)
    exp.build_model()
    exp.load_variables(datasets_model.variables)

    if args.grid and len(datasets_model.test_datasets) > 1:
        raise ValueError("Only one test dataset supported when using --grid")

    results = []
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
            eval_result = exp.evaluate(dataset, write_out=True)
            results.append(eval_result)

    if args.json:
        with open(args.json, "w") as f_out:
            json.dump(results, f_out)
            f_out.write("\n")

    for session in exp.config.model.tf_manager.sessions:
        session.close()
