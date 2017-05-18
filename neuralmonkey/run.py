import os
import argparse

from neuralmonkey.logging import log, log_print
from neuralmonkey.config.configuration import Configuration
from neuralmonkey.learning_utils import (evaluation, run_on_dataset,
                                         print_final_evaluation)

CONFIG = Configuration()
CONFIG.add_argument('tf_manager')
CONFIG.add_argument('output')
CONFIG.add_argument('postprocess')
CONFIG.add_argument('evaluation')
CONFIG.add_argument('runners')
CONFIG.add_argument('batch_size')
CONFIG.add_argument('threads', required=False, default=4)
CONFIG.add_argument('runners_batch_size', required=False, default=None)
# ignore arguments which are just for training
CONFIG.ignore_argument('val_dataset')
CONFIG.ignore_argument('trainer')
CONFIG.ignore_argument('name')
CONFIG.ignore_argument('train_dataset')
CONFIG.ignore_argument('epochs')
CONFIG.ignore_argument('test_datasets')
CONFIG.ignore_argument('initial_variables')
CONFIG.ignore_argument('validation_period')
CONFIG.ignore_argument('val_preview_input_series')
CONFIG.ignore_argument('val_preview_output_series')
CONFIG.ignore_argument('val_preview_num_examples')
CONFIG.ignore_argument('val_separate_output')
CONFIG.ignore_argument('logging_period')
CONFIG.ignore_argument('minimize')
CONFIG.ignore_argument('random_seed')
CONFIG.ignore_argument('save_n_best')
CONFIG.ignore_argument('overwrite_output_dir')


def default_variable_file(output_dir):
    variables_file = os.path.join(output_dir, "variables.data")
    cont_index = 1

    def continuation_file():
        return os.path.join(output_dir,
                            "variables.data.cont-{}.best".format(cont_index))
    while os.path.exists(continuation_file()):
        variables_file = continuation_file()
        cont_index += 1

    return variables_file


def initialize_for_running(output_dir, tf_manager, variable_files) -> None:
    """Restore either default variables of from configuration.

    Arguments:
       output_dir: Training output directory.
       tf_manager: TensorFlow manager.
       variable_files: Files with variables to be restored or None if the
           default variables should be used.
    """
    # pylint: disable=no-member
    log_print("")

    if variable_files is None:
        default_varfile = default_variable_file(output_dir)

        log("Default variable file '{}' will be used for loading variables."
            .format(default_varfile))

        variable_files = [default_varfile]

    for vfile in variable_files:
        if not os.path.exists("{}.index".format(vfile)):
            log("Index file for var prefix {} does not exist".format(vfile),
                color="red")
            exit(1)

    tf_manager.restore(variable_files)

    log_print("")


def main() -> None:
    # pylint: disable=no-member,broad-except
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", metavar="INI-FILE",
                        help="the configuration file of the experiment")
    parser.add_argument('datasets', metavar='INI-TEST-DATASETS',
                        help="the configuration of the test datasets")
    parser.add_argument("-g", "--grid", dest="grid", action="store_true",
                        help="look at the SGE variables for slicing the data")
    args = parser.parse_args()

    test_datasets = Configuration()
    test_datasets.add_argument('test_datasets')
    test_datasets.add_argument('variables')

    CONFIG.load_file(args.config)
    CONFIG.build_model()
    test_datasets.load_file(args.datasets)
    test_datasets.build_model()
    datasets_model = test_datasets.model
    initialize_for_running(CONFIG.model.output, CONFIG.model.tf_manager,
                           datasets_model.variables)

    print("")

    evaluators = [(e[0], e[0], e[1]) if len(e) == 2 else e
                  for e in CONFIG.model.evaluation]

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

        if CONFIG.model.runners_batch_size is None:
            runners_batch_size = CONFIG.model.batch_size
        else:
            runners_batch_size = CONFIG.model.runners_batch_size

        execution_results, output_data = run_on_dataset(
            CONFIG.model.tf_manager, CONFIG.model.runners,
            dataset, CONFIG.model.postprocess, write_out=True,
            batch_size=runners_batch_size)
        # TODO what if there is no ground truth
        eval_result = evaluation(evaluators, dataset, CONFIG.model.runners,
                                 execution_results, output_data)
        if eval_result:
            print_final_evaluation(dataset.name, eval_result)
