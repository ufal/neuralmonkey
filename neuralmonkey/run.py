import sys
import os

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
CONFIG.add_argument('threads', required=False, default=4)
CONFIG.add_argument('runners_batch_size', required=False, default=None)
# ignore arguments which are just for training
CONFIG.ignore_argument('val_dataset')
CONFIG.ignore_argument('trainer')
CONFIG.ignore_argument('name')
CONFIG.ignore_argument('train_dataset')
CONFIG.ignore_argument('epochs')
CONFIG.ignore_argument('batch_size')
CONFIG.ignore_argument('test_datasets')
CONFIG.ignore_argument('initial_variables')
CONFIG.ignore_argument('validation_period')
CONFIG.ignore_argument('val_preview_input_series')
CONFIG.ignore_argument('val_preview_output_series')
CONFIG.ignore_argument('val_preview_num_examples')
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
    if len(sys.argv) != 3:
        print("Usage: run.py <run_ini_file> <test_datasets>")
        exit(1)

    test_datasets = Configuration()
    test_datasets.add_argument('test_datasets')
    test_datasets.add_argument('variables')

    CONFIG.load_file(sys.argv[1])
    CONFIG.build_model()
    test_datasets.load_file(sys.argv[2])
    test_datasets.build_model()
    datesets_model = test_datasets.model
    initialize_for_running(CONFIG.model.output, CONFIG.model.tf_manager,
                           datesets_model.variables)

    print("")

    evaluators = [(e[0], e[0], e[1]) if len(e) == 2 else e
                  for e in CONFIG.model.evaluation]

    for dataset in datesets_model.test_datasets:
        execution_results, output_data = run_on_dataset(
            CONFIG.model.tf_manager, CONFIG.model.runners,
            dataset, CONFIG.model.postprocess, write_out=True)
        # TODO what if there is no ground truth
        eval_result = evaluation(evaluators, dataset, CONFIG.model.runners,
                                 execution_results, output_data)
        if eval_result:
            print_final_evaluation(dataset.name, eval_result)
