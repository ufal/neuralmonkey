import os
import codecs
import numpy as np
import tensorflow as tf

from neuralmonkey.logging import log, Logging, debug
from neuralmonkey.checking import check_dataset_and_model, CheckingException
from neuralmonkey.config.configuration import Configuration
from neuralmonkey.entrypoints.entrypoint import EntryPoint
from neuralmonkey.runners.ensemble_runner import EnsembleRunner

class Model(EntryPoint):
    """This class represents a sequence-to-sequence model.
    Main functionality of this class is to process datasets through the model.
    This class is also responsible for creating feed dictionaries for the model.

    The Model object can be also used as an entry point of a NeuralMonkey
    application. In this case, an argument specifying the datasets to be
    processed is expected.
    """
    def __init__(self, decoder, encoders, runner, postprocess=lambda x: x,
                 **kwargs):
        """ Creates a new instance of a Model.

        Arguments:
            decoder: Decoder for the model
            encoders: Encoder objects
            runner: A runner which will be used for computing the output
            postprocess: A function which will be called on the raw output

            tfconfig: A ConfigProto object to create the TensorFlow session.
        """
        super().__init__(kwargs.get("tfconfig"))

        self.decoder = decoder
        self.encoders = encoders # TODO resolve issue #27
        self.runner = runner
        self.postprocess = postprocess


    def feed_dicts(self, dataset, train=True):
        """This method gather feed dictionaries from all encoder and decoder
        objects.

        Arguments:
            dataset: Dataset for creating the feed dicts
            train: Boolean flag, True during training
        """
        #pylint: disable=invalid-name
        #fd is always for feed dicts.
        fd = {}
        for encoder in self.encoders:
            fd.update(encoder.feed_dict(dataset, train=train))

        fd.update(self.decoder.feed_dict(dataset, train=train))

        return fd


    def run_on_dataset(self, sessions, dataset, save_output=False):
        """Runs the model on a dataset, performs postprocessing

        Arguments:
            sess: TensorFlow session to use (stores model parameters)
            dataset: Dataset on which to run the model
            save_output: If True, output is saved to a file (specified in the
                         dataset ini config)

        Returns:
            Tuple of the postprocessed and raw outputs.
        """
        if isinstance(self.runner, EnsembleRunner):
            result_raw, opt_loss, dec_loss = self.runner(sessions, dataset,
                                                         self.feed_dicts)
        else:
            result_raw, opt_loss, dec_loss = self.runner(sessions[0], dataset,
                                                         self.feed_dicts)

        result = self.postprocess(result_raw)
        debug("Raw result: {}".format(result_raw[0]), "rawResults")

        if save_output:
            if self.decoder.data_id in dataset.series_outputs:
                path = dataset.series_outputs[self.decoder.data_id]

                if isinstance(result, np.ndarray):
                    np.save(path, result)
                    log("Result saved as numpy array to '{}'".format(path))
                else:
                    with codecs.open(path, 'w', 'utf-8') as f_out:
                        f_out.writelines([" ".join(sent)+"\n"
                                          for sent in result])
                    log("Result saved as plain text in '{}'".format(path))
            else:
                log("There is no output file for dataset: {}"
                    .format(dataset.name), color='red')

        return result, opt_loss, dec_loss



    def execute(self, *args):
        """Executes this model as an entry point

        Additional parameter in *args is required - path to config file for
        dataset.

        TODO solve how variables are gonna be passed to this entrypoint
        TODO solve evaluation. In this executable, no evaluation is done
        """
        if len(args) != 1:
            print("Command requires one additional argument"
                  " (run configuration)")
            exit(1)

        run_config = Configuration()
        run_config.add_argument("test_datasets")
        run_config.add_argument("variables")
        run_args = run_config.load_file(args[0])
        print("")

        #pylint: disable=no-member
        #these are added programmatically
        variables_files = run_args.variables
        datasets = run_args.test_datasets

        for variables_file in variables_files:
            if not os.path.exists(variables_file):
                log("Variables file does not exist: {}".format(variables_file),
                    color="red")
                exit(1)

        log("Initializing TensorFlow session(s).")
        sessions = [self.create_session_from_variables(v)
                    for v in variables_files]

        try:
            for dataset in datasets:
                check_dataset_and_model(dataset, self, test=True)
        except CheckingException as exc:
            log(str(exc), color="red")
            exit(1)

        for dataset in datasets:
            result, _, _ = self.run_on_dataset(sessions, dataset,
                                               save_output=True)
            # TODO if we have reference, show also reference
            Logging.show_sample(result, randomized=True)
