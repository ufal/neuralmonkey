import os
from shutil import copyfile

import tensorflow as tf

from neuralmonkey.logging import Logging, log, log_print, debug
from neuralmonkey.checking import check_dataset_and_model
from neuralmonkey.config.configuration import Configuration
from neuralmonkey.saving import Saving

def initialize_tf(initial_variables, threads):
    if initial_variables:
        log("Loading variables from {}".format(initial_variables))
        saver.restore(sess, initial_variables)

    log("Session initialization done.")
    return sess, saver




class EntryPoint(object):

    def __init__(self, tfconfig):
        self.tfconfig = tfconfig


    def execute(self, *args):
        """ Execute the entry point

        Arguments:
            args: additional arguments from command line
        """
        raise Exception("Cannot call directly - abstract method")


    def create_session(self, random_init=True):
        log("Initializing the TensorFlow session.")
        session = tf.Session(config=self.tfconfig)

        if random_init:
            session.run(tf.initialize_all_variables())
            log("Random variable initialization done.")


    def create_session_from_variables(self, variables_file):
        session = self.create_session()

        log("Loading variables from file {}.".format(variables_file))
        saver = tf.train.Saver()
        saver.restore(variables_file)
        log("Variables restored.")

        return session




class Model(EntryPoint):
    """ This class represents a sequence-to-sequence model.
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
        """ This method gather feed dictionaries from all encoder and decoder
        objects.

        Arguments:
            dataset: Dataset for creating the feed dicts
            train: Boolean flag, True during training
        """
        fd = {}
        for encoder in self.encoders:
            fd.update(encoder.feed_dict(dataset, train=train))

        fd.update(self.decoder.feed_dict(dataset, train=train))

        return fd


    def run_on_dataset(self, sess, dataset, save_output=False):
        """ Runs the model on a dataset, performs postprocessing

        Arguments:
            sess: TensorFlow session to use (stores model parameters)
            dataset: Dataset on which to run the model
            save_output: If True, output is saved to a file (specified in the
                         dataset ini config)

        Returns:
            Tuple of the postprocessed and raw outputs.
        """
        result_raw, opt_loss, dec_loss = self.runner(sess, dataset,
                                                     self.feed_dicts)
        result = self.postprocess(result_raw)

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

        ## evaluation will be done elsewhere
        return result, result_raw



    def execute(self, *args):
        """ Executes this model as an entry point

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

        variables_file = run_args.variables
        datasets = run_args.test_datasets

        if not os.path.exists(variables_file):
            log("Variables file does not exist: {}".format(variables_file),
                color="red")
            exit(1)

        log("Initializing TensorFlow session.")
        sess = self.create_session()

        log("Loading variables from {}".format(variables_file))
        saver = tf.train.Saver()
        saver.restore(sess, variables_file)
        print("")

        try:
            for dataset in datasets:
                check_dataset_and_model(dataset, self, test=True)
        except Exception as exc:
            log(exc.message, color="red")
            exit(1)

        for dataset in datasets:
            result, result_raw = self.run_on_dataset(sess, dataset,
                                                     save_output=True)
            # TODO if we have reference, show also reference
            Logging.show_sample(result, randomized=True)









class Experiment(EntryPoint):

    def __init__(self, model, trainer, train_dataset, val_dataset,
                 evaluators, output, **kwargs):
        super().__init__(kwargs.get("tfconfig"))

        self.model = model
        self.trainer = trainer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.evaluators = evaluators
        self.output_dir = output

        self.epochs = kwargs.get("epochs", 10)
        self.batch_size = kwargs.get("batch_size", 64)
        self.logging_period = kwargs.get("logging_period", 20)
        self.validation_period = kwargs.get("validation_period", 500)
        self.initial_variables = kwargs.get("initial_variables", None)
        self.save_n_best_vars = kwargs.get("save_n_best", 1)
        self.overwrite_output_dir = kwargs.get("overwrite_output_dir", False)
        self.name = kwargs.get("name", "translation")
        self.random_seed = kwargs.get("random_seed", None)
        self.test_datasets = kwargs.get("test_datasets", [])
        self.minimize = kwargs.get("minimize", False)

        self.training_step = 0
        self.training_seen_instances = 0

        if minimize:
            self.saved_scores = [np.inf for _ in range(save_n_best_vars)]
            self.best_score = np.inf
        else:
            self.saved_scores = [-np.inf for _ in range(save_n_best_vars)]
            self.best_score = -np.inf

        self.best_score_epoch = 0
        self.best_score_batch_no = 0


        ### set up variable files ##### TODOODODODO
        variables_file = "/tmp/variables.data"
        variables_link = "/tmp/variables.data.best"

        self.saver = Saving(variables_file, variables_link,
                            self.save_n_best_vars)




    def log_evaluation(self, train=False):
        """ Logs the evaluation results and writes summaries to TensorBoard """
        raise Exception("niy")


    def run_epoch(self, session):
        """ Runs one epoch of the experiment

        First, this method shuffles the training dataset and splis it into
        mini batches. Then, on each batch, it evaluates the computation graph
        and possibly log the score on the batch or run the evaluation.
        """
        self.train_dataset.shuffle()
        train_batched_datasets = self.train_dataset.batch_dataset(
            self.batch_size)

        for batch_number, batch_dataset in enumerate(train_batched_datasets):
            self.training_step += 1

            if (self.training_step + 1) % self.logging_period == 0:
                summary = self.run_batch(batch_dataset, summary=True)

                _, _, train_evaluation = self.model.run_on_dataset(
                    session, batch_dataset, write_out=False)

                ## process evaluation

            else:
                self.run_batch(session, batch_dataset)

            if (self.training_step + 1) % self.validation_period == 0:
                self.validate()


    def run_batch(self, session, batch_dataset, summary=False):
        """ Runs one batch of training throught the computation graph

        This method creates feed dictionaries, gets target sentences for
        loss computation and run the trainer.

        Arguments:
            session: TF Session
            batch_dataset: the dataset
            summary: whether or not to create summaries (for tensorboard)
        """
        feed_dict = self.model.feed_dicts(batch_dataset, train=True)
        target_sentences = batch_dataset.get_series(self.model.decoder.data_id)

        self.training_seen_instances += len(target_sentences)

        return self.trainer.run(session, feed_dict, summary)



    def validate(self, session):

        """
        run model on validation data, get score
        save if high score
        hooray if best score - symlink and stuff
        log

        Arguments:
            session: TF Session
        """

        decoded, decoded_raw, val_evaluation = self.model.run_on_dataset(
            session, self.val_dataset)

        score = val_evaluation[self.evaluators[-1].name]

        def is_better(score1, score2, minimize):
            if minimize:
                return score1 < score2
            else:
                return score1 > score2

        def argworst(scores, minimize):
            if minimize:
                return np.argmax(scores)
            else:
                return np.argmin(scores)

        if is_better(score, self.best_score, self.minimize):
            self.best_score = this_score
            self.best_score_epoch = i + 1
            self.best_score_batch_no = batch_n

        worst_index = argworst(self.saved_scores, self.minimize)
        worst_score = self.saved_scores[worst_index]

        if is_better(score, worst_score, self.minimize):
            worst_var_file = variables_files[worst_index]
            saver.save(sess, worst_var_file)
            saved_scores[worst_index] = this_score
            log("Variable file saved in {}".format(worst_var_file))




        raise Exception("niy")




    def execute(self, *args):
        if self.random_seed is not None:
            tf.set_random_seed(self.random_seed)

        try:
            check_dataset_and_model(self.train_dataset, self.model)
            check_dataset_and_model(self.val_dataset, self.model)

            for test in self.test_datasets:
                check_dataset_and_model(self.test, self.model, test=True)

        except Exception as exc:
            log(exc.message, color='red')
            exit(1)

        self.output = OutputDirectory(self.output_dir)

        ## sess, saver = initialize_tf(self.initial_variables, self.threads)
        session = self.create_session()

        try:
            for epoch in range(1, self.epochs + 1):
                log_print("")
                log("Epoch {} starts".format(epoch), color="red")

                self.run_epoch()

        except KeyboardInterrupt:
            log("Training interrupted by user.")

        ## restore best (if link exists)

        log("Training finished. Maximum {} on validation data: {:.2f}, epoch {}"
            .format(evaluation_labels[-1], best_score, best_score_epoch))













class OutputDirectory(object):

    def __init__(path, overwrite=False):
        self.path = path
        self.overwrite = overwrite

        ## TODO nicer exceptions

        if os.path.isdir(path) and os.path.exists(self.ini_file):
            if overwrite:
                log("Experiment directory '{}' exists, "
                    "overwriting enabled, proceeding."
                    .format(path))
            else:
                log("Experiment directory '{}' exists, "
                    "overwriting disabled."
                    .format(path), color="red")
                raise Exception("Cannot create output dir")

        if not os.path.isdir(path):
            try:
                os.mkdir(path)
            except Exception as exc:
                log("Failed to create experiment directory: {}. Exception: {}"
                    .format(path, exc), color="red")
                raise Exception("Cannot create output dir")

        self.cont_index = 0

        while (os.path.exists(self.log_file)
               or os.path.exists(self.ini_file)
               or os.path.exists(self.commit_file)
               or os.path.exists(self.diff_file)
               or os.path.exists(self.var_prefix)
               or os.path.exists("{}.0".format(self.var_prefix))):
            self.cont_index += 1

        copyfile(sys.argv[1], ini_file)
        Logging.set_log_file(log_file)
        Logging.print_header(args.name)

        os.system("git log -1 --format=%H > {}".format(git_commit_file))
        os.system("git --no-pager diff --color=always > {}"
                  .format(git_diff_file))





    def get_filename(filename):
        if self.cont_index == 0:
            return os.path.join(self.path, filename)
        else:
            return os.path.join(self.path,
                                "{}.cont-{}".format(filename, self.cont_index))



    @property
    def ini_file(self):
        return self.get_filename("experiment.ini")

    @property
    def log_file(self):
        return self.get_filename("experiment.log")

    @property
    def commit_file(self):
        return self.get_filename("git_commit")

    @property
    def diff_file(self):
        return self.get_filename("git_diff")

    @property
    def var_prefix(self):
        return self.get_filename("variables.data")

    @property
    def link_best_vars(self):
        return "{}.best".format(self.var_prefix)
