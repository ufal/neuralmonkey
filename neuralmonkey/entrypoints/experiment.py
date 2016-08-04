#tests: lint

import os

import tensorflow as tf
import numpy as np

from termcolor import colored

from neuralmonkey.output_dir import OutputDirectory
from neuralmonkey.logging import Logging, log, log_print, debug
from neuralmonkey.checking import check_dataset_and_model
from neuralmonkey.entrypoints.entrypoint import EntryPoint
from neuralmonkey.exceptions import CheckingException


def format_eval_name(name):
    if hasattr(name, '__call__'):
        return name.__name__
    else:
        return str(name)


class TrainingProgress(object):
    # pylint: disable=too-few-public-methods
    # TODO refactor with something more pythonic
    def __init__(self, minimize, n_best):
        self.best_score = np.inf if minimize else -np.inf
        self.best_score_epoch = 0
        self.best_score_batch = 0
        self.saved_scores = [self.best_score for _ in range(n_best)]
        self.best_saved_index = 0


class Experiment(EntryPoint):
    # pylint: disable=too-many-instance-attributes
    # Big object needs many attributes. Division would do more harm than good.

    def __init__(self, model, trainer, train_dataset, val_dataset,
                 evaluation, output, **kwargs):
        # pylint: disable=too-many-arguments
        # Just as many arguments as is necessary.
        super().__init__(kwargs.get("tfconfig"))

        self.name = kwargs.get("name", "translation")
        self.model = model

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_datasets = kwargs.get("test_datasets", [])
        self.check_data()

        self.trainer = trainer
        self.epochs = kwargs.get("epochs", 10)
        self.batch_size = kwargs.get("batch_size", 64)
        self.random_seed = kwargs.get("random_seed", None)

        self.evaluators = evaluation
        self.minimize = kwargs.get("minimize", False)
        self.initial_variables = kwargs.get("initial_variables", None)
        self.save_n_best_vars = kwargs.get("save_n_best", 1)

        overwrite = kwargs.get("overwrite_output_dir", False)
        self.output = OutputDirectory(output, overwrite)

        self.logging_period = kwargs.get("logging_period", 20)
        self.validation_period = kwargs.get("validation_period", 500)


    def check_data(self):
        try:
            check_dataset_and_model(self.train_dataset, self.model)
            check_dataset_and_model(self.val_dataset, self.model)

            for test in self.test_datasets:
                check_dataset_and_model(test, self.model, test=True)

        except CheckingException as exc:
            log(str(exc), color='red')
            exit(1)


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
        # TODO proc to tady bylo?
        #target_sentences = batch_dataset.get_series(self.model.decoder.data_id)

        return self.trainer.run(session, feed_dict, summary)



    def evaluate(self, session, dataset, save_output=False):
        """ Run model on a dataset, return evaluation and decoded data

        Argumetns:
            session: TF Session
            dataset: Dataset to evaluate on
            save_output: If true, decoded output is saved

        Returns: tuple of evaluation (dict of evaluation metrics)
                 and the postprocessed decoded data
        """
        decoded, opt_loss, dec_loss = self.model.run_on_dataset(
            session, dataset, save_output=save_output)

        evaluation = {}
        if dataset.has_series(self.model.decoder.data_id):
            ## TODO is there a case when this is not true?

            test_targets = dataset.get_series(self.model.decoder.data_id)
            evaluation["opt_loss"] = opt_loss
            evaluation["dec_loss"] = dec_loss

            for func in self.evaluators:
                evaluation[func] = func(decoded, test_targets)

        return evaluation, decoded


    def log_evaluation(self, evaluation, validation=False):
        """ Logs the evaluation results

        Arguments:
            evaluation: Dictionary with evaluation results
            validation: If True, switch on validation logging mode
        """
        eval_string = "    ".join(
            ["{}: {:.2f}".format(f.name, evaluation[f.name])
             for f in self.evaluators[:-1]])

        if len(self.evaluators) >= 1:
            main_evaluator = self.evaluators[-1]
            eval_string += colored("    {}: {:.2f}"
                                   .format(format_eval_name(main_evaluator),
                                           evaluation[main_evaluator]),
                                   attrs=['bold'])

        log("opt. loss: {:.4f}    dec. loss: {:.4f}    {}"
            .format(evaluation['opt_loss'], evaluation['dec_loss'],
                    eval_string),
            color="blue" if validation else "yellow")



    def write_summaries(self, seen_instances, evaluation, tb_writer,
                        validation=False):
        """ Writes TensorBoard summaries

        Arguments:
            seen_instances: Number of processed training examples
            evaluation: Dictionary with evaluation results
            tb_writer: TensorBoard writer object
            validation: If True, summaries are written with validation tags
        """
        #pylint: disable=no-self-use
        # TODO assess and possibly refactor this one
        tagvals = [("{}_{}".format("val" if validation else "train",
                                   format_eval_name(n)), val)
                   for n, val in evaluation.items()]

        #tb_writer.add_summary(summary_str, seen_instances)

        # TODO DO THIS !! HISTOGRAMS !!
        #if histograms_str:
        #    tb_writer.add_summary(histograms_str, seen_instances)

        external_str = tf.Summary(value=[tf.Summary.Value(tag=t, simple_value=v)
                                         for t, v in tagvals])
        tb_writer.add_summary(external_str, seen_instances)



    def is_better_score(self, score1, score2):
        """ Checks whether a score is better than another one.

        Arguments:
            score1: The first score
            score2: The second score

        Returns: True if the first score is better then the second one,
                 False otherwise.
        """
        if self.minimize:
            return score1 < score2
        else:
            return score1 > score2


    def argworst_score(self, scores):
        """ Gets the index of the worst score in array

        Arguments:
            scores: An array of scores

        Returns: Index to 'scores', pointing to the worst score in the array
        """
        if self.minimize:
            return np.argmax(scores)
        else:
            return np.argmin(scores)



    def run_training_loop(self, session, saver, progress):
        """ Executes the training loop on the model.

        Note: This method should not return anything since it should expect to
        be interrupted by the user.

        Arguments:
            session: TF Session to use
            saver: TF Saver instance used to store scores
            progress: Object that ensures persistence of results when this
                      function is interrupted
        """
        #pylint: disable=too-many-locals,too-many-nested-blocks
        # Not sure if this deserves to be divided to more functions or not

        training_step = 0
        training_seen_instances = 0

        log("Initializing TensorBoard summary writer.")
        tb_writer = tf.train.SummaryWriter(self.output.path, session.graph)
        log("TesorBoard writer initialized.")

        for epoch in range(1, self.epochs + 1):
            log_print("")
            log("Epoch {} starts".format(epoch), color="red")

            self.train_dataset.shuffle()
            train_batched_datasets = self.train_dataset.batch_dataset(
                self.batch_size)

            for batch_number, batch_dataset in enumerate(
                    train_batched_datasets):
                training_step += 1
                training_seen_instances += len(batch_dataset)

                self.run_batch(session, batch_dataset)

                if (training_step + 1) % self.logging_period == 0:
                    evaluation, _ = self.evaluate(session, batch_dataset)
                    self.log_evaluation(evaluation)
                    self.write_summaries(training_seen_instances,
                                         evaluation, tb_writer)

                if (training_step + 1) % self.validation_period == 0:
                    evaluation, decoded = self.evaluate(
                        session, self.val_dataset)

                    score = evaluation[self.evaluators[-1]]

                    worst_index = self.argworst_score(progress.saved_scores)
                    worst_score = progress.saved_scores[worst_index]

                    if self.is_better_score(score, worst_score):
                        # Replace the variables in worst saved model with these.
                        var_file = self.output.get_variable_file(worst_index)
                        saver.save(session, var_file)
                        progress.saved_scores[worst_index] = score
                        log("Variable file saved in {}".format(var_file),
                            color="blue")

                        # This could also be one tab to the left
                        if self.is_better_score(score, progress.best_score):
                            progress.best_score = score
                            progress.best_score_epoch = epoch
                            progress.best_score_batch = batch_number + 1

                            # Update symlink to best variables
                            if os.path.islink(self.output.link_best_vars):
                                os.unlink(self.output.link_best_vars)
                            os.symlink(os.path.basename(var_file),
                                       self.output.link_best_vars)

                        log("Best scores saved so far: {}".
                            format(progress.saved_scores), color="blue")

                    log("Validation (epoch {}, batch_number {}):"
                        .format(epoch, batch_number + 1), color="blue")

                    self.log_evaluation(evaluation, validation=True)
                    self.write_summaries(training_seen_instances,
                                         evaluation, tb_writer, validation=True)

                    if score == progress.best_score:
                        best_score_str = colored(
                            "{:.2f}".format(progress.best_score),
                            attrs=['bold'])
                    else:
                        best_score_str = "{:.2f}".format(progress.best_score)

                    log("best {} on validation: {} (in epoch {}, "
                        "after batch number {})"
                        .format(format_eval_name(self.evaluators[-1]),
                                best_score_str,
                                progress.best_score_epoch,
                                progress.best_score_batch),
                        color="blue")

                    Logging.show_sample(decoded, self.val_dataset.get_series(
                        self.model.decoder.data_id))


    def execute(self, *args):
        if self.random_seed is not None:
            tf.set_random_seed(self.random_seed)

        self.output.populate()
        Logging.print_header(self.name)

        session = self.create_session()
        debug("Session: {}".format(session))
        saver = tf.train.Saver(max_to_keep=self.save_n_best_vars)

        if self.initial_variables:
            log("Loading variables from {}".format(self.initial_variables))
            saver.restore(session, self.initial_variables)

        log("Session initialization done")

        training_progress = TrainingProgress(self.minimize,
                                             self.save_n_best_vars)

        try:
            self.run_training_loop(session, saver, training_progress)
        except KeyboardInterrupt:
            log("Training interrupted by user.")

        if os.path.islink(self.output.link_best_vars):
            saver.restore(session, self.output.link_best_vars)

        log("Training finished. Maximum {} on validation data: {:.2f}, epoch {}"
            .format(format_eval_name(self.evaluators[-1]),
                    training_progress.best_score,
                    training_progress.best_score_epoch))

        for dataset in self.test_datasets:
            evaluation, _ = self.evaluate(session, dataset, save_output=True)

            log("Evaluating model on '{}'".format(dataset.name))
            for func in evaluation:
                name = format_eval_name(func)
                log("... {}:{:>25.4f}".format(name, evaluation[func]))

            log_print("")

        log("Finished.")
