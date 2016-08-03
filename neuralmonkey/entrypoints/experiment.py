import os

import tensorflow as tf

from neuralmonkey.output_dir import OutputDirectory
from neuralmonkey.logging import Logging, log, log_print, debug
from neuralmonkey.checking import check_dataset_and_model
from neuralmonkey.saving import Saving
from neuralmonkey.entrypoints.entrypoint import EntryPoint


class Experiment(EntryPoint):

    def __init__(self, model, trainer, train_dataset, val_dataset,
                 evaluators, output, **kwargs):
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

        self.evaluators = evaluators
        self.minimize = kwargs.get("minimize", False)
        self.initial_variables = kwargs.get("initial_variables", None)
        self.save_n_best_vars = kwargs.get("save_n_best", 1)

        self.output_dir = output
        self.overwrite_output_dir = kwargs.get("overwrite_output_dir", False)

        self.logging_period = kwargs.get("logging_period", 20)
        self.validation_period = kwargs.get("validation_period", 500)

        if minimize:
            self.saved_scores = [np.inf for _ in range(save_n_best_vars)]
            self.best_score = np.inf
        else:
            self.saved_scores = [-np.inf for _ in range(save_n_best_vars)]
            self.best_score = -np.inf

        self.best_score_epoch = 0
        self.best_score_batch_no = 0

        self.training_step = 0
        self.training_seen_instances = 0

        ### set up variable files ##### TODOODODODO
        variables_file = "/tmp/variables.data"
        variables_link = "/tmp/variables.data.best"

        self.saver = Saving(variables_file, variables_link,
                            self.save_n_best_vars)







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





    def validate(self, session, batch_number):

        """
        run model on validation data, get score
        save if high score
        hooray if best score - symlink and stuff
        log

        Arguments:
            session: TF Session
        """


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

            # update symlink
            if best_score == score:
                os.unlink(link_best_vars)
                os.symlink(os.path.basename(words_var_file), link_best_vars)

            log("Best scores saved so far: {}".format(saved_scores))


        log("Validation (epoch {}, batch_number {}):".
            format(i + 1, batch_number), color="blue")

        ## process_evaluation

        if best_score == score:
            best_score_str = colored("{:.2f}".format(best_score),
                                     attrs=['bold'])
        else:
            best_score_str = "{:.2f}".format(best_score)

        log("best {} on validation: {} (in epoch {}, after batch number {})"
            .format(evaluation_labels[-1], best_score_str,
                    best_score_epoch, best_score_batch_no),
            color="blue")

        Logging.show_sample(decoded, reference)





    def evaluate(self, session, dataset, validation=False):
        """ Run model on a dataset, evaluate"""

        decoded, _, opt_loss, dec_loss = self.model.run_on_dataset(
            session, dataset, save_output=False)

        evaluation = {}
        if dataset.has_series(self.model.decoder.data_id): ## TODO is there a case when this is not true?
            test_targets = dataset.get_series(self.model.decoder.data_id)
            evaluation["opt_loss"] = opt_loss
            evaluation["dec_loss"] = dec_loss

            for func in self.evaluators:
                evaluation[func.name] = func(decoded, test_targets)

        return evaluation



    def log_evaluation(self, evaluation, validation=False)
        """ Logs the evaluation results

        Arguments:
            evaluation: Dictionary with evaluation results
            validation: If True, switch on validation logging mode
        """
        eval_string = "    ".join(
            ["{}: {:.2f}".format(f.name, evaluation_res[f.name])
             for f in self.evaluators[:-1]])

        if len(self.evaluators) >= 1:
            main_evaluator = self.evaluators[-1]
            eval_string += colored("    {}: {:.2f}"
                                   .format(main_evaluator.name,
                                           evaluation_res[main_evaluator.name]),
                                   attrs=['bold'])

        log("opt. loss: {:.4f}    dec. loss: {:.4f}    {}"
            .format(eval_result['opt_loss'], eval_result['dec_loss'],
                    eval_string),
            color="yellow" if validation else "blue")





    def write_summaries(self, summary_str, tb_writer, validation=False):
        """ Writes TensorBoard summaries

        Arguments:
            summary_str: Summary string with results
            tb_writer: TensorBoard writer object
            validation: If True, summaries are written with validation tags
        """
        def format_eval_name(name):
            if hasattr(name, '__call__'):
                return name.__name__
            else:
                return str(name)

        tagvals = [("{}_{}".format("val" if validation else "train",
                                   format_eval_name(n)), val)
                   for n, val in eval_result(items())]

        tb_writer.add_summary(summary_str, seen_instances)

        if histograms_str:
            tb_writer.add_summary(histograms_str, seen_instances)

        external_str = tf.Summary(value=[tf.Summary.Value(tag=t, simple_value=v)
                                         for t, v in tagvals])
        tb_writer.add_summary(external_str, seen_instances)






    def run_training_loop(self, session):
        for epoch in range(1, self.epochs + 1):
            log_print("")
            log("Epoch {} starts".format(epoch), color="red")

            self.train_dataset.shuffle()
            train_batched_datasets = self.train_dataset.batch_dataset(
                self.batch_size)

            for batch_number, batch_dataset in enumerate(train_batched_datasets):
                self.training_step += 1

                if (self.training_step + 1) % self.logging_period == 0:
                    summary = self.run_batch(batch_dataset, summary=True)

                    evaluation = self.evaluate(session, batch_dataset)

                    self.log_evaluation(evaluation)
                    self.write_summaries(evaluation, tb_writer)




                else:
                    self.run_batch(session, batch_dataset)

                if (self.training_step + 1) % self.validation_period == 0:
                    evaluation = self.evaluate(session, val_dataset,
                                               validation=True)
                    self.log_evaluation(evaluation, validation=True)
                    self.write_summaries(evaluation, tb_writer, validation=True)








    def check_data(self):
        try:
            check_dataset_and_model(self.train_dataset, self.model)
            check_dataset_and_model(self.val_dataset, self.model)

            for test in self.test_datasets:
                check_dataset_and_model(self.test, self.model, test=True)

        except CheckingException as exc:
            log(str(exc), color='red')
            exit(1)






    def execute(self, *args):
        if self.random_seed is not None:
            tf.set_random_seed(self.random_seed)

        self.output = OutputDirectory(self.output_dir)

        ## sess, saver = initialize_tf(self.initial_variables, self.threads)
        session = self.create_session()

        try:
            self.run_training_loop(session)
        except KeyboardInterrupt:
            log("Training interrupted by user.")

        ## restore best (if link exists)

        log("Training finished. Maximum {} on validation data: {:.2f}, epoch {}"
            .format(evaluation_labels[-1], best_score, best_score_epoch))
