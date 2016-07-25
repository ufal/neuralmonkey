import tensorflow as tf

from neuralmonkey.logging import log, log_print, debug
from neuralmonkey.checking import check_dataset_and_coders


class Model(object):

    ## nezavislej na datech, ale potrebuje slovnik
    ## architektura, znovupouzitelny jak pro trenink tak pro run

    ## kdyz uz potrebuje slovnik, mohl by si i sam delat prelouskavani z vet
    ## do vektoru (+ preprocessing)

    def __init__(self, decoder, encoders, runner, postprocess=lambda x: x):

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

        result_raw, opt_loss, dec_loss = self.runner(
            sess, dataset, self.encoders + [self.decoder])
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







class Experiment(object):


    def __init__(self, model, trainer, train_dataset, val_dataset,
                 evaluation, **kwargs):

        self.model = model
        self.trainer = trainer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.evaluation = evaluation

        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 64)
        self.logging_period = kwargs.get('logging_period', 20)
        self.validation_period = kwargs.get('validation_period', 500)

        self.training_step = 0
        self.training_seen_instances = 0



    def log_evaluation(self, train=False):
        """ Logs the evaluation results and writes summaries to TensorBoard """

        pass



    def run(self):
        """Run the experiment"""

        pass


    def run_loop(self, epochs):
        for epoch in range(1, epochs + 1):
            log_print("")
            log("Epoch {} starts".format(epoch), color="red")

            self.run_epoch()



    def run_epoch(self):
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

                _, _, train_evaluation = self.run_on_dataset(batch_dataset,
                                                             write_out=False)

                ## process evaluation

            else:
                self.run_batch(batch_dataset)

            if (self.training_step + 1) % self.validation_period == 0:
                self.validate()




    def run_batch(self, batch_dataset, summary=False):
        """ Runs one batch of training throught the computation graph

        This method creates feed dictionaries, gets target sentences for
        loss computation and run the trainer.

        Arguments:
            batch_dataset: the dataset
            summary: whether or not to create summaries (for tensorboard)
        """
        feed_dict = self.model.feed_dicts(batch_dataset, train=True)
        target_sentences = batch_dataset.get_series(self.model.decoder.data_id)

        self.training_seen_instances += len(target_sentences)

        return self.trainer.run(self.session, feed_dict, summary)



    def validate(self):

        """
        run model on validation data, get score
        save if high score
        hooray if best score - symlink and stuff
        log
        """






        pass











class Model(object):


### feed dicty budou stejne provazany jako konfigurace




### konfigurace se tu řešit nebude

### (NE)?bude se tu řešit vytváření složky

### bude se tu řešit random seed

### bude se tu řešit checking

### bude tu rozhodne training_loop


### bude se tu tvořit session a bude se tu ukldádat a loadovat tf


### rozdělit main na experiment a model?



###  veci co se resi v main a v training loop:
"""
stávající:
main:

- create config
- random seed
- create/reuse dir
- check model
- initialize log dir
- set up logging
- initialize tensorflow session and saver

train loop:

- restore variables
- set up saving
- init tensorboard
- training loop

Z KONFIGU:

training má:

běh programu potřebuje:

output
overwrite_output_dir


samotný trénování potřebuje:

model(decoder, [encoders]),
batch_size
runner
trainer
evaluation
postprocess
train_dataset
val_dataset
epochs
validation_period
logging_period
save_n_best

threads

... a pak datasety. ty jsou spíš součást trainingu. ALE NE VOCABULARIES - ty uz
jsou soucast modelu.


"""
