import tensorflow as tf

from neuralmonkey.logging import log, log_print, debug
from neuralmonkey.checking import check_dataset_and_coders


class Experiment(object):


    def __init__(self, decoder, encoders, runner, trainer, train_dataset,
                 val_dataset, evaluation, **kwargs):

        self.decoder = decoder
        self.encoders = encoders # TODO resolve issue #27
        self.runner = runner
        self.trainer = trainer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.evaluation = evaluation

        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 64)
        self.postprocess = kwargs.get('postprocess', lambda x: x)

        self.logging_period = kwargs.get('logging_period', 20)
        self.validation_period = kwargs.get('validation_period', 500)

        self.training_step = 0
        self.training_seen_instances = 0




    def run(self):
        """Run the experiment"""

        pass


    def run_loop(self, epochs):

        for epoch in range(1, epochs + 1):

            log_print("")
            log("Epoch {} starts".format(epoch), color="red")

            self.train_dataset.shuffle()
            train_batched_datasets = self.train_dataset.batch_dataset(
                self.batch_size)

            for batch_number, batch_dataset in enumerate(
                    train_batched_datasets):
                self.training_step += 1

                if (self.training_step + 1) % self.logging_period == 0:
                    summary = self.loop_batch(batch_dataset, summary=True)

                    _, _, train_evaluation = self.run_on_dataset(
                        batch_dataset, write_out=False)
                else:
                    self.loop_batch(batch_dataset)

                if (self.training_step + 1) % self.validation_period == 0:
                    self.validate()




    def validate(self):

        """
        run model on validation data, get score
        save if high score
        hooray if best score - symlink and stuff
        log
        """

        pass



    def loop_batch(self, batch_dataset, summary=False):


        feed_dict = self.decoder.feed_dicts(batch_dataset train=True)
        target_sentences = batch_dataset.get_series(self.decoder.data_id)

        self.training_seen_instances += len(target_sentences)

        return self.trainer.run(self.session, feed_dict, summary)









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
