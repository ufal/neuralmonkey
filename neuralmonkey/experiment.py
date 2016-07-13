import tensorflow as tf

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




    def run(self):
        """Run the experiment"""

        pass






    def loop_batch(self, batch_dataset):



class Model(object):







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
