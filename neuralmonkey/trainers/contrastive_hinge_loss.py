import tensorflow as tf

from neuralmonkey.decoders import SequenceRegressor
from neuralmonkey.trainers.generic_trainer import Objective


def contrastive_hinge_loss(regressor: SequenceRegressor,
                           alpha: float = 0.1,
                           weight: float = None) -> Objective:

    valid_train_inputs = tf.gather(
        regressor.train_inputs, regressor.loss_valid_indices)

    valid_predictions = tf.gather(
        regressor.predictions, regressor.loss_valid_indices)

    normalized_refs = tf.nn.l2_normalize(valid_train_inputs, 1)
    normalized_preds = tf.nn.l2_normalize(valid_predictions, 1)

    shuffled_refs = tf.random_shuffle(normalized_refs)

    random_dist = 1 - tf.reduce_sum(shuffled_refs * normalized_preds, axis=[1])
    dist = 1 - tf.reduce_sum(normalized_refs * normalized_preds, axis=[1])

    loss = tf.reduce_mean(tf.maximum(dist - random_dist + alpha, 0))

    return Objective(
        name="{} - contrastive hinge loss".format(regressor.name),
        decoder=regressor,
        loss=loss,
        gradients=None,
        weight=weight,
    )
