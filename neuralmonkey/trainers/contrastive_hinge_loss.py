import tensorflow as tf

from neuralmonkey.model.stateful import Stateful
from neuralmonkey.trainers.generic_trainer import Objective


def contrastive_hinge_loss(source: Stateful,
                           reference: Stateful,
                           alpha: float = 0.1,
                           weight: float = None) -> Objective:

    refs = tf.stop_gradient(reference.output)

    normalized_refs = tf.nn.l2_normalize(refs, 1)
    normalized_preds = tf.nn.l2_normalize(source.output, 1)

    shuffled_refs = tf.stop_gradient(tf.random_shuffle(normalized_refs))

    random_dist = 1 - tf.reduce_sum(shuffled_refs * normalized_preds, axis=[1])
    dist = 1 - tf.reduce_sum(normalized_refs * normalized_preds, axis=[1])

    loss = tf.reduce_mean(tf.maximum(dist - random_dist + alpha, 0))

    return Objective(
        name="{} - contrastive hinge loss".format(source.name),
        decoder=reference,
        loss=loss,
        gradients=None,
        weight=weight,
    )
