import tensorflow as tf

class ImageEncoder(object):
    def __init__(self, shape, dropout_placeholder=None):
        assert(len(shape) == 3)
        with tf.variable_scope("image_encoder"):
            self.image_features = tf.placeholder(tf.float32,
                    shape=[None] + shape, name="image_input")

            self.encoded = tf.reduce_mean(self.image_features, reduction_indices=[1, 2],
                                          name="average_image")
            self.attention_tensor = \
                tf.reshape(self.image_features, [-1, shape[0] * shape[1], shape[2]], name="flatten_image")
                #tf.transpose(tf.reshape(self.image_features,
                #                        [-1, shape[0] * shape[1], shape[2]], name="flatten_image"), [0, 2, 1])
                #tf.reshape(self.image_features, [-1, shape[0] * shape[1], shape[2]], name="flatten_image")

class VectorImageEncoder(object):
    def __init__(self, dimension, dropout_placeholder=None):
        self.image_features = tf.placeholder(tf.float32, shape=[None, dimension])

        self.encoded = self.image_features
