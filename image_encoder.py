import tensorflow as tf

class ImageEncoder(object):
    def __init__(self):
        self.image_features = tf.placeholder(tf.float32,
                shape=[None, 14, 14, 512], "image_input")

        self.encoded = tf.reduce_mean(self.image_features, reduction_indices=[1, 2],
                                      name="average_image")
        self.attention_tensor = tf.reshape(self.image_features,
                [None, 196, 512], name="flatten_image")
