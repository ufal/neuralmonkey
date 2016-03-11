import tensorflow as tf

class ImageEncoder(object):
    def __init__(self, dropout_placeholder=None):
        with tf.variable_scope("image_encoder"):
            self.image_features = tf.placeholder(tf.float32,
                    shape=[None, 14, 14, 512], name="image_input")

            self.encoded = tf.reduce_mean(self.image_features, reduction_indices=[1, 2],
                                          name="average_image")
            self.attention_tensor = tf.reshape(self.image_features,
                    [-1, 196, 512], name="flatten_image")

            if dropout_placeholder:
                self.encoded = tf.nn.dropout(self.encoded, dropout_placeholder, name="avg_image_dropped")
