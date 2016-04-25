import tensorflow as tf

class VectorImageEncoder(object):
    def __init__(self, dimension, dropout_placeholder=None):
        self.image_features = tf.placeholder(tf.float32, shape=[None, dimension])

        self.encoded = self.image_features

        self.attention_tensor = None

    def feed_dict(self, images, batch_size, dicts=None):
        if dicts == None:
            dicts = [{} for _ in range(len(sentences) / batch_size + int(len(sentences) % batch_size))]

        for fd, start in zip(dicts, range(0, len(images.shape[0], batch_size))):
            fd[self.image_features] = images[start:start+batch_size]

class ImageEncoder(object):
    def __init__(self, input_shape, output_shape, dropout_placeholder=None):
        assert(len(input_shape) == 3)
        with tf.variable_scope("image_encoder"):
            self.image_features = tf.placeholder(tf.float32,
                    shape=[None] + input_shape, name="image_input")

            self.flat = tf.reduce_mean(self.image_features, reduction_indices=[1, 2],
                                          name="average_image")

            project_W = tf.Variable(tf.random_normal([input_shape[2], output_shape]), name="img_init_proj_W")
            project_b = tf.Variable(tf.zeros([output_shape]), name="img_init_b")

            self.encoded = tf.tanh(tf.matmul(self.flat, project_W) + project_b)

            self.attention_tensor = \
                tf.reshape(self.image_features, [-1, input_shape[0] * input_shape[1], input_shape[2]], name="flatten_image")

    def feed_dict(self, images, batch_size, dicts=None):
        if dicts == None:
            dicts = [{} for _ in range(len(sentences) / batch_size + int(len(sentences) % batch_size))]

        for fd, start in zip(dicts, range(0, len(images.shape[0], batch_size))):
            fd[self.image_features] = images[start:start+batch_size]

