import tensorflow as tf

class VectorEncoder(object):
    def __init__(self, dimension, output_shape, data_id):
        self.image_features = tf.placeholder(tf.float32, shape=[None, dimension])
        self.data_id = data_id

        self.flat = self.image_features

        project_w = tf.get_variable(shape=[dimension, output_shape], name="img_init_proj_W")
        project_b = tf.Variable(tf.zeros([output_shape]), name="img_init_b")

        self.encoded = tf.tanh(tf.matmul(self.flat, project_w) + project_b)

        self.attention_tensor = None
        self.attention_object = None

    def feed_dict(self, dataset, train=False):
        return {self.image_features: dataset.series[self.data_id]}


class PostCNNImageEncoder(object):
    def __init__(self, input_shape, output_shape, data_id,
                 dropout_keep_p=1.0, attention_type=None):
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.data_id = data_id
        self.dropout_keep_p = dropout_keep_p
        self.attention_type = attention_type


        with tf.variable_scope("image_encoder"):
            self.dropout_placeholder = tf.placeholder(tf.float32)
            self.image_features = tf.placeholder(tf.float32,
                                                 shape=[None] + input_shape,
                                                 name="image_input")

            self.flat = tf.reduce_mean(self.image_features, reduction_indices=[1, 2],
                                       name="average_image")

            project_w = tf.Variable(tf.random_normal([input_shape[2], output_shape]),
                                    name="img_init_proj_W")
            project_b = tf.Variable(tf.zeros([output_shape]), name="img_init_b")

            self.encoded = tf.tanh(tf.matmul(self.flat, project_w) + project_b)

            self.attention_tensor = \
                tf.reshape(self.image_features,
                           [-1, input_shape[0] * input_shape[1], input_shape[2]],
                           name="flatten_image")

            self.attention_object = \
                attention_type(self.attention_tensor,
                               scope="attention_img",
                               dropout_placeholder=self.dropout_placeholder) \
                if attention_type else None

    def feed_dict(self, dataset, train=False):
        res = {self.image_features: dataset.series[self.data_id]}

        if train:
            res[self.dropout_placeholder] = self.dropout_keep_p
        else:
            res[self.dropout_placeholder] = 1.0

        return res

