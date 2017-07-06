import tensorflow as tf
import numpy as np

# Attention: this a3c output is discret!

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer



def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class Policy(object):
    def __init__(self, ob_space, ac_space, scope):
        self.state_shape = ob_space
        self.action_shape = ac_space
        self.scope = scope

        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))


        with tf.variable_scope(scope):
            self.l_a = tf.nn.elu(linear(x, 256, "la", normalized_columns_initializer(0.01)))
            self.logits = linear(self.l_a, ac_space, "action", normalized_columns_initializer(0.01))

            self.l_c = tf.nn.elu(linear(x, 128, "lc", normalized_columns_initializer(0.01)))
            self.vf = tf.reshape(linear(self.l_c, 1, "value", normalized_columns_initializer(1.0)), [-1])


            self.sample = categorical_sample(self.logits, ac_space)[0, :]
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)



    def act(self, sess, ob, *args):
        return sess.run([self.sample, self.vf], {self.x: [ob]})


    def value(self, sess, ob, *args):
        return sess.run(self.vf, {self.x: [ob]})[0]