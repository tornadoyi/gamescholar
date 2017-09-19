import numpy as np
import tensorflow as tf



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



class Model(object):

    def __init__(self,
                 input_shape,
                 action_size,
                 optimizer,
                 entropy_beta=0.01,
                 global_model_vars = None,
                 ):


        self.input_shape = input_shape
        self.action_size = action_size
        self.optimizer = optimizer
        self.entropy_beta = entropy_beta
        self.global_model_vars = global_model_vars


        # s should be (batch, features)
        self.s = tf.placeholder(tf.float32, [None] + list(self.input_shape), name='s')
        self.a = tf.placeholder(tf.float32, [None, self.action_size], name="a")
        self.adv = tf.placeholder(tf.float32, [None], name="adv")
        self.r = tf.placeholder(tf.float32, [None], name="r")

        self.batch_size = tf.to_float(tf.shape(self.s)[0])

        # create network
        self._create_network()

        # create loss
        self._create_loss()

        # create summary
        self._create_summary()

        # collect var list
        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name)


    def _create_network(self):

        '''
        s = tf.expand_dims(self.s, axis=0)

        lstm_size = 256
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
        self.state_in = lstm_cell.zero_state(1, tf.float32)
        lstm_outputs, self.state_out = tf.nn.dynamic_rnn(lstm_cell, s, initial_state=self.state_in, time_major=False)

        x = tf.reshape(lstm_outputs, [-1, lstm_size])
        '''

        self.state_in = self.state_out = tf.constant(0.0)
        x1 = tf.nn.relu(linear(self.s, 256, "pi_input", normalized_columns_initializer(0.01)))
        x2 = tf.nn.relu(linear(self.s, 128, "vf_input", normalized_columns_initializer(0.01)))


        self.logits = linear(x1, self.action_size, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x2, 1, "value", normalized_columns_initializer(1.0)), [-1])

        self.sample = categorical_sample(self.logits, self.action_size)[0, :]
        self.probs = tf.nn.softmax(self.logits)


        self.model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)


    def _create_loss(self):
        log_prob_tf = tf.nn.log_softmax(self.logits)
        self.pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.a, [1]) * self.adv)

        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(self.vf - self.r))
        self.entropy = - tf.reduce_sum(self.probs * log_prob_tf)

        self.loss = self.pi_loss + 0.5 * self.vf_loss - self.entropy * self.entropy_beta

        self.grads = tf.gradients(self.loss, self.model_vars)

        self.grads_norm, _ = tf.clip_by_global_norm(self.grads, 40.0)

        global_model_vars = self.global_model_vars or self.model_vars
        self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.model_vars, global_model_vars)])

        grads_and_vars = list(zip(self.grads_norm, global_model_vars))

        self.train_op = self.optimizer.apply_gradients(grads_and_vars)


    def _create_summary(self):
        summary_ops = [
            tf.summary.scalar("model/pi_loss", self.pi_loss / self.batch_size),
            tf.summary.scalar("model/value_loss", self.vf_loss / self.batch_size),
            tf.summary.scalar("model/entropy", self.entropy / self.batch_size),
            tf.summary.scalar("model/grad_norm", tf.global_norm(self.grads)),
        ]
        self.summary_op = tf.summary.merge(summary_ops)




    def choose_action(self, sess, s, features):
        return sess.run([self.sample, self.vf, self.state_out], {self.s: [s], self.state_in: features})


    def predict_value(self, sess, s, features):
        return sess.run(self.vf, {self.s: [s], self.state_in: features})[0]


    def learn(self, sess, fetches=[], feed_dict={}):  # run by a local
        return sess.run([self.train_op] + fetches, feed_dict)[1:]  # local grads applies to global net


    def pull(self, sess): sess.run(self.sync)


    def get_initial_features(self, sess): return sess.run([self.state_in])[0]