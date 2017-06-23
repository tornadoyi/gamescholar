import tensorflow as tf
import numpy as np

# Attention: this a3c output is discret!

class ACNet(object):
    def __init__(self,
                 sess, scope, n_state, n_action, optimizer,
                 global_ac=None,
                 entropy_beta=0.001):
        self.sess = sess
        self.scope = scope
        self.n_state = n_state
        self.n_action = n_action
        self.optimizer = optimizer
        self.global_ac = global_ac


        if global_ac is None:  # get global network
            with tf.variable_scope(scope):
                self.S = tf.placeholder(tf.float32, [None, n_state], 'S')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.S = tf.placeholder(tf.float32, [None, n_state], 'S')
                self.A = tf.placeholder(tf.int32, [None, ], 'A')
                self.R = tf.placeholder(tf.float32, [None, 1], 'R')

                self.pi, self.v = self._build_net()

                td = tf.subtract(self.R, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.pi) * tf.one_hot(self.A, n_action, dtype=tf.float32),
                                             axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    entropy = -tf.reduce_sum(self.pi * tf.log(self.pi), axis=1,
                                             keep_dims=True)  # encourage exploration
                    self.exp_v = entropy_beta * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                self.total_loss = self.a_loss + self.c_loss

                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_ac.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_ac.c_params)]
                with tf.name_scope('push'):
                    self.push_a_params_op = optimizer.apply_gradients(zip(self.a_grads, global_ac.a_params))
                    self.push_c_params_op = optimizer.apply_gradients(zip(self.c_grads, global_ac.c_params))

    def _build_net(self):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.S, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            pi = tf.layers.dense(l_a, self.n_action, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.S, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        return pi, v


    def learn(self, feed_dict, fetches=[]):  # run by a local
        return self.sess.run([self.push_a_params_op, self.push_c_params_op] + fetches, feed_dict)[2:]  # local grads applies to global net

    def pull(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = self.sess.run(self.pi, feed_dict={self.S: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def predict_value(self, s):
        return self.sess.run(self.v, {self.S: s})[0, 0]



