import numpy as np
import gym
import config
import tensorflow as tf
from gymgame.engine import Vector2


LR = 1e-2

UPDATE_STEPS = 1

CONTINUES = False

class Network(object):
    def __init__(self, sess, n_states, n_action, optimizer, scope):
        self.sess = sess
        self.n_states = n_states
        self.n_action = n_action
        self.optimizer = optimizer
        self.scope = scope

        self.s = tf.placeholder(tf.float32, [None, n_states], name='x')  # input State
        self.y = tf.placeholder(tf.float32, [None, n_action], name='y')  # label

        if CONTINUES:
            self.pi = self._build_net_continues()
        else:
            self.pi = self._build_net_discrete()

        self.loss = tf.nn.l2_loss(self.y - self.pi)


        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        self.grads = tf.gradients(self.loss, self.params)
        self.clip_grads, _ = tf.clip_by_global_norm(self.grads, 1.0)
        #self.clip_grads = self.grads

        self.train_op = optimizer.apply_gradients(zip(self.clip_grads, self.params))


    def _build_net_discrete(self):
        def normalized_columns_initializer(std=1.0):
            def _initializer(shape, dtype=None, partition_info=None):
                out = np.random.randn(*shape).astype(np.float32)
                out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                return tf.constant(out)

            return _initializer

        with tf.variable_scope('actor'):
            w_init = normalized_columns_initializer(1.0)

            layer = self.s
            layer = tf.layers.dense(layer, 256, tf.nn.tanh, kernel_initializer=w_init, name='input')
            layer = tf.layers.dense(layer, self.n_action, tf.nn.softmax, kernel_initializer=w_init, name='output')

        return layer


    def _build_net_continues(self):
        def normalized_columns_initializer(std=1.0):
            def _initializer(shape, dtype=None, partition_info=None):
                out = np.random.randn(*shape).astype(np.float32)
                out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                return tf.constant(out)

            return _initializer

        with tf.variable_scope('actor'):
            w_init = normalized_columns_initializer(1.0)

            layer = self.s
            layer = tf.layers.dense(layer, 256, tf.nn.tanh, kernel_initializer=w_init, name='input')
            layer = tf.layers.dense(layer, self.n_action, None, kernel_initializer=w_init, name='output')

            # normalize
            norm = tf.sqrt(tf.reduce_sum(tf.square(layer)))
            layer = layer / tf.maximum(norm, 1e-6)
        return layer




    def choose_action(self, s):
        directs = self.sess.run(self.pi, feed_dict={self.s: s[np.newaxis, :]})
        direct = np.squeeze(directs)
        return direct


    def learn(self, feed_dict, fetches = []):
        return self.sess.run([self.train_op] + fetches, feed_dict=feed_dict)[1:]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def get_label_continues(s):

    player = s[0:2]
    npc = s[2:].reshape((-1, 2))

    d2 = np.sum(np.square(player - npc), axis=1)
    index = np.argmin(d2)

    target = npc[index]
    direct = Vector2(*(target - player)).normalized

    return direct


def get_label_discrete(s):
    direct = get_label_continues(s)
    label = np.zeros(4, dtype=float)
    if direct.x >= 0: label[1] = abs(direct.x)
    else: label[3] = abs(direct.x)

    if direct.y >= 0: label[0] = abs(direct.y)
    else: label[2] = abs(direct.y)

    norm = np.sum(label)
    label = label if norm == 0 else label / norm

    return label




def run(render=False):

    env = gym.make(config.config.GAME_NAME)
    _s = env.reset()
    N_S = _s.shape[0]  # env.observation_space.shape[0]

    if CONTINUES:
        N_A = 2  # env.action_space.n
    else:
        N_A = 4


    # create network
    sess = tf.Session()
    optimizer = tf.train.RMSPropOptimizer(LR, name='RMSPropA')
    net = Network(sess, N_S, N_A, optimizer, scope='net')


    # init variables
    sess.run(tf.global_variables_initializer())

    total_step = 0
    while True:

        buffer_s, buffer_a, buffer_y = [], [], []
        s = env.reset()
        done = False

        while not done:
            a = net.choose_action(s)

            if CONTINUES: y = get_label_continues(s)
            else: y = get_label_discrete(s)


            buffer_s.append(s)
            buffer_a.append(a)
            buffer_y.append(y)

            s_, r, done, info = env.step(a)
            if render: env.render()


            if total_step % UPDATE_STEPS == 0 or done:
                buffer_s, buffer_a, buffer_y = np.array(buffer_s), np.array(buffer_a), np.array(buffer_y)
                feed_dict = {
                    net.s: buffer_s,
                    net.y: buffer_y,
                }

                # learn
                loss = net.learn(feed_dict, [net.loss])

                # clear buffer
                buffer_s, buffer_a, buffer_y = [], [], []

                # print log
                if total_step % 100 == 0:
                    print('loss: {0}'.format(loss))


            s = s_


            total_step += 1




if __name__ == '__main__':
    run()