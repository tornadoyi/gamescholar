import tensorflow as tf
import numpy as np
import time
import os
import scipy.signal

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class TrainWorker(object):

    def __init__(self,
                 ac, env,
                 global_step, summary_writer,
                 render = False,
                 gamma=0.9, lambda_=1.0, update_nsteps=20):

        self.ac = ac
        self.env = env
        self.global_step = global_step
        self.summary_writer = summary_writer
        self.gamma = gamma
        self.lambda_ = lambda_
        self.update_nsteps = update_nsteps
        self.render = render

        self.op_next_gloabl_step = self.global_step.assign_add(1)
        self._create_summary()



    def __call__(self, sess):
        self.sess = sess
        steps = 1
        buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []

        # pull model to local first
        self.ac.pull(sess)

        while True:

            s = self._reset()
            init_features = self.ac.get_initial_features(sess)
            features = init_features

            while True:
                a, v, next_features = self.ac.choose_action(sess, s, features)
                s_, r, done, info = self._step(a)
                if self.render: self.env.render()

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                buffer_v.append(v)

                if steps % self.update_nsteps == 0 or done:  # update global and assign to local net

                    v_ = 0 if done else self.ac.predict_value(sess, s, next_features)

                    rewards = np.asarray(buffer_r)
                    vpred_t = np.asarray(buffer_v + [v_])

                    rewards_plus_v = np.asarray(buffer_r + [v_])
                    batch_r = discount(rewards_plus_v, self.gamma)[:-1]
                    delta_t = rewards + self.gamma * vpred_t[1:] - vpred_t[:-1]
                    # this formula for the advantage comes "Generalized Advantage Estimation":
                    # https://arxiv.org/abs/1506.02438
                    batch_adv = discount(delta_t, self.gamma * self.lambda_)

                    feed_dict = {
                        self.ac.s: np.asarray(buffer_s),
                        self.ac.a: np.asarray(buffer_a),
                        self.ac.adv: batch_adv,
                        self.ac.r: batch_r,
                    }

                    summary = self.ac.learn(sess, [self.ac.summary_op], feed_dict)[0]

                    # sync from gloabl ac
                    self.ac.pull(sess)

                    # clear buffer
                    buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []


                    # update init features
                    init_features = next_features

                    # summary
                    self.summary_writer.add_summary(tf.Summary.FromString(summary), sess.run(self.global_step))
                    self.summary_writer.flush()


                s = s_
                features = next_features
                steps += 1

                if done:
                    self._end()
                    break



    def _create_summary(self):
        self.reward = tf.placeholder(tf.float32, shape=())
        self.running_reward = tf.placeholder(tf.float32, shape=())
        summary_ops = [
            tf.summary.scalar("woker/reward", self.reward),
            tf.summary.scalar("woker/running_reward", self.running_reward)
        ]
        self.summary_op = tf.summary.merge(summary_ops)


    def _reset(self):
        self.v_reward = 0
        return self.env.reset()


    def _step(self, a):
        s_, r, done, info = self.env.step(a)
        self.sess.run(self.op_next_gloabl_step)
        self.v_reward += r
        return s_, r, done, info


    def _end(self):
        if not hasattr(self, 'v_runing_reward'): self.v_runing_reward = self.v_reward
        self.v_runing_reward = 0.99 * self.v_runing_reward + 0.01 * self.v_reward

        summary, global_step = self.sess.run([self.summary_op, self.global_step], feed_dict = {
            self.reward: self.v_reward,
            self.running_reward: self.v_runing_reward,
        })
        self.summary_writer.add_summary(tf.Summary.FromString(summary), global_step)
        self.summary_writer.flush()







class PlayWorker(object):
    def __init__(self,
                 ac, env,
                 global_step, summary_writer,
                 render=False,
                 ):
        self.ac = ac
        self.env = env
        self.global_step = global_step
        self.summary_writer = summary_writer
        self.render = render

        self._create_summary()


    def __call__(self, sess):
        self.sess = sess

        while True:
            # pull model to local every game round
            self.ac.pull(sess)

            s = self._reset()
            init_features = self.ac.get_initial_features(sess)
            features = init_features

            while True:
                a, v, next_features = self.ac.choose_action(sess, s, features)
                s_, r, done, info = self._step(a)
                if self.render: self.env.render()

                s = s_
                features = next_features

                if done:
                    self._end()
                    break


    def _create_summary(self):
        self.reward = tf.placeholder(tf.float32, shape=())
        self.running_reward = tf.placeholder(tf.float32, shape=())
        summary_ops = [
            tf.summary.scalar("play/reward", self.reward),
            tf.summary.scalar("play/running_reward", self.running_reward)
        ]
        self.summary_op = tf.summary.merge(summary_ops)


    def _reset(self):
        self.v_reward = 0
        return self.env.reset()


    def _step(self, a):
        s_, r, done, info = self.env.step(a)
        self.v_reward += r
        return s_, r, done, info


    def _end(self):
        if not hasattr(self, 'v_runing_reward'): self.v_runing_reward = self.v_reward
        self.v_runing_reward = 0.99 * self.v_runing_reward + 0.01 * self.v_reward

        summary, global_step = self.sess.run([self.summary_op, self.global_step], feed_dict = {
            self.reward: self.v_reward,
            self.running_reward: self.v_runing_reward,
        })
        self.summary_writer.add_summary(tf.Summary.FromString(summary), global_step)
        self.summary_writer.flush()