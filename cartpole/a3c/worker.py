import tensorflow as tf
import numpy as np
import time
import os
import scipy.signal

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Worker(object):

    def __init__(self, ac, env, gamma=0.9, lambda_=1.0):
        self.env = env
        self.gamma = gamma
        self.lambda_ = lambda_
        self.ac = ac


    def train(self, update_nsteps=20, should_stop=None):
        total_step = 1
        buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []
        while (should_stop is None) or (not should_stop()):

            s = self.env.reset()

            while True:
                a, v = self.ac.choose_action(s)
                s_, r, done, info = self.env.step(np.argmax(a))

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                buffer_v.append(v)

                if total_step % update_nsteps == 0 or done:  # update global and assign to local net

                    v_ = 0 if done else self.ac.predict_value(s)

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

                    self.ac.learn(feed_dict)

                    # sync from gloabl ac
                    self.ac.pull()

                    # clear buffer
                    buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []


                s = s_
                total_step += 1

                if done: break



    def test(self, should_stop=None, render=False):
        total_step = 0
        running_reward = 0.0
        epoch = 0
        epoch_reward = 0

        while (should_stop is None) or (not should_stop()):
            # upgrade
            self.ac.pull()

            # reset env
            s = self.env.reset()

            while True:

                # choose and do action
                a, v = self.ac.choose_action(s)

                # do
                s, r, done, info = self.env.step(np.argmax(a))
                epoch_reward += r

                # render
                if render: self.env.render()

                # step + 1
                total_step += 1

                # check terminal
                if done:
                    epoch += 1
                    running_reward = running_reward * 0.99 + 0.01 * epoch_reward

                    print('epoch: {0} reward: {1}  runing reward: {2}'.format(epoch, epoch_reward, running_reward))

                    epoch_reward = 0
                    break




