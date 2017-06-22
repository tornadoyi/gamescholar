import tensorflow as tf
import numpy as np
from ac_net import ACNet



class ACWorker(object):

    def __init__(self, ac, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.ac = ac

    def train(self, update_nsteps=20, should_stop=None, step_callback=None, train_callback=None):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while (should_stop is None) or (not should_stop()):
            s = self.env.reset()

            # if GLOBAL_EP % 1 == 0:
            #     saver.save(SESS, "models-pig/a3c-sw1-player", global_step=GLOBAL_EP)
            while True:
                a = self.ac.choose_action(s)
                s_, r, done, info = self.env.step(self._transform_action(a))
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % update_nsteps == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.ac.predict_value(s_[np.newaxis, :])

                    # calculate R
                    buffer_R = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + self.gamma * v_s_
                        buffer_R.append(v_s_)
                        buffer_R.reverse()

                    # learn and sync to global ac
                    buffer_s, buffer_a, buffer_R = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_R)
                    feed_dict = {
                        self.ac.S: buffer_s,
                        self.ac.A: buffer_a,
                        self.ac.R: buffer_R,
                    }
                    self.ac.learn(feed_dict)

                    # sync from gloabl ac
                    self.ac.pull()

                    # clear buffer
                    buffer_s, buffer_a, buffer_r = [], [], []

                    # callback
                    if train_callback is not None: train_callback()

                s = s_
                total_step += 1
                if step_callback is not None: step_callback()

                if done: break



    def test(self, should_stop=None, render=False):
        total_step = 0
        while (should_stop is None) or (not should_stop()):
            total_step += 1
            s = self.env.reset()

            while True:
                if render: self.env.render()

                # do action
                a = self.ac.choose_action(s)
                s_, r, done, info = self.env.step(self._transform_action(a))
                if done: break


    def _transform_action(self, a):
        """transform discrete action to continous action space"""
        move_toward = [
            (0,0), (0,1), (0,-1), (-1,0), (1,0)
        ]
        return move_toward[a]
