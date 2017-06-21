import tensorflow as tf
import numpy as np
from .ac_net import ACNet


class ACWorker(object):
    def __init__(self, ac, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.AC = ac

    def work(self, sess, update_nsteps=20, should_stop=None, step_callback=None, train_callback=None):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while (should_stop is None) or (not should_stop()):
            s = self.env.reset()

            # if GLOBAL_EP % 1 == 0:
            #     saver.save(SESS, "models-pig/a3c-sw1-player", global_step=GLOBAL_EP)
            while True:
                a = self.AC.choose_action(s)
                a = self._transform_action(a)
                s_, r, done, info = self.env.step(a)
                print('action:', a, 'reward:', r)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % update_nsteps == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + self.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                    if train_callback is not None: train_callback()

                s = s_
                total_step += 1
                if step_callback is not None: step_callback()

                if done: break

    def _transform_action(self, a):
        """transform discrete action to continous action space"""
        move_toward = [
            (0,0), (0,1), (0,-1), (-1,0), (1,0)
        ]
        return move_toward[a]
