import tensorflow as tf
import numpy as np
import time
import os
import scipy.signal
from gymgame.engine import Vector2


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Head(object):
    def __init__(self, env, ac, update_step=20, gamma=0.9, lambda_=1.0):
        self.env = env
        self.game = env.game
        self.update_step = update_step
        self.gamma = gamma
        self.lambda_ = lambda_
        self.ac = ac
        self.terminal = None


    def reset(self, master, slave):
        self.master = master
        self.slave = slave
        self.map = self.master.map
        self.map_size = self.map.bounds.size
        self.terminal = self.slave.map is None
        self.buffer_s, self.buffer_a, self.buffer_r, self.buffer_v = [], [], [], []
        assert self.terminal is False


    def save_state(self):
        s = self._state()
        self.buffer_s.append(s)


    def predict(self):
        a, v = self.ac.choose_action(self.buffer_s[-1])
        self.buffer_v.append(v)
        return a, v


    def step(self, a):
        self.buffer_a.append(a)
        self.buffer_r.append(self._reward())
        self.terminal = self.slave.map is None

        if self.game.steps % self.update_step == 0 or self.terminal or self.game.terminal:
            self.learn(self.terminal or self.game.terminal)
            self.sync()



    def sync(self): self.ac.pull()



    def learn(self, terminal):
        if len(self.buffer_s) == 0: return
        v_ = 0 if terminal else self.ac.predict_value(self._state())

        rewards = np.asarray(self.buffer_r)
        vpred_t = np.asarray(self.buffer_v + [v_])

        rewards_plus_v = np.asarray(self.buffer_r + [v_])
        batch_r = discount(rewards_plus_v, self.gamma)[:-1]
        delta_t = rewards + self.gamma * vpred_t[1:] - vpred_t[:-1]
        # this formula for the advantage comes "Generalized Advantage Estimation":
        # https://arxiv.org/abs/1506.02438
        batch_adv = discount(delta_t, self.gamma * self.lambda_)

        feed_dict = {
            self.ac.s: np.asarray(self.buffer_s),
            self.ac.a: np.asarray(self.buffer_a),
            self.ac.adv: batch_adv,
            self.ac.r: batch_r,
        }

        self.ac.learn(feed_dict)

        # clear
        self.buffer_s, self.buffer_a, self.buffer_r, self.buffer_v = [], [], [], []


    def _state(self):
        slave_pos = self.slave.attribute.position
        master_pos = self.master.attribute.position
        s = (slave_pos - master_pos) / self.map_size
        return s


    def _reward(self): pass




class BulletHead(Head):

    DANGER_STEP = 10

    def reset(self, *args, **kwargs):
        super(BulletHead, self).reset(*args, **kwargs)
        self.max_range = np.max(self.map_size)
        self.distance = self._distance()
        self.hit_distance = self.master.attribute.radius + self.slave.attribute.radius
        self.danger_distance = self.game.delta_time * self.slave.attribute.speed * self.DANGER_STEP + self.hit_distance



    def _reward(self):
        slave_pos = self.slave.attribute.position
        master_pos = self.master.attribute.position
        d = slave_pos.distance(master_pos)
        sign = 1 if (d - self.distance) > 0 else -1

        if d > self.danger_distance:
            r = 0
        else:
            r = 1 / (np.e ** (d - self.hit_distance))
            r = np.min([r, 1])
            r *= sign

        self.distance = d
        return r



    def _distance(self):
        slave_pos = self.slave.attribute.position
        master_pos = self.master.attribute.position
        d = slave_pos.distance(master_pos)
        return d
