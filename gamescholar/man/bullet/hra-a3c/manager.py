import numpy as np
from easydict import EasyDict as edict
import tensorflow as tf
from policy import Policy
from ac import ActorCritic
from head import BulletHead
import config
import gym


LR = 1e-3  # 1e-4 in openai

UPDATE_GLOBAL_ITER = 20

GAMMA = 0.9

ENTROPY_BETA = 0.01

LAMBDA = 1.0

GAME_NAME = config.GAME_NAME

N_S, N_A = (2, ), 4

bullet_config = None

def init_global_bullet_policy():
    global bullet_config
    bullet_config = edict(
        global_pi=Policy(N_S, N_A, 'global'),
        optimizer=tf.train.AdamOptimizer(LR),
    )


class Manager():


    def __init__(self, sess, index):
        self.sess = sess
        self.index = index

        # env
        self.env = gym.make(GAME_NAME).unwrapped
        self.game = self.env.game

        self.bullet_head_cache = []


        # reset bullets
        self.env.reset()
        self.init_bullet_head_cache()



    def train(self, should_stop=None):
        while (should_stop is None) or (not should_stop()):

            # reset game
            self.env.reset()

            # get heads
            heads = self._reset_bullet_heads()

            while True:
                # pick heads
                heads = [head for head in heads if not head.terminal]

                # save state
                for head in heads: head.save_state()

                # predict action
                sum = np.zeros(4, dtype=float)
                for head in heads:
                    a, v = head.predict()
                    sum += a * (np.e ** v)

                idx = np.argmax(sum)
                a = np.eye(4)[idx]


                # game step
                _, r, done, _ = self.env.step(a)


                # do step
                for head in heads: head.step(a)


                if done:
                    break



    def test(self, render=False):

        runing_reward = None

        epoch = 0

        while True:

            # data
            game_reward = 0

            epoch += 1

            # reset game
            self.env.reset()

            # get heads
            heads = self._reset_bullet_heads()


            while True:
                # pick heads
                heads = [head for head in heads if not head.terminal]

                # save state
                for head in heads: head.save_state()

                # predict action
                sum = np.zeros(4, dtype=float)
                for head in heads:
                    a, v = head.predict()
                    sum += a * (np.e ** v)

                idx = np.argmax(sum)
                a = np.eye(4)[idx]


                # game step
                _, r, done, _ = self.env.step(a)
                if render: self.env.render()

                game_reward += r

                if done:
                    if runing_reward is None:
                        runing_reward = self.env.game.steps
                    else:
                        runing_reward = runing_reward * 0.99 + 0.01 * self.env.game.steps


                    print('epoch: {0} reward: {1} runing reward: {2}'.format(epoch, self.env.game.steps, runing_reward))

                    break





    def _reset_bullet_heads(self):
        player = self.game.map.players[0]
        bullets = self.game.map.bullets
        assert len(bullets) <= len(self.bullet_head_cache)

        heads = self.bullet_head_cache[0:len(bullets)]
        for i in range(len(bullets)):
            head, bullet = heads[i], bullets[i]
            head.reset(player, bullet)
            head.sync()

        return heads




    def init_bullet_head_cache(self):

        def _create(index):
            name = 'bullet-{0}-{1}'.format(self.index, index)
            pi = Policy(N_S, N_A, name)
            ac = ActorCritic(self.sess, pi, bullet_config.optimizer,
                             global_pi=bullet_config.global_pi,
                             entropy_beta=ENTROPY_BETA)

            head = BulletHead(self.env, ac, update_step=UPDATE_GLOBAL_ITER, gamma=GAMMA, lambda_=LAMBDA)
            return head


        for i in range(len(self.game.map.bullets)):
            head = _create(i)
            self.bullet_head_cache.append(head)


