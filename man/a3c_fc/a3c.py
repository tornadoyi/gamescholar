"""
flat-a3c launch entry
A3C with a flattened 1D array state (with all info of all objects)
"""

import tensorflow as tf
import threading
import gym
import os
import shutil
import time

from policy import Policy
from ac import ActorCritic
from worker import Worker
import config

LR = 0.001  # 1e-4 in openai


OUTPUT_GRAPH = True

LOG_DIR = './log'

N_WORKERS = 8

UPDATE_GLOBAL_ITER = 20

GAMMA = 0.9

ENTROPY_BETA = 0.01

LAMBDA = 1.0

GAME_NAME = config.GAME_NAME


def run(render=False):
    env = gym.make(GAME_NAME).unwrapped
    env.reset()
    N_S, N_A = env.observation_space.shape, 4#env.action_space.n
    env.close()

    sess = tf.InteractiveSession()

    #optimizer = tf.train.RMSPropOptimizer(LR, name='RMSPropA')
    optimizer = tf.train.AdamOptimizer(LR)

    global_pi = Policy(N_S, N_A, 'global')

    # Create train worker
    workers = []
    for i in range(N_WORKERS):
        i_name = 'pi_%i' % i  # worker name
        env = gym.make(GAME_NAME).unwrapped
        pi = Policy(N_S, N_A, i_name)
        ac = ActorCritic(sess, pi, optimizer, global_pi=global_pi, entropy_beta=ENTROPY_BETA)
        worker = Worker(ac, env, GAMMA, LAMBDA)
        workers.append(worker)


    # init variables
    sess.run(tf.global_variables_initializer())


    # train workers
    worker_threads = []
    for i in range(len(workers)):
        worker = workers[i]
        if len(workers) > 1 and i == 0:
            job = lambda: worker.test(render=render)
        else:
            job = lambda: worker.train(update_nsteps=UPDATE_GLOBAL_ITER)

        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)


    # wait
    COORD = tf.train.Coordinator()
    COORD.join(worker_threads)


if __name__ == '__main__':
    run()