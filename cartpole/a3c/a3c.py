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
import gym
from ac_net import ACNet
from ac_worker import ACWorker


LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic

GAME = 'CartPole-v0'

OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = 8
MAX_GLOBAL_EP = 30000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 20
GAMMA = 0.9
ENTROPY_BETA = 0.001

SAVE_SECOND = 60 * 10
SAVE_PATH = "models/a3c"

class SaveWorker(object):
    def __init__(self, sess):
        self.sess = sess
        self.saver = tf.train.Saver()

    def __call__(self, *args, **kwargs):
        dir = os.path.dirname(SAVE_PATH)
        if not os.path.isdir(dir): os.makedirs(dir)
        i = 0
        while True:
            i += 1
            time.sleep(SAVE_SECOND)
            self.saver.save(self.sess, SAVE_PATH, global_step=i)


def run(render=False):
    env = gym.make(GAME)
    s = env.reset()
    N_S, N_A = env.observation_space.shape[0], env.action_space.n
    env.close()

    sess = tf.InteractiveSession()

    OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
    # OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
    GLOBAL_AC = ACNet(sess, GLOBAL_NET_SCOPE, N_S, N_A, OPT_A,
                      entropy_beta=ENTROPY_BETA)  # we only need its params

    # Create train worker
    workers = []
    for i in range(N_WORKERS):
        i_name = 'W_%i' % i  # worker name
        env = gym.make(GAME)
        ac = ACNet(sess, i_name, N_S, N_A, OPT_A, global_ac=GLOBAL_AC, entropy_beta=ENTROPY_BETA)
        workers.append(ACWorker(ac, env, GAMMA, name=i_name))

    # create test worker
    env = gym.make(GAME)
    ac = ACNet(sess, 'test', N_S, N_A, OPT_A, global_ac=GLOBAL_AC, entropy_beta=ENTROPY_BETA)
    tester = ACWorker(ac, env, GAMMA, name="test")

    # create save worker
    saver = SaveWorker(sess)

    # init variables
    sess.run(tf.global_variables_initializer())

    '''
    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, sess.graph)
    '''

    worker_threads = []

    # train workers
    for worker in workers:
        job = lambda: worker.train()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)

    # test worker
    job = lambda: tester.test(render=render)
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)

    # save worker
    job = lambda: saver()
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)


    # wait
    COORD = tf.train.Coordinator()
    COORD.join(worker_threads)


if __name__ == '__main__':
    run(True)