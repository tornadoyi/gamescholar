"""
flat-a3c launch entry
A3C with a flattened 1D array state (with all info of all objects)
"""

import tensorflow as tf
import threading
import gym
import os
import shutil
from game import config
from ac_net import ACNet
from ac_worker import ACWorker


LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic


OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = 2
MAX_GLOBAL_EP = 30000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 20
GAMMA = 0.9
ENTROPY_BETA = 0.001

def run():
    env = gym.make(config.GAME_NAME)
    s = env.reset()
    N_S = s.shape[0] # env.observation_space.shape[0]
    N_A = 5 # env.action_space.n
    env.close()

    sess = tf.InteractiveSession()

    OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
    # OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
    GLOBAL_AC = ACNet(sess, GLOBAL_NET_SCOPE, N_S, N_A, OPT_A,
                      entropy_beta=ENTROPY_BETA)  # we only need its params

    workers = []
    # Create worker
    assert N_WORKERS >= 2  # there must be a rendering worker and a training worker
    for i in range(N_WORKERS):
        run_mode = "rendering" if i == 0 else "training"
        i_name = 'W_%i' % i  # worker name
        env = gym.make(config.GAME_NAME)
        env.env.game.speed_scale = 8
        ac = ACNet(sess, i_name, N_S, N_A, OPT_A, global_ac=GLOBAL_AC, entropy_beta=ENTROPY_BETA)
        workers.append(ACWorker(ac, env, GAMMA, run_mode=run_mode))

    COORD = tf.train.Coordinator()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, sess.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work(sess)
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)


if __name__ == '__main__':
    run()