import tensorflow as tf
import gym

from model import A3CModel
from worker import A3CWorker
import config


LR = 0.001  # 1e-4 in openai

UPDATE_GLOBAL_ITER = 20

GAMMA = 0.9

ENTROPY_BETA = 0.01

LAMBDA = 1.0

GAME_NAME = config.GAME_NAME



def run(sess, index, worker_device):
    # env
    env = gym.make(GAME_NAME).unwrapped
    env.reset()
    N_S, N_A = env.observation_space.shape, 4  # env.action_space.n


    # model
    optimizer = tf.train.AdamOptimizer(LR)

    with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
        with tf.variable_scope("global"):
            g_model = A3CModel(N_S, N_A, optimizer, ENTROPY_BETA)


    with tf.device(worker_device):
        with tf.variable_scope("local"):
            model = A3CModel(N_S, N_A, optimizer, ENTROPY_BETA, g_model.var_list)



    # worker
    worker = A3CWorker(sess, model, env, GAMMA, LAMBDA)
    worker.train()