import os
import tensorflow as tf
import gym
import logging

from model import Model
from worker import TrainWorker
import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


LR = 0.001  # 1e-4 in openai

UPDATE_STEPS = 20

GAMMA = 0.9

ENTROPY_BETA = 0.01

LAMBDA = 1.0

GAME_NAME = 'CartPole-v0'#config.GAME_NAME #'CartPole-v0'



class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


def run(server, args):
    # device and config
    worker_device = "/job:worker/task:{}/{}:*".format(args.index, args.backend)
    config = tf.ConfigProto(device_filters=["/job:ps", worker_device])

    # summary writer
    # summary writer
    summary_writer = tf.summary.FileWriter(os.path.join(args.log_dir, 'worker_{}'.format(args.index)))

    # env
    env = gym.make(GAME_NAME).unwrapped
    env.reset()
    N_S, N_A = env.observation_space.shape, env.action_space.n


    # model
    optimizer = tf.train.AdamOptimizer(LR)
    with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
        with tf.variable_scope("global"):
            g_model = Model(N_S, N_A, optimizer, ENTROPY_BETA)
            global_step = tf.Variable(0.0, trainable=False, dtype=tf.float64)
            variables_to_save = g_model.var_list + [global_step]


    with tf.device(worker_device):
        with tf.variable_scope("local"):
            model = Model(N_S, N_A, optimizer, ENTROPY_BETA, g_model.model_vars)



    # worker
    worker = TrainWorker(model, env, global_step, summary_writer, GAMMA, LAMBDA, UPDATE_STEPS)

    # saver
    saver = FastSaver(variables_to_save)

    # initializer
    init_all_op = tf.global_variables_initializer()
    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    sv = tf.train.Supervisor(is_chief=(args.index == 0),
                             logdir=args.log_dir,
                             saver=saver,
                             summary_op=None,
                             init_op=tf.variables_initializer(variables_to_save),
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)


    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        logger.info("Starting training at step=%d", sess.run(global_step))
        worker(sess)