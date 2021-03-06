"""
This is the tensorflow worker to run a3c, core code is run() funtion
"""

import os
import tensorflow as tf
from tensorflow.python.client import device_lib
import gym
import gymgame
import logging
import option
import config as game_config

from model import Model
from agent import TrainAgent, PlayAgent



GAME_NAME = game_config.GAME_NAME

LR = 1e-4  # 1e-4 in openai
UPDATE_STEPS = 20
GAMMA = 0.9
ENTROPY_BETA = 0.01
LAMBDA = 1.0


def run(server, args):
    """This is the main entry of tensorflow worker process"""
    # device and config
    if args.backend == 'cpu':
        worker_device = "/job:worker/task:{}/{}:*".format(args.index, args.backend)
    else:
        gpus = [d for d in device_lib.list_local_devices() if d.device_type == 'GPU']
        worker_device = "/job:worker/task:{}/{}:{}".format(args.index, args.backend, args.index % len(gpus))

    config = tf.ConfigProto(device_filters=["/job:ps", worker_device])

    # summary writer
    summary_writer = tf.summary.FileWriter(os.path.join(args.log_dir, 'worker_{}'.format(args.index)))

    # env
    env = gym.make(GAME_NAME)
    env.env.init_params('127.0.0.1', 5000, 1002, True, dashboard=game_config.Renderer)
    N_S = env.env.input_state_shape
    N_A = env.env.num_action

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

    # agent
    if args.mode == 'play':
        worker = PlayAgent(model, env, global_step, summary_writer, args.render)
    else:
        worker = TrainAgent(model, env, global_step, summary_writer, args.render, GAMMA, LAMBDA, UPDATE_STEPS)

    # saver
    saver = FastSaver(variables_to_save)

    # initializer
    init_all_op = tf.global_variables_initializer()
    def init_fn(ses):
        logging.info("Initializing all parameters.")
        ses.run(init_all_op)

    sv = tf.train.Supervisor(is_chief=True,
                             logdir=args.log_dir,
                             saver=saver,
                             summary_op=None,
                             init_op=tf.variables_initializer(variables_to_save),
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=global_step,
                             save_model_secs=1e+5,
                             save_summaries_secs=args.save_summaries_secs)


    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        logging.info("Starting training at step=%d" % sess.run(global_step))
        worker(sess)


class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)
