"""
This is the tensorflow worker to run a3c, core code is run() funtion
"""

import os
import tensorflow as tf
import gym
import logging
import config as game_config

from model import Model
from agent import TrainAgent, PlayAgent



GAME_NAME = game_config.GAME_NAME

LR = 1e-4  # 1e-4 in openai
UPDATE_STEPS = 20
GAMMA = 0.9
ENTROPY_BETA = 0.01
LAMBDA = 1.0


def create(*args, **kwargs): return A3C(*args, **kwargs)


class A3C(object):
    def __init__(self, server, args):
        self.server = server
        self.args = args

        config = tf.ConfigProto(device_filters=["/job:ps", args.worker_device])

        # summary writer
        summary_writer = tf.summary.FileWriter(os.path.join(args.log_dir, 'worker_{}'.format(args.index)))

        # env
        env = gym.make(GAME_NAME).unwrapped
        N_S = env.observation_space.shape
        N_A = env.action_space.n

        # model
        optimizer = tf.train.AdamOptimizer(LR)
        with tf.device(tf.train.replica_device_setter(1, worker_device=args.worker_device)):
            with tf.variable_scope("global"):
                g_model = Model(N_S, N_A, optimizer, ENTROPY_BETA)
                global_step = tf.Variable(0.0, trainable=False, dtype=tf.float64)
                variables_to_save = g_model.var_list + [global_step]

        with tf.device(args.worker_device):
            with tf.variable_scope("local"):
                model = Model(N_S, N_A, optimizer, ENTROPY_BETA, g_model.model_vars)

        # agent
        if args.mode == 'play':
            worker = PlayAgent(model, env, global_step, summary_writer, args.render)
        else:
            worker = TrainAgent(model, env, global_step, summary_writer, args.render, GAMMA, LAMBDA, UPDATE_STEPS)

        # saver
        saver = FastSaver(variables_to_save)
        save_model_secs = args.save_model_secs if args.auto_save else None
        save_summaries_secs = args.save_summaries_secs if args.auto_save else None

        # initializer
        init_all_op = tf.global_variables_initializer()

        def init_fn(ses):
            logging.info("Initializing all parameters.")
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
                                 save_model_secs=save_model_secs,
                                 save_summaries_secs=save_summaries_secs)


        def loop():
            with sv.managed_session(server.target, config=config) as sess, sess.as_default():
                logging.info("Starting training at step=%d", sess.run(global_step))
                g = worker(sess)
                while True: yield next(g)

        self.generator = loop()


    def __call__(self, *args, **kwargs): next(self.generator)




class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)
