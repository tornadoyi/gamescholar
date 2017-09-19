import time
from collections import deque
import tensorflow as tf, numpy as np
from mpi4py import MPI
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments


class PPO(object):
    def __init__(self, ob_space, ac_space, model_func,
                 clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
                 adam_epsilon=1e-5,
                 ):

        with tf.variable_scope('pi'):
            self.pi = pi = model_func(ob_space, ac_space)

        with tf.variable_scope('pi_old'):
            self.pi_old = pi_old = model_func(ob_space, ac_space)

        self.adv = tf.placeholder(dtype=tf.float32, shape=[None], name='adv')  # Target advantage function (if applicable)
        self.ret = tf.placeholder(dtype=tf.float32, shape=[None], name='ret')  # Empirical return

        self.lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])  # learning rate multiplier, updated with schedule
        clip_param = clip_param * self.lrmult  # Annealed cliping parameter epislon

        self.ac = ac = pi.pdtype.sample_placeholder([None])

        kloldnew = pi_old.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = U.mean(kloldnew)
        meanent = U.mean(ent)
        pol_entpen = (-entcoeff) * meanent

        ratio = tf.exp(pi.pd.logp(ac) - pi_old.pd.logp(ac))  # pnew / pold
        surr1 = ratio * self.adv  # surrogate from conservative policy iteration
        surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * self.adv  #
        pol_surr = - U.mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
        vf_loss = U.mean(tf.square(pi.vpred - self.ret))
        self.total_loss = pol_surr + pol_entpen + vf_loss

        # gradients
        self.grads = tf.gradients(self.total_loss, pi.train_vars)
        self.flat_grads = U.flatgrad(self.total_loss, pi.train_vars)

        # optimizer
        self.optimizer = MpiAdam(pi.train_vars, epsilon=adam_epsilon)

        # assign new pi to old pi
        self.op_assign_old_eq_new = tf.group(*[tf.assign(oldv, newv) for (oldv, newv) in zipsame(pi_old.global_vars, pi.global_vars)])


        U.initialize()
        self.optimizer.sync()



    def update_old_by_new(self): tf.get_default_session().run(self.op_assign_old_eq_new)


    def learn(self, ob, ac, adv, ret, lrmult, optimize_step_size):
        g = tf.get_default_session().run(
            self.flat_grads,
            {self.pi.ob: ob, self.pi_old.ob: ob, self.ac: ac, self.adv: adv, self.ret: ret, self.lrmult: lrmult}
        )
        self.optimizer.update(g, optimize_step_size * lrmult)


    def act(self, ob, stochastic): return self.pi.act(ob, stochastic)


    def update_ob_norm(self, ob): return self.pi.update_ob_norm(ob)