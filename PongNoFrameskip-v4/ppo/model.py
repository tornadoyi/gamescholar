import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd

class Model(object):
    recurrent = False

    def __init__(self, ob_space, ac_space, ):
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.pdtype = make_pdtype(ac_space)

        self.ob = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape), name="ob")
        self.stochastic = tf.placeholder(dtype=tf.bool, shape=())


        # create fc network
        self._create_network()

        # get all train vars
        self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name)


    def _create_network(self): raise NotImplementedError('_create_network should be implemented')


    def _create_logit_value(self, action_layer, value_layer, gaussian_fixed_var=False):
        # actor
        if gaussian_fixed_var and isinstance(self.ac_space, gym.spaces.Box):
            mean = U.dense(action_layer, self.pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(action_layer, self.pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = self.pdtype.pdfromflat(pdparam)
        self.ac = U.switch(self.stochastic, self.pd.sample(), self.pd.mode())

        # critic
        self.vpred = U.dense(value_layer, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]


    def act(self, ob, stochastic):
        ac1, vpred1 = tf.get_default_session().run([self.ac, self.vpred],
                                     {self.ob: ob[None], self.stochastic: stochastic})
        return ac1[0], vpred1[0]

    def update_ob_norm(self, ob): pass




class MLPModel(Model):
    def __init__(self, ob_space, ac_space, ob_filter=True, gaussian_fixed_var=True):
        self.ob_filter = ob_filter
        self.gaussian_fixed_var = gaussian_fixed_var
        super(MLPModel, self).__init__(ob_space, ac_space)


    def _create_network(self):
        x = self.ob

        # create ob filter
        if self.ob_filter:
            self.ob_rms = RunningMeanStd(shape=self.ob_space.shape)
            x = tf.clip_by_value((self.ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)


        # actor
        l = x
        l = tf.nn.tanh(U.dense(l, 32, "a_1", weight_init=U.normc_initializer(1.0)))
        l = tf.nn.tanh(U.dense(l, 32, "a_2", weight_init=U.normc_initializer(1.0)))
        action_layer = l

        # critic
        l = x
        l = tf.nn.tanh(U.dense(l, 32, "c_1", weight_init=U.normc_initializer(1.0)))
        l = tf.nn.tanh(U.dense(l, 32, "c_2", weight_init=U.normc_initializer(1.0)))
        value_layer = l

        self._create_logit_value(action_layer, value_layer, self.gaussian_fixed_var)


    def update_ob_norm(self, ob):
        if not hasattr(self, 'ob_rms'): return
        self.ob_rms.update(ob)


class CNNModel(Model):
    def __init__(self, ob_space, ac_space, kind='large'):
        self.kind = kind
        
        super(CNNModel, self).__init__(ob_space, ac_space)


    def _create_network(self):
        l = self.ob / 255.0
        if self.kind == 'small':  # from A3C paper
            l = tf.nn.relu(U.conv2d(l, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            l = tf.nn.relu(U.conv2d(l, 32, "l2", [4, 4], [2, 2], pad="VALID"))
            l = U.flattenallbut0(l)
            l = tf.nn.relu(U.dense(l, 256, 'lin', U.normc_initializer(1.0)))
        elif self.kind == 'large':  # Nature DQN
            l = tf.nn.relu(U.conv2d(l, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            l = tf.nn.relu(U.conv2d(l, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            l = tf.nn.relu(U.conv2d(l, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            l = U.flattenallbut0(l)
            l = tf.nn.relu(U.dense(l, 512, 'lin', U.normc_initializer(1.0)))
        else:
            raise NotImplementedError

        self._create_logit_value(l, l)