import tensorflow as tf




class ActorCritic(object):
    def __init__(self,
                 sess,
                 pi, optimizer,
                 global_pi=None,
                 entropy_beta=0.01):

        self.sess = sess
        self.optimizer = optimizer
        self.global_pi = global_pi
        self.entropy_beta = entropy_beta
        self.pi = pi

        self.s = self.pi.x
        self.a = tf.placeholder(tf.float32, [None, pi.action_shape], name="actor")
        self.adv = tf.placeholder(tf.float32, [None], name="adv")
        self.r = tf.placeholder(tf.float32, [None], name="r")
        self.guide = tf.placeholder(tf.float32, [None, pi.action_shape], name="guide")


        log_prob_tf = tf.nn.log_softmax(pi.logits)
        prob_tf = tf.nn.softmax(pi.logits)

        # the "policy gradients" loss:  its derivative is precisely the policy gradient
        # notice that self.ac is a placeholder that is provided externally.
        # adv will contain the advantages, as calculated in process_rollout
        pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.a, [1]) * self.adv)

        # loss of value function
        vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
        entropy = - tf.reduce_sum(prob_tf * log_prob_tf)


        self.loss = pi_loss + 0.5 * vf_loss - entropy * self.entropy_beta


        grads = tf.gradients(self.loss, pi.var_list)

        grads, _ = tf.clip_by_global_norm(grads, 40.0)

        # copy weights from the parameter server to the local model
        self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.global_pi.var_list)])

        grads_and_vars = list(zip(grads, self.global_pi.var_list))

        # each worker has a different set of adam optimizer parameters
        self.train_op = optimizer.apply_gradients(grads_and_vars)



    def choose_action(self, s, *features): return self.pi.act(self.sess, s, *features)


    def predict_value(self, s, *features): return self.pi.value(self.sess, s, *features)


    def learn(self, feed_dict, fetches=[]):  # run by a local
        return self.sess.run([self.train_op] + fetches, feed_dict)[1:]  # local grads applies to global net


    def pull(self): self.sess.run(self.sync)




