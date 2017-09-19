import numpy as np
from baselines.common import Dataset


class Worker(object):
    def __init__(self, env, ppo, render=False, log_output=False):
        self.env = env
        self.ppo = ppo
        self.render = render
        self.log_output = log_output
        self.steps = 0


    def _reset(self): return self.env.reset()

    def _step(self, a):
        s = self.env.step(a)
        if self.render: self.env.render()
        self.steps += 1
        self.record(s)
        return s


    def record(self, s):
        if not self.log_output: return
        _, r, t, _ = s
        if not hasattr(self, 'epoch_reward'): self.epoch_reward = 0
        self.epoch_reward += r
        if t:
            if not hasattr(self, 'runing_reward'): self.runing_reward = self.epoch_reward
            self.runing_reward = self.runing_reward * 0.9 + self.epoch_reward * 0.1
            print('steps: {}  rewards: {}'.format(self.steps, self.runing_reward))
            self.epoch_reward = 0




class TrainWorker(Worker):
    def __init__(self, env, ppo, render=False, log_output=False,
                 train_data_size=256, optimize_size=None, optimize_epochs=4,
                 gamma=0.99, lambda_=0.95,
                 optimize_step_size = 1e-3,
                 max_steps=np.inf,
                 lr_decay=False):

        super(TrainWorker, self).__init__(env, ppo, render, log_output)
        self.train_data_size = train_data_size
        self.optimize_size = optimize_size
        self.optimize_epochs = optimize_epochs
        self.gamma = gamma
        self.lambda_ = lambda_
        self.optimize_step_size = optimize_step_size
        self.max_steps = max_steps
        self.lr_decay = lr_decay if self.max_steps is not np.inf else False


    def __call__(self, *args, **kwargs):
        data_gen = self._data_generator(self.train_data_size, stochastic=True)

        while True:
            batch = next(data_gen)
            batch = self._add_vtarg_and_adv(batch, self.gamma, self.lambda_)
            obs, acs, advs, td_lam_ret = batch['obs'], batch['acs'], batch['advs'], batch['td_lam_ret']
            advs = (advs - advs.mean()) / advs.std() # standardized advantage function estimate
            d = Dataset(dict(obs=obs, acs=acs, advs=advs, td_lam_ret=td_lam_ret), shuffle=not self.ppo.pi.recurrent)

            # update obs normalization
            self.ppo.update_ob_norm(obs)

            # update old by new
            self.ppo.update_old_by_new()

            # set current learning rate
            cur_lrmult = 1.0 if not self.lr_decay else max(1.0 - float(self.steps) / self.max_steps, 0)

            # learn
            optimize_size = self.optimize_size or obs.shape[0]
            for _ in range(self.optimize_epochs):
                for batch in d.iterate_once(optimize_size):
                    self.ppo.learn(batch["obs"], batch["acs"], batch["advs"], batch["td_lam_ret"], cur_lrmult, self.optimize_step_size)



    def _data_generator(self, batch, stochastic):
        ac = self.env.action_space.sample()  # not used, just so we have the datatype
        new = True  # marks if we're on first timestep of an episode
        ob = self._reset()

        # Initialize history arrays
        obs = np.array([ob for _ in range(batch)])
        rewards = np.zeros(batch, 'float32')
        v_preds = np.zeros(batch, 'float32')
        news = np.zeros(batch, 'int32')
        acs = np.array([ac for _ in range(batch)])
        pre_acs = acs.copy()

        # generate
        t = 0
        while True:
            # predict action
            pre_ac = ac
            ac, v_pred = self.ppo.act(ob, stochastic)


            # return samples
            if t > 0 and t % batch == 0:
                yield {'obs': obs, 'rewards': rewards, 'v_preds': v_preds, 'news': news,
                       'acs': acs, 'pre_acs': pre_acs, 'next_vpred': v_pred * (1 - new)}

            # collect data
            i = t % batch
            obs[i] = ob
            v_preds[i] = v_pred
            news[i] = new
            acs[i] = ac
            pre_acs[i] = pre_ac

            # step
            ob, r, new, _ = self._step(ac)
            rewards[i] = r


            if new:
                ob = self._reset()

            # next step
            t += 1


    def _add_vtarg_and_adv(self, data, gamma, lam):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        """
        new = np.append(data["news"],
                        0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(data["v_preds"], data["next_vpred"])
        T = len(data["rewards"])
        data["advs"] = gaelam = np.empty(T, 'float32')
        rew = data["rewards"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        data["td_lam_ret"] = data["advs"] + data["v_preds"]
        return data



class PlayWorker(Worker):
    pass