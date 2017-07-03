import numpy as np
import gym
from dqn import DeepQNetwork
import config
import tensorflow as tf


W = [0.25, 0.25, 0.25, 0.25]


env = gym.make(config.config.GAME_NAME)
_s = env.reset()
N_S = _s.shape[0]  # env.observation_space.shape[0]
N_A = 4#env.action_space.n

sess = tf.Session()


def run(render):
    net = DeepQNetwork(sess,
                       N_A, N_S,
                       learning_rate=0.01,
                       reward_decay=0.9,
                       e_greedy=0.9,
                       replace_target_iter=200,
                       memory_size=2000,
                       scope='dqn_{0}'.format(0),
                       # output_graph=True
                       )

    sess.run(tf.global_variables_initializer())


    step = 0
    for episode in range(300):
        # initial observation
        s = env.reset()

        while True:
            # fresh env
            if render: env.render()

            # RL choose action based on observation
            a, q = net.choose_action(s)


            # RL take action and get next observation and reward
            s_, r, d, _ = env.step(a)

            print('rewards: {0}'.format(r))

            net.store_transition(s, a, r, s_)


            if (step > 200) and (step % 5 == 0):
                net.learn()

            # swap observation
            s = s_

            # break while loop when end of this episode
            if d:
                break
            step += 1


if __name__ == '__main__':
    run(False)