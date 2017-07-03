import gym
import gym_sandbox
from dqn import DeepQNetwork

GAME = 'police-bokeh-server-v0'
env = gym.make(GAME)
_s = env.reset()
N_S = _s.shape[0]  # env.observation_space.shape[0]
N_A = env.action_space.n


RL = DeepQNetwork(N_A, N_S,
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  replace_target_iter=200,
                  memory_size=2000,
                  # output_graph=True
                  )

step = 0
for episode in range(300):
    # initial observation
    observation = env.reset()

    while True:
        # fresh env
        env.render()

        # RL choose action based on observation
        action = RL.choose_action(observation)

        # RL take action and get next observation and reward
        observation_, reward, done = env.step(action)

        RL.store_transition(observation, action, reward, observation_)

        if (step > 200) and (step % 5 == 0):
            RL.learn()

        # swap observation
        observation = observation_

        # break while loop when end of this episode
        if done:
            break
        step += 1

# end of game
print('game over')
env.destroy()

RL.plot_cost()
