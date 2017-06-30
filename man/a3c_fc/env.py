import numpy as np
from gym import spaces
from gymgame.engine import Vector2, Vector3, extension
from gymgame.tinyrpg import man

config = man.config
Attr = config.Attr


# env constant
#config.MAP_SIZE = Vector2(20, 20)


@extension(man.Serializer)
class SerializerExtension():

    def _deserialize_action(self, data):
        # data 1x3 (x, y, speed)
        direct = Vector2(data[0], data[1])
        speed = None#data[2]
        actions = [('player-0', config.Action.move_toward, direct, speed)]
        return actions



def run(render=False):
    import gym
    import time
    env = gym.make(config.GAME_NAME)
    env.reset()
    t = time.time()
    for i in range(5000):#while True:
        if env.env.terminal: env.reset()
        time.sleep(1.0 / 60)
        env.step([-1, 1, 1])
        if render: env.render()

    print('cost: {0}'.format(time.time() - t))


if __name__ == '__main__':
    run()
