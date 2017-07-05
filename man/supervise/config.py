import copy
import random
import numpy as np
from gym import spaces
from gymgame.engine import Vector2, extension
from gymgame.tinyrpg import man

config = man.config
Attr = config.Attr

config.GAME_PARAMS.max_steps = 300

config.GAME_PARAMS.fps = 20

config.MAP_SIZE = Vector2(10, 10)

config.NUM_PLAYERS = 1

config.NUM_BULLET = 0

config.NUM_COIN = 10

config.COIN_REVIVE = True

config.PLAYER_INIT_RADIUS = (0.0, 0.0)




@extension(man.Serializer)
class SerializerExtension():

    DIRECTS = [Vector2.up, Vector2.right, Vector2.down, Vector2.left]


    def _serialize_character_flat(self, k, char):
        attr = char.attribute
        #k.do(attr.hp, None, k.n_div_tag, Attr.hp)
        k.do(attr.position, None, lambda v, norm: v / norm.game.map.bounds.max)


    def _deserialize_action(self, data):
        # continues
        if len(data) == 2:
            direct =  Vector2(*data)
        else:
            direct = SerializerExtension.DIRECTS[np.argmax(data)]

        actions = [('player-0', config.Action.move_toward, direct, None)]
        return actions





def run(render=False):
    import gym
    import time
    env = gym.make(config.GAME_NAME)
    env.reset()
    t = time.time()
    for i in range(5000):#while True:
        if env.terminal: env.reset()
        env.step(1)
        if render: env.render()


if __name__ == '__main__':
    run()