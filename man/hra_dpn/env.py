import copy
import random
import numpy as np
from gym import spaces
from gymgame.engine import Vector2, extension
from gymgame.tinyrpg import man

config = man.config
Attr = config.Attr


config.MAP_SIZE = Vector2(10, 10)

config.GRID_SIZE = Vector2(10, 10)

config.NUM_BULLET = 0

config.NUM_COIN = 5

config.COIN_REVIVE = False


COIN_POOL_SIZE = config.NUM_COIN * 2


@extension(man.Serializer)
class SerializerExtension():

    DIRECTS = [Vector2.up, Vector2.right, Vector2.down, Vector2.left]

    def _serialize_state(self, k, game):
        grid = self._serialize_map(k, game.map)
        s_coins = np.zeros(config.NUM_COIN)
        for c in game.map.coins:
            id = int(c.attribute.id.split('-')[-1])
            s_coins[id] = 1

        return np.hstack([grid.ravel(), s_coins])


    def _deserialize_action(self, data):
        i = np.argmax(data)
        direct = SerializerExtension.DIRECTS[i]
        actions = [('player-0', config.Action.move_toward, direct, None)]
        return actions


@extension(man.Data)
class DataExtension():

    '''
    COIN_INIT_POSITIONS = [Vector2(7, 4),
                           Vector2(3, 9),
                           Vector2(9, 9),
                           Vector2(8, 0),
                           Vector2(9, 5),
                           Vector2(3, 1),
                           Vector2(8, 3),
                           Vector2(0, 1),
                           Vector2(4, 3),
                           Vector2(4, 1)]
    '''
    COIN_INIT_POSITIONS = [Vector2(*v) for v in np.random.randint(0, COIN_POOL_SIZE, COIN_POOL_SIZE*2).reshape(COIN_POOL_SIZE, -1)]

    def _create_coin_infos(self):
        coins = []
        pos_indexes = random.sample(range(10), 5)
        for i in range(config.NUM_COIN):
            coin = copy.deepcopy(config.BASE_COIN)
            coin.id = coin.id.format(i)
            coin.position = DataExtension.COIN_INIT_POSITIONS[pos_indexes[i]]
            coins.append(coin)
        return coins


@extension(man.EnvironmentGym)
class EnvExtension():
    pass
    '''
    def _reward(self):
        map = self.game.map
        player = map.players[0]
        coins = map.coins
        s_coins = np.zeros(config.NUM_COIN, dtype=float)
        for c in coins:
            id = int(c.attribute.id.split('-')[-1])
            d = c.attribute.position.distance(player.attribute.position)
            r = 1 if d == 0 else 1 / d
            s_coins[id] = r

        return s_coins
    '''



def run(render=False):
    import gym
    import time
    env = gym.make(config.GAME_NAME)
    env.reset()
    t = time.time()
    for i in range(5000):#while True:
        if env.env.terminal: env.reset()
        env.step([1, 0, 0, 0])
        if render: env.render()


if __name__ == '__main__':
    run()