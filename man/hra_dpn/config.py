import copy
import random
import numpy as np
from gym import spaces
from gymgame.engine import Vector2, extension
from gymgame.tinyrpg import man
from easydict import EasyDict as edict

config = man.config
Attr = config.Attr

config.GAME_PARAMS.max_steps = 30 #300

config.GAME_PARAMS.fps = 20

config.MAP_SIZE = Vector2(10, 10)

config.GRID_SIZE = Vector2(10, 10)

config.NUM_BULLET = 0

config.NUM_COIN = 5

config.COIN_REVIVE = False

config.PLAYER_INIT_RADIUS = (0.0, 0.0)


COIN_POOL_SIZE = config.NUM_COIN * 2

config.BASE_PLAYER = edict(
    id = "player-{0}",
    position = Vector2(0, 0),
    direct = Vector2(0, 0),
    speed = 20.0,
    radius = 1,
    max_hp = 1,
)


config.BASE_COIN = edict(
    id = "coin-{0}",
    position = Vector2(0, 0),
    direct = Vector2(0, 0),
    radius = 0.4,
    max_hp = 1,
)


@extension(man.Serializer)
class SerializerExtension():

    DIRECTS = [Vector2.up, Vector2.right, Vector2.down, Vector2.left]

    def _serialize_state(self, k, game):
        grid = self._serialize_map(k, game.map)
        player_grid = grid[:, :, 0]
        s_coins = np.zeros(COIN_POOL_SIZE)
        for c in game.map.coins:
            id = int(c.attribute.id.split('-')[-1])
            s_coins[id] = 1

        return np.hstack([player_grid.ravel(), s_coins])


    def _deserialize_action(self, data):
        direct = SerializerExtension.DIRECTS[data]
        actions = [('player-0', config.Action.move_toward, direct, None)]
        return actions


@extension(man.Data)
class DataExtension():

    COIN_INIT_POSITIONS = [Vector2(7, 4),
                           Vector2(3, 7),
                           Vector2(6, 6),
                           Vector2(7, 2),
                           Vector2(2, 8),
                           Vector2(3, 1),
                           Vector2(4, 3),
                           Vector2(3, 5),
                           Vector2(7, 6),
                           Vector2(2, 1)]

    #COIN_INIT_POSITIONS = [Vector2(*v) for v in np.random.randint(0, COIN_POOL_SIZE, COIN_POOL_SIZE*2).reshape(COIN_POOL_SIZE, -1)]

    def _create_coin_infos(self):
        coins = []
        pos_indexes = random.sample(range(10), 5)
        for i in range(config.NUM_COIN):
            pos_idx = pos_indexes[i]
            tag = '{0}-{1}'.format(i, pos_idx)
            coin = copy.deepcopy(config.BASE_COIN)
            coin.id = coin.id.format(tag)
            coin.position = DataExtension.COIN_INIT_POSITIONS[pos_idx]
            coins.append(coin)
        return coins



@extension(man.EnvironmentGym)
class EnvExtension():
    def _reward(self):
        players = self.game.map.players
        hits = np.array([player.step_hits for player in players], dtype=float)
        coins = np.array([player.step_coins for player in players], dtype=float)
        r = coins - hits
        r[r == 0] = -1 #/ 10
        #r = r / 10
        return r[0] if len(r) == 1 else r

    '''
    def _reward(self):
        map = self.game.map
        player = map.players[0]
        coins = map.coins
        s_coins = np.zeros(config.NUM_COIN, dtype=float)
        for c in coins:
            id = int(c.attribute.id.split('-')[-2])
            d = c.attribute.position.distance(player.attribute.position)
            r = 1 if d == 0 else np.min([1, 1 / d])
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