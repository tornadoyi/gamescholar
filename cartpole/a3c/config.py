import numpy as np
from gymgame.engine import extension, Vector2
from gymgame.tinyrpg.man import config, Serializer, EnvironmentGym
from gym import spaces



config.GAME_PARAMS.max_steps = 100

config.NUM_BULLET = 0

config.NUM_COIN = 5

config.COIN_INIT_RADIUS = (0.99, 1.0)

GAME_NAME = config.GAME_NAME


@extension(EnvironmentGym)
class EnvExtension():
    def _init_action_space(self): return spaces.Discrete(4)


@extension(Serializer)
class SerializerExtension():

    DIRECTS = [Vector2.up, Vector2.right, Vector2.down, Vector2.left]


    def _deserialize_action(self, data):
        # continues
        if len(data) == 4:
            direct = SerializerExtension.DIRECTS[np.argmax(data)]
        else:
            direct = Vector2(*data)

        actions = [('player-0', config.Action.move_toward, direct, None)]
        return actions



    def _serialize_map(self, k, map):
        s_players = k.do_object(map.players, self._serialize_player)
        s_coins = k.do_object(map.coins, self._serialize_npc)
        s_bullets = k.do_object(map.bullets, self._serialize_npc)

        s_coins -= np.tile(s_players, len(map.coins))
        s_bullets -= np.tile(s_players, len(map.bullets))

        if self._grid_size is None:
            return np.hstack([s_coins, s_bullets])

        else:
            bounds = map.bounds
            grid_coins = self._objects_to_grilend(bounds, map.coins, s_coins, self._coin_shape)
            grid_bullets = self._objects_to_grid(bounds, map.bullets, s_bullets, self._bullet_shape)

            assemble = []
            if grid_coins is not None: assemble.append(grid_coins)
            if grid_bullets is not None: assemble.append(grid_bullets)

            return np.concatenate(assemble, axis=2)


    def _serialize_character_flat(self, k, char):
        attr = char.attribute
        k.do(attr.position, None, lambda v, norm: v / norm.game.map.bounds.max)