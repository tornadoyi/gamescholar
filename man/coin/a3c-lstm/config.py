import numpy as np
from gymgame.engine import extension, Vector2
from gymgame.tinyrpg.man import config, Serializer


config.GAME_PARAMS.fps = 20

config.GAME_PARAMS.max_steps = 100

config.NUM_BULLET = 0

config.NUM_COIN = 5


GAME_NAME = config.GAME_NAME




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


    def _serialize_character_flat(self, k, char):
        attr = char.attribute
        k.do(attr.position, None, lambda v, norm: v / norm.game.map.bounds.max)