import numpy as np
from gymgame.engine import extension, Vector2
from gymgame.tinyrpg.man import config, Serializer


config.GAME_PARAMS.fps = 1

config.GAME_PARAMS.max_steps = 1000

config.NUM_BULLET = 5

config.NUM_COIN = 0

config.BASE_PLAYER.speed = 0.5

config.BASE_BULLET.speed = 0.3


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
