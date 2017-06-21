from gymgame.engine import Vector2
from gymgame.tinyrpg.man import EnvironmentGym, Game, config

# env constant
config.NUM_NPC = 1
config.MAP_SIZE = Vector2(30, 30)


# reward
def _reward(self):
    return 1

# state transform
_state_grid_size = Vector2(int(config.MAP_SIZE.x/ config.PLAYER_RADIUS/ 2),
                           int(config.MAP_SIZE.y / config.PLAYER_RADIUS / 2))

