from gym import spaces
from gymgame.engine import Vector, Vector2, Vector3
from gymgame.tinyrpg.man import EnvironmentGym, Game, config

# env constant
config.NUM_NPC = 1
config.MAP_SIZE = Vector2(30, 30)

# action space
# use move toward (x,y)   -- x~[0,1] y ~[0,1]
# special case: use (0, 0) as idle
_action_space = spaces.Box(low=-1, high=1, shape=(2,))


# reward
def _reward(self):
    return 1

# state transform
_state_grid_shape = Vector()

