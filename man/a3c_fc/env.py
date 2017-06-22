import numpy as np
from gym import spaces

from gymgame.engine import Vector2, Vector3, extension
from gymgame.tinyrpg import man

config = man.config
Attr = config.Attr


# env constant
config.MAP_SIZE = Vector2(30, 30)
config.NUM_NPC = 64


def alive_object_count(list):
    count = 0
    for o in list:
        if o.attribute.hp < 1e-6: continue
        count += 1
    return count


@extension(man.NPC)
class NPCExtension():
    def _update(self):
        self.move_toward(self.attribute.direct)
        if self.attribute.hp < 1e-6 or not self._map.in_bounds(self.attribute.position):
            self._revive()


    def _revive(self):
        player = self._map.players[0]
        position = config.gen_init_position((0.9, 0.99))
        direct = config.gen_npc_direct(position, player.attribute.position)
        self.attribute.position = position
        self.attribute.direct = direct
        self.attribute.hp = self.attribute.max_hp



@extension(man.Game)
class GameExtension():
    def _check_terminal(self):
        player_alive = alive_object_count(self._map.players)
        npc_alive = alive_object_count(self._map.npcs)
        return (player_alive == 0) or (npc_alive == 0)


@extension(man.EnvironmentGym)
class EnvironmentExtension():

    @property
    def player(self): return self.game.map.players[0]

    def _reset(self):
        s = super(man.EnvironmentGym, self)._reset()
        self.npc_count = alive_object_count(self.game.map.npcs)
        self.player_hp = self.player.attribute.hp
        return s

    # action space
    # use move toward (x,y)   -- x~[0,1] y ~[0,1]
    # special case: use (0, 0) as idle
    def _init_action_space(self): return spaces.Box(low=-1, high=1, shape=(2,))

    def _reward(self):
        if self.terminal: return -0.1
        if self.game.steps > 4: return 0.1
        return 0.0


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
    while True:
        if env.env.terminal: env.reset()
        time.sleep(1.0 / 60)
        env.step([-1, 1, 1])
        if render: env.render()


if __name__ == '__main__':
    run()
