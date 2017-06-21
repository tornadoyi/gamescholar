from gym import spaces
from gymgame.engine import Vector2, Vector3, extension
from gymgame.tinyrpg import man

config = man.config
Attr = config.Attr

# env constant
config.NUM_NPC = 1
config.MAP_SIZE = Vector2(30, 30)

def alive_object_count(list):
    count = 0
    for o in list:
        if o.attribute.hp < 1e-6: continue
        count += 1
    return count

@extension(man.NPC)
class NPCExtension():
    def _update(self):
        if self.attribute.hp < 1e-6: return
        self.move_toward(self.attribute.direct)
        if self.attribute.hp < 1e-6 or not self._map.in_bounds(self.attribute.position):
            self.attribute.hp = 0
            self.attribute.position = config.MAP_SIZE
            self.attribute.direct = Vector2.zero


@extension(man.game)
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
        super(man.EnvironmentGym, self)._reset()
        self.npc_count = alive_object_count(self.game.map.npcs)
        self.player_hp = self.player.attribute.hp


    # action space
    # use move toward (x,y)   -- x~[0,1] y ~[0,1]
    # special case: use (0, 0) as idle
    def _init_action_space(self): return spaces.Box(low=-1, high=1, shape=(2,))

    def _reward(self):
        def _sync():
            self.npc_count = alive_object_count(self.game.map.npcs)
            self.player_hp = self.player.attribute.hp

        sub_hp = self.player_hp - self.player.attribute.hp
        if sub_hp > 0:
            _sync()
            return -sub_hp / 10.0

        sub_npc = self.npc_count - alive_object_count(self.game.map.npcs)
        if sub_npc > 0:
            _sync()
            return sub_npc / 10.0

        return 0.0


@extension(man.Serializer)
class SerializerExtension():

    #def serialize_state(self, game):
        #man.Serializer.serialize_state(self, game)


    def _deserialize_action(self, data):
        # data 1x3 (x, y, speed)
        direct = Vector2(data[0], data[1])
        speed = data[2]
        actions = [(config.PLAYER_IDS[0], config.Action.move_toward, direct, speed)]
        return actions


    '''
    def _select_character(self, k):
        self._select_object(k)
        k.add(Attr.hp, None, k.n_div_tag(Attr.hp))
    '''


def run_game(render=False):
    import gym
    import time
    env = gym.make(config.GAME_NAME)
    env.reset()

    while True:
        if env.env.terminal: env.reset()
        time.sleep(1.0 / 60)
        env.step([-1, 1, 1])
        if render: env.render()


if __name__ == "__main__":
    run_game()