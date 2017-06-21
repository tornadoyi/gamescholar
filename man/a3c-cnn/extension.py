from gym import spaces
from gymgame.engine import Vector2, Vector3, extension
from gymgame.tinyrpg import man

config = man.config


# env constant
config.NUM_NPC = 1
config.MAP_SIZE = Vector2(30, 30)



@extension(man.EnvironmentGym)
class EnvironmentExtension():

    @property
    def player(self): return self.game.players[0].attribute.hp

    @property
    def npcs(self): return self.game.npcs

    def _reset(self):
        man.EnvironmentGym._reset(self)
        self.npc_count = len(self.npcs)
        self.player_hp = self.player.attribute.hp


    # action space
    # use move toward (x,y)   -- x~[0,1] y ~[0,1]
    # special case: use (0, 0) as idle
    def _init_action_space(self): return spaces.Box(low=-1, high=1, shape=(2,))

    def _reward(self):
        def _sync():
            self.npc_count = len(self.npcs)
            self.player_hp = self.player.attribute.hp

        sub_hp = self.player.attribute.hp - self.player_hp
        if sub_hp > 0:
            _sync()
            return -sub_hp / 10.0

        sub_npc = len(self.npcs) - self.npc_count
        if sub_npc > 0:
            _sync()
            return sub_npc / 10.0

        return 0.0


@extension(man.Serializer)
class SerializerExtension():
    def serialize_state(self, game):
        pass

    def _deserialize_action(self, data):
        # data 1x3 (x, y, speed)
        direct = Vector2(data[0], data[1])
        speed = data[3]
        actions = [(config.PLAYER_IDS[0], config.Action.move_toward, direct, speed)]
        return actions




# state transform
_state_grid_shape = Vector3(
    int(config.MAP_SIZE.x/ config.PLAYER_RADIUS/ 2),
    int(config.MAP_SIZE.y / config.PLAYER_RADIUS / 2),
    3*2  # (count, direction, speed)  of all players/npcs
)

