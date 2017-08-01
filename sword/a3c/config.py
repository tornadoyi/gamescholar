import numpy as np
from easydict import EasyDict as edict
from gymgame.engine import extension, Vector2
from gymgame.tinyrpg.sword import config, Serializer, EnvironmentGym
from gymgame.tinyrpg.framework import Skill, Damage, SingleEmitter
from gym import spaces



GAME_NAME = config.GAME_NAME

config.MAP_SIZE = Vector2(10, 10)

config.GAME_PARAMS.fps = 24

config.NUM_PLAYERS = 1

config.NUM_NPC = 1

config.PLAYER_INIT_RADIUS = (0.0, 1.0)

config.NPC_INIT_RADIUS = (0.0, 1.0)

config.NPC_SKILL_COUNT = 1


config.SKILL_DICT = {
    'normal_attack' : Skill(
        id = 'normal_attack',
        cast_time = 0.3,
        mp_cost = 0,
        target_required = True,
        target_relation = config.Relation.enemy,
        cast_distance = 0.5,
        target_factors = [Damage(30.0, config.Relation.enemy)]
    ),

    'normal_shoot' : Skill(
        id = 'normal_shoot',
        cast_time = 0.3,
        mp_cost = 0,
        bullet_emitter = SingleEmitter(
            speed=1.0,
            penetration=1.0,
            max_range=config.MAP_SIZE.x * 0.8,
            radius=0.1,
            factors=[Damage(10.0, config.Relation.enemy)])
    ),

    'puncture_shoot' : Skill(
        id = 'normal_shoot',
        cast_time = 0.3,
        mp_cost = 0,
        bullet_emitter = SingleEmitter(
            speed=1.0,
            penetration=np.Inf,
            max_range=config.MAP_SIZE.x * 0.8,
            radius=0.1,
            factors=[Damage(10.0, config.Relation.enemy)])
    ),
}

config.PLAYER_SKILL_DICT = {'normal_shoot': config.SKILL_DICT['normal_shoot']}

config.NPC_SKILL_DICT = {'normal_attack': config.SKILL_DICT['normal_attack']}


config.BASE_PLAYER = edict(
    id = "player-{0}",
    position = Vector2(0, 0),
    direct = Vector2(0, 0),
    speed = 0.3 * config.GAME_PARAMS.fps,
    radius = 0.5,
    max_hp = 100.0,
    camp = config.Camp[0],
    skills=list(config.PLAYER_SKILL_DICT.values())
)


config.BASE_NPC = edict(
    id = "npc-{0}",
    position = Vector2(0, 0),
    direct = Vector2(0, 0),
    speed = 0.1 * config.GAME_PARAMS.fps,
    radius = 0.5,
    max_hp = 100.0,
    camp = config.Camp[1],
    skills=[]
)





@extension(EnvironmentGym)
class EnvExtension():
    def _init_action_space(self): return spaces.Discrete(4)


@extension(Serializer)
class SerializerExtension():

    DIRECTS = [Vector2.up, Vector2.right, Vector2.down, Vector2.left]


    def _deserialize_action(self, data):

        if np.ndim(data) == 0: direct = SerializerExtension.DIRECTS[data]
        elif len(data) == len(SerializerExtension.DIRECTS): direct = SerializerExtension.DIRECTS[np.argmax(data)]
        else: direct = Vector2(*data)

        actions = [('player-0', config.Action.move_toward, direct, None)]
        return actions



    def _serialize_map(self, k, map):
        s_npcs = k.do_object(map.npcs, self._serialize_character)
        return s_npcs



    def _serialize_character(self, k, char):

        def norm_position(v, norm):
            map = norm.game.map
            player = map.players[0]
            return (v - player.attribute.position) / map.bounds.max


        attr = char.attribute
        k.do(attr.position, None, norm_position)
        k.do(attr.hp, None, k.n_div_tag, config.Attr.hp)