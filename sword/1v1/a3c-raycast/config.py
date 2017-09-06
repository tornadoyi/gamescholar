import numpy as np
from easydict import EasyDict as edict
from gymgame.engine import extension, Vector2
from gymgame.tinyrpg.sword import config, Serializer, EnvironmentGym, Game
from gymgame.tinyrpg.framework import Skill, Damage, SingleEmitter
from gymgame.tinyrpg.framework.render import Renderer, PlayerRenderer, ModuleRenderer
from gymgame.engine.geometry import geometry2d as g2d
from gym import spaces
from bokeh.plotting import figure

#  game config
GAME_NAME = config.GAME_NAME

config.RENDER_MODE = "notebook"  # you need run `bokeh serve` firstly

config.MAP_SIZE = Vector2(10, 10)

config.GAME_PARAMS.fps = 24

config.GAME_PARAMS.max_steps = 300

config.NUM_PLAYERS = 1

config.NUM_NPC = 1

config.PLAYER_INIT_RADIUS = (0.0, 0.0)

config.NPC_INIT_RADIUS = (1.0, 1.0)

config.NPC_SKILL_COUNT = 1



config.SKILL_DICT = {
    'normal_attack' : Skill(
        id = 'normal_attack',
        cast_time = 0.0,
        mp_cost = 0,
        target_required = True,
        target_relation = config.Relation.enemy,
        cast_distance = 1.0,
        target_factors = [Damage(200.0, config.Relation.enemy)]
    ),

    'normal_shoot' : Skill(
        id = 'normal_shoot',
        cast_time = 0.0,
        mp_cost = 0,
        bullet_emitter = SingleEmitter(
            speed=1.0 * config.GAME_PARAMS.fps,
            penetration=1.0,
            max_range=config.MAP_SIZE.x * 0.8,
            radius=0.1,
            factors=[Damage(2.0, config.Relation.enemy)])
    ),

    'puncture_shoot' : Skill(
        id = 'normal_shoot',
        cast_time = 0.0,
        mp_cost = 0,
        bullet_emitter = SingleEmitter(
            speed=1.0 * config.GAME_PARAMS.fps,
            penetration=np.Inf,
            max_range=config.MAP_SIZE.x * 0.8,
            radius=0.1,
            factors=[Damage(2.0, config.Relation.enemy)])
    ),
}

config.PLAYER_SKILL_LIST = [config.SKILL_DICT['puncture_shoot']]

config.NPC_SKILL_LIST = [config.SKILL_DICT['normal_attack']]


config.BASE_PLAYER = edict(
    id = "player-{0}",
    position = Vector2(0, 0),
    direct = Vector2(0, 0),
    speed = 0.5 * config.GAME_PARAMS.fps,
    radius = 0.5,
    max_hp = 100.0,
    camp = config.Camp[0],
    skills=config.PLAYER_SKILL_LIST
)


config.BASE_NPC = edict(
    id = "npc-{0}",
    position = Vector2(0, 0),
    direct = Vector2(0, 0),
    speed = 0.25 * config.GAME_PARAMS.fps,
    radius = 0.5,
    max_hp = 100.0,
    camp = config.Camp[1],
    skills=config.NPC_SKILL_LIST
)



# algorithm config

NUM_EYES = 30

EYE_DYNAMIC_VIEW = config.BASE_NPC.speed * 20

EYE_STATIC_VIEW = 1.5

DIRECT_MASK_RANGE = 0.8

RISK_RANGE = 5.0

RISK_REWARD_RATIO = 1e-2  #  because player_atk / npc_hp = 2e-2




class Eye(object):
    def __init__(self, index, angles):
        vx = Vector2(1, 0)
        self.index = index
        self.angles = angles
        self.angle = np.mean(self.angles)
        self.direct = vx.rotate(self.angle)
        self.directs = (vx.rotate(self.angles[0]), vx.rotate(self.angles[1]))

        # temp
        self.sensed_object = None
        self.sensed_range = None


    def reset(self, o=None, range=None):
        self.sensed_object = o
        self.sensed_range = range




@extension(Game)
class ExtensionGame():
    def _reset(self):
        super(Game, self)._reset()

        map = self.map
        player, npcs = map.players[0], map.npcs

        # set eyes
        angle =  360.0 / NUM_EYES
        eyes = [Eye(i, (i * angle, (i + 1) * angle)) for i in range(NUM_EYES)]
        player.eyes = eyes  # setattr(player, 'eyes', eyes)



@extension(EnvironmentGym)
class EnvExtension():
    def _init_action_space(self): return spaces.Discrete(9)

    def _reset(self):
        s = super(EnvironmentGym, self)._reset()

        map = self.game.map
        player, npcs = map.players[0], map.npcs

        # record frame hp for reward
        self.max_hp = max([player.attribute.hp] + [o.attribute.hp for o in npcs])
        self.pre_player_hp = player.attribute.hp
        self.pre_npc_hp = sum([o.attribute.hp for o in npcs])

        return s

    def _reward(self):
        map = self.game.map
        player, npcs = map.players[0], map.npcs

        r = 0

        # hp reward
        if player.attribute.hp < 1e-6:
            r = -1
        else:
            sub_player_hp = player.attribute.hp - self.pre_player_hp
            npc_hp = 0 if len(npcs) == 0 else sum([o.attribute.hp for o in npcs])
            sub_npc_hp = npc_hp - self.pre_npc_hp

            self.pre_player_hp = player.attribute.hp
            self.pre_npc_hp = npc_hp

            r = (sub_player_hp - sub_npc_hp) / self.max_hp

            if len(npcs) == 0: r += player.attribute.hp / self.max_hp

        #if r > 0: print(r)

        # risk reward
        risk = 0
        if RISK_RANGE is not None:
            for e in player.eyes:
                if e.sensed_object is None or type(e.sensed_object) == type(map.bounds): continue
                if e.sensed_range > RISK_RANGE: continue
                risk += np.exp(-e.sensed_range + 1)

        r -= risk * RISK_REWARD_RATIO

        return r



@extension(Serializer)
class SerializerExtension():

    DIRECTS = [Vector2.up,
               Vector2.up + Vector2.right,
               Vector2.right,
               Vector2.right + Vector2.down,
               Vector2.down,
               Vector2.down + Vector2.left,
               Vector2.left,
               Vector2.left + Vector2.up,
               ]


    def get_action_mask(self, env):
        directs = SerializerExtension.DIRECTS
        map = env.game.map
        player = map.players[0]
        src_pos = player.attribute.position

        mask_move = [0.0] * len(directs)
        for i in range(len(directs)):
            dst_pos = src_pos + directs[i] * DIRECT_MASK_RANGE
            can_move = map.in_bounds(dst_pos)
            mask_move[i] = 1.0 if can_move else 0.0

        mask_skill = [0.0 if player.skill.busy else 1.0]

        return np.array(mask_move + mask_skill)


    def _deserialize_action(self, data):
        index, target = data
        if index < 8:
            direct = SerializerExtension.DIRECTS[index]
            actions = [('player-0', config.Action.move_toward, direct, None)]

        else:
            skill_index = index - 8
            skill_id = config.BASE_PLAYER.skills[skill_index].id
            actions = [('player-0', config.Action.cast_skill, skill_id, target, None)]

        return actions




    def _serialize_map(self, k, map):
        player, npcs = map.players[0], map.npcs

        npc_pos = np.array([o.attribute.position for o in npcs]).reshape([-1, 2])
        player_pos = player.attribute.position

        vec = npc_pos - player_pos
        radian = np.arctan2(vec[:, 1], vec[:, 0])
        radian = (radian + 2*np.pi) % (2*np.pi)
        eye_indexes = (radian / (2 * np.pi / NUM_EYES)).astype(np.int)

        distance = np.sqrt(np.sum(np.square(vec), axis=1))
        indexes = np.arange(len(distance))
        eyes = player.eyes
        for i in range(len(eyes)):
            e = eyes[i]
            cond = (eye_indexes == i)
            idx = indexes[cond]

            # check dynamic
            if len(idx) > 0:
                index = idx[np.argmax(distance[idx])]

                # see dynamic object
                if distance[index] <= EYE_DYNAMIC_VIEW:
                    e.reset(npcs[index], distance[index])

                # see nothing
                else:
                    e.reset()


            # check wall
            else:
                point = g2d.raycast(player_pos, e.direct, EYE_STATIC_VIEW, map.bounds)

                # see wall
                if point is not None:
                    e.reset(map.bounds, point.distance(player_pos))

                # see nothing
                else:
                    e.reset()


        # serialize npc
        s = np.zeros(NUM_EYES * 5 + 3)
        for i in range(len(eyes)):
            e = eyes[i]
            st = i * 5
            s[st + 0] = 1.0  # dynamic range
            s[st + 1] = 1.0  # static range
            s[st + 2] = 0.0  # speed x
            s[st + 3] = 0.0  # speed y
            s[st + 4] = 0.0  # hp

            if e.sensed_object is None: continue
            elif type(e.sensed_object) == type(map.bounds): s[st + 1] = e.sensed_range / EYE_STATIC_VIEW
            else:
                s_o = k.do_object(e.sensed_object, self._serialize_character)
                direct, speed, hp, _ = np.split(s_o, [2, 3, 4])
                vx, vy = direct * speed
                s[st + 0] = e.sensed_range / EYE_DYNAMIC_VIEW
                s[st + 2] = vx
                s[st + 3] = vy
                s[st + 4] = hp


        # serialize player
        s_o = k.do_object(player, self._serialize_character)
        direct, speed, hp, _ = np.split(s_o, [2, 3, 4])
        vx, vy = direct * speed
        s[-3 + 0] = vx
        s[-3 + 1] = vy
        s[-3 + 2] = hp


        return s




    def _serialize_character(self, k, char):
        attr = char.attribute
        k.do(attr.direct, None, None, None)
        k.do(attr.speed, None, k.n_div_tag, config.Attr.speed)
        k.do(attr.hp, None, k.n_div_tag, config.Attr.hp)






@extension(PlayerRenderer)
class PlayerRendererExtension():
    def initialize(self, *args, **kwargs):
        super(PlayerRenderer, self).initialize(*args, **kwargs)
        self._initialize_eyes()
        return self._initialize_action_mask()


    def __call__(self):
        super(PlayerRenderer, self).__call__()
        self._draw_eyes()
        self._draw_action_mask()



    def _initialize_eyes(self):
        self.rd_detect = self.render_state.map.wedge(
            x=[], y=[],
            radius=[],
            start_angle=[],
            end_angle=[],
            fill_color=[],
            line_color=None,
            fill_alpha=[])


    def _draw_eyes(self):
        colors = []
        radius = []
        start_angles = []
        end_angles = []
        xs, ys = [], []

        for player in self.game.map.players:
            x, y = player.attribute.position
            for e in player.eyes:
                start_angles.append(np.radians(e.angles[0]))
                end_angles.append(np.radians(e.angles[1]))
                #radius.append(e.sensed_range or 0.0)
                radius.append(0.0 if e.sensed_range is None else np.max([e.sensed_range, 1.0]))
                xs.append(x)
                ys.append(y)

                if e.sensed_object is None:
                    colors.append(None)

                else:
                    # npc
                    if type(e.sensed_object) != type(self.game.map.bounds):
                        colors.append("red")
                    else:
                        colors.append("grey")

        self.rd_detect.data_source.data['fill_color'] = colors
        self.rd_detect.data_source.data['x'] = xs
        self.rd_detect.data_source.data['y'] = ys
        self.rd_detect.data_source.data['radius'] = radius
        self.rd_detect.data_source.data['start_angle'] = start_angles
        self.rd_detect.data_source.data['end_angle'] = end_angles
        self.rd_detect.data_source.data['fill_alpha'] = [0.1] * len(colors)



    def _initialize_action_mask(self):
        # create new plot
        plt = figure(plot_width=200, plot_height=200, title='action mask')
        plt.xaxis.visible = False
        plt.xgrid.visible = False
        plt.yaxis.visible = False
        plt.ygrid.visible = False

        # set xs xy
        center = Vector2(1, 1)
        xs, ys = [], []
        for d in SerializerExtension.DIRECTS:
            v = d + center
            xs.append(v.x)
            ys.append(v.y)

        # skill
        xs.append(center.x)
        ys.append(center.y)

        self.action_mask_xs = xs
        self.action_mask_ys = ys

        self._rd_mask = plt.rect(
            self.action_mask_xs, self.action_mask_ys,
            0.9,
            0.9,
            fill_alpha=0.6,
            color=["silver"] * 10,
            line_color='silver',
        )

        return plt


    def _draw_action_mask(self):
        a_mask = self.game.env.serializer.get_action_mask(self.game.env)

        self._rd_mask.data_source.data['x'] = self.action_mask_xs
        self._rd_mask.data_source.data['y'] = self.action_mask_ys
        self._rd_mask.data_source.data['fill_color'] = ['green' if v > 0 else 'red' for v in a_mask]