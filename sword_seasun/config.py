from easydict import EasyDict as edict
from gym_seasun.gym_sw1 import envs
from gym_seasun.framework import extension
from gym_seasun.engine import Vector2, Bounds2
from gym_seasun.engine.geometry import geometry2d as g2d
from gym_seasun.gym_sw1.plot import RealWorldDashboard
import numpy as np

config = envs.config
GAME_NAME = 'SW1-1-VS-N-v0'

# config.CONFIG.model = config.NPC_MODELS.ALL_IN    npcs结构为二维，包括近战类6个不同的野怪和眩晕类1个野怪，ALL_IN为所有野怪一起出现
# cconfig.CONFIG.model = config.NPC_MODELS.RANDOM_GROUP   npcs结构为二维，包括近战类和眩晕类，RANDOM_GROUP为每一类中随机出现一个野怪

# config.CONFIG.model = config.NPC_MODELS.ALL_IN
# config.CONFIG.npcs = [[edict({
#     'tpl_id': 1106,
#     'level': 50,
#     'count': 1,
#     'location': {
#         'position': [1709, 3321],
#         'range': 10
#     }
# })
# ]]


MAP_BOUNDS = Bounds2(Vector2(0, 0), Vector2(80, 80))

NUM_EYES = 30

VIEW_RANGE = 1200

EYE_DYNAMIC_VIEW = VIEW_RANGE * 0.8

EYE_STATIC_VIEW = VIEW_RANGE * 0.2

DIRECTS = [
    Vector2.up,
    Vector2.up + Vector2.right,
    Vector2.right,
    Vector2.right + Vector2.down,
    Vector2.down,
    Vector2.down + Vector2.left,
    Vector2.left,
    Vector2.left + Vector2.up,
]

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


@extension(envs.Env)
class Env():
    # def _trans_state(self, state):
    #     return state
    @property
    def input_state_shape(self): return (NUM_EYES * 5 + 3, )

    @property
    def num_action(self): return 9

    def _post_reset(self):
        self.userdata = edict(player=edict())
        angle = 360.0 / NUM_EYES
        eyes = [Eye(i, (i * angle, (i + 1) * angle)) for i in range(NUM_EYES)]
        self.userdata.player.eyes = eyes

        self._reset_eyes()


    def _cal_reward(self):
        return 0

    def _cal_done(self, state):
        if self.round_count >= 300: return True
        if self.round_count >= 10 and len(self.current_state.npc) == 0: return True
        if self.current_state.player.base.hp < 1.0: return True
        return False



    def _normalization(self, _):
        self._reset_eyes()

        # serialize npc
        eyes = self.userdata.player.eyes
        s = np.zeros(NUM_EYES * 5 + 3)
        for i in range(len(eyes)):
            e = eyes[i]
            st = i * 5
            s[st + 0] = 1.0  # dynamic range
            s[st + 1] = 1.0  # static range
            s[st + 2] = 0.0  # speed x
            s[st + 3] = 0.0  # speed y
            s[st + 4] = 0.0  # hp

            if e.sensed_object is None:
                continue
            elif type(e.sensed_object) == type(MAP_BOUNDS):
                s[st + 1] = e.sensed_range / EYE_STATIC_VIEW
            else:
                o = e.sensed_object
                speed, hp, direct = o.move_speed, o.hp, DIRECTS[self.face_to_2_direct(o.face_to)]
                vx, vy = direct * speed
                s[st + 0] = e.sensed_range / EYE_DYNAMIC_VIEW
                s[st + 2] = vx
                s[st + 3] = vy
                s[st + 4] = hp

        # serialize player
        player = self.current_state.player
        speed, hp, direct = player.base.move_speed, player.base.hp, DIRECTS[self.face_to_2_direct(player.base.face_to)]
        vx, vy = direct * speed
        s[-3 + 0] = vx
        s[-3 + 1] = vy
        s[-3 + 2] = hp

        return s

    def _reset_eyes(self):
        npcs = self.current_state.npc
        player = self.current_state.player
        npc_pos = np.array([(o.x, o.y) for o in npcs]).reshape([-1, 2])
        player_pos = np.array([player.base.x, player.base.y])


        vec = npc_pos - player_pos
        radian = np.arctan2(vec[:, 1], vec[:, 0])
        radian = (radian + 2 * np.pi) % (2 * np.pi)
        eye_indexes = (radian / (2 * np.pi / NUM_EYES)).astype(np.int)

        distance = np.sqrt(np.sum(np.square(vec), axis=1))
        indexes = np.arange(len(distance))

        eyes = self.userdata.player.eyes
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
                point = g2d.raycast(player_pos, e.direct, EYE_STATIC_VIEW, MAP_BOUNDS)

                # see wall
                if point is not None:
                    e.reset(MAP_BOUNDS, point.distance(player_pos))

                # see nothing
                else:
                    e.reset()

    def get_action_mask(self):
        return np.ones([9, ])




class Renderer(RealWorldDashboard):
    def on_create_plot(self):
        self._initialize_eyes()

    def on_update_plot(self):
        self._draw_eyes()


    def _initialize_eyes(self):
        self.rd_detect = self.plt_loc.wedge(
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

        player = self.env.current_state.player
        x, y = player.base.x, player.base.y
        eyes = self.env.userdata.player.eyes
        for e in eyes:
            start_angles.append(np.radians(e.angles[0]))
            end_angles.append(np.radians(e.angles[1]))
            radius.append(e.sensed_range or 0.0)
            xs.append(x)
            ys.append(y)

            if e.sensed_object is None:
                colors.append(None)

            else:
                # npc
                if type(e.sensed_object) != Bounds2:
                    colors.append("red")
                else:
                    colors.append("grey")

        colors[0] = 'red'
        colors[10] = 'green'
        colors[15] = 'blue'
        colors[20] = 'yellow'

        self.rd_detect.data_source.data['fill_color'] = colors
        self.rd_detect.data_source.data['x'] = xs
        self.rd_detect.data_source.data['y'] = ys
        self.rd_detect.data_source.data['radius'] = radius
        self.rd_detect.data_source.data['start_angle'] = start_angles
        self.rd_detect.data_source.data['end_angle'] = end_angles
        self.rd_detect.data_source.data['fill_alpha'] = [0.1] * len(colors)