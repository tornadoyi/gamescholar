"""
Human interact with gym env
pip install pynput
"""

import time
import gym
import config
from pynput import keyboard
from pynput.keyboard import Key


def listen_to_mouse():
    from pynput import mouse

    def on_move(x, y):
        print('Pointer moved to {0}'.format(
            (x, y)))

    def on_click(x, y, button, pressed):
        print('{0} at {1}'.format(
            'Pressed' if pressed else 'Released',
            (x, y)))
        if not pressed:
            # Stop listener
            return False

    def on_scroll(x, y, dx, dy):
        print('Scrolled {0}'.format(
            (x, y)))

    # Collect events until released
    with mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll) as listener:
        listener.join()


def listen_to_keyboard():
    def on_press(key):
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))
        except AttributeError:
            print('special key {0} pressed'.format(
                key))

    def on_release(key):
        print('{0} released'.format(
            key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    # Collect events until released
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()


def run(render=False):
    env = gym.make(config.GAME_NAME)
    env = env.unwrapped
    env.reset()

    ACTION_KEYS = {
        Key.up: 0,
        Key.right: 2,
        Key.down: 4,
        Key.left: 6,
        Key.space: 8
    }

    def on_press(key):
        if key in ACTION_KEYS:
            env.step((ACTION_KEYS[key], env.game.map.npcs[0]))
            if render: env.render()
            if env.terminal: env.reset()

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == '__main__':
    # listen_to_keyboard()
    # listen_to_mouse()
    run(True)
