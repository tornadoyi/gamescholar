"""
Here's the main entry of the whole algorithm
"""

import os
import sys
import copy
import option


def cmd_kill_session(session): return 'tmux kill-session -t {} '.format(session)

def cmd_new_session(session, window=None):
    cmd = 'tmux new-session -s {} '.format(session)
    if window is not None: cmd += '-n {} '.format(window)
    cmd += '-d bash '
    return cmd

def cmd_new_window(session, window):
    cmd_win = 'tmux new-window -t {} -n {} bash'.format(session, window)
    cmd_path = cmd_send_keys(session, window, 'cd {}'.format(os.getcwd()))
    return '{} & {}'.format(cmd_win, cmd_path)

def cmd_send_keys(session, window, cmd, enter=True):
    cmd = "tmux send-keys -t {}:{} '{}' ".format(session, window, cmd)
    if enter: cmd += 'Enter '
    return cmd


def cmd_execute_python(session, window, file, arg_dict):
    args = ''
    for k,v in arg_dict.items(): args += '{}={} '.format(k, v)
    cmd = '{} {} {}'.format(sys.executable, file, args)
    return cmd_send_keys(session, window, cmd)


def main():
    # clear start log
    if os.path.exists(option.START_LOG_DIR): os.remove(option.START_LOG_DIR)

    # parse args
    args, arg_dict = option.args, option.arg_dict
    session_name = args.session_name

    # recreate log dir
    if not os.path.exists(args.log_dir): os.mkdir(args.log_dir)

    # create session and kill olds
    cmds = []
    cmds.append(cmd_kill_session(session_name))
    cmds.append(cmd_new_session(session_name))

    # ps
    arg = copy.deepcopy(arg_dict)
    arg['--job-name'] = 'ps'
    cmds.append(cmd_new_window(session_name, 'ps'))
    cmds.append(cmd_execute_python(session_name, 'ps', 'process.py', arg))
    cmds.append('sleep 3')

    # workers
    for i in range(args.num_workers):
        win = 'worker_{}'.format(i)
        arg = copy.deepcopy(arg_dict)
        arg['--index'] = i
        arg['--job-name'] = 'worker'

        cmds.append(cmd_new_window(session_name, win))
        cmds.append(cmd_execute_python(session_name, win, 'process.py', arg))

    # tensorboard
    cmds.append(cmd_new_window(session_name, 'tb'))
    cmds.append(cmd_send_keys(session_name, 'tb', 'tensorboard --logdir {}'.format(args.log_dir)))

    # htop
    cmds.append(cmd_new_window(session_name, 'htop'))
    cmds.append(cmd_send_keys(session_name, 'htop', 'htop'))

    # run commands
    for cmd in cmds: os.system(cmd)


if __name__ == "__main__":
    main()
