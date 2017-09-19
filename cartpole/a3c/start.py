import argparse
import os
import sys
import shutil
from multiprocessing import cpu_count


parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('--num-workers', default=cpu_count(), type=int, help="Number of workers")
parser.add_argument('--backend', default='cpu', type=str, help='cpu, gpu')
parser.add_argument('--worker-path', type=str, help='worker file path for load')
parser.add_argument('--log-dir', type=str, default="./log", help="Log directory path")
parser.add_argument('--session-name', type=str, default="project", help="session name")




def cmd_kill_session(session): return 'tmux kill-session -t {} '.format(session)

def cmd_new_session(session, window=None):
    cmd = 'tmux new-session -s {} '.format(session)
    if window is not None: cmd += '-n {} '.format(window)
    cmd += '-d bash '
    return cmd

def cmd_new_window(session, window): return 'tmux new-window -t {} -n {} bash'.format(session, window)

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
    # parse args
    args = parser.parse_args()
    session_name = args.session_name
    

    # recreate log dir
    if not os.path.exists(args.log_dir): os.mkdir(args.log_dir)



    # create session and kill olds
    cmds = []
    cmds.append(cmd_kill_session(session_name))
    cmds.append(cmd_new_session(session_name))

    # ps
    cmds.append(cmd_new_window(session_name, 'ps'))
    cmds.append(cmd_execute_python(session_name, 'ps', 'process.py',
                                   {'--index': 0,
                                    '--job-name': 'ps',
                                    '--num-workers': args.num_workers,
                                    }))


    # workers
    for i in range(args.num_workers):
        win = 'worker_{}'.format(i)
        cmds.append(cmd_new_window(session_name, win))
        cmds.append(cmd_execute_python(session_name, win, 'process.py',
                                       {'--index': i,
                                        '--job-name': 'worker',
                                        '--num-workers': args.num_workers,
                                        '--backend': args.backend,
                                        '--log-dir': args.log_dir,
                                        '--worker-path': args.worker_path,
                                        }))
        if i == 0: cmds.append('sleep 3s')


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
