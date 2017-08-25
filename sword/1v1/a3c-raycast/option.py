"""
handle cmd line option
"""

import os
import logging
import argparse
from easydict import EasyDict as edict
from multiprocessing import cpu_count

# ==================================== args ==================================== #

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1", True)

def num_worker_type(v):
    v = int(v)
    return v if v > 0 else cpu_count()

# base args
parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('--index', default=0, type=int, help='Task index')
parser.add_argument('--job-name', default="worker", help='worker or ps')
parser.add_argument('--num-workers', default=1, type=num_worker_type, help='Number of workers')
parser.add_argument('--backend', default='cpu', type=str, choices=['cpu', 'gpu'], help='cpu, gpu')
parser.add_argument('--log-dir', type=str, default="./log", help="Log directory path")
parser.add_argument('--worker-path', default='', type=str, help='worker file path for load')


# extension args
parser.add_argument('--mode', default='train', type=str, choices=['train', 'play-online', 'play-offline'], help='train, play-online, play-offline')
parser.add_argument('--render', type=str2bool, default=False, help="Need Render")
parser.add_argument('--auto-save', type=str2bool, default=True, help="save automatically")
parser.add_argument('--save-model-secs', type=int, default=30, help="save model per seconds")
parser.add_argument('--save-summaries-secs', type=int, default=30, help="save summaries per seconds")

args = None


def init(log_tag='default', clean_log=False):
    global args
    args = edict(parser.parse_args().__dict__)
    _init_log(log_tag, clean_log)
    return args



def args_to_argv(args):
    argv = []
    for a in parser._actions:
        if not hasattr(args, a.dest): continue
        v = getattr(args, a.dest)
        argv.append('{}={}'.format(a.option_strings[-1], v))
    return argv


# ==================================== cluster ==================================== #

PS_PORT = 12222

WORKER_START_PORT = 20000

MAX_WORKER_COUNT = 100

def _create_hosts(ip, port, num):
    hosts = []
    for i in range(num):
        hosts.append('{}:{}'.format(ip, port))
        port += 1
    return hosts


cluster = {
    'ps': _create_hosts('127.0.0.1', PS_PORT, 1),
    'worker': _create_hosts('127.0.0.1', WORKER_START_PORT, MAX_WORKER_COUNT)
}




# ==================================== log ==================================== #

ALL_LOG_DIR = None

ERROR_LOG_DIR = None

def _init_log(log_tag, clean_log):
    global ALL_LOG_DIR, ERROR_LOG_DIR

    if not os.path.exists(args.log_dir): os.mkdir(args.log_dir)

    ALL_LOG_DIR = os.path.join(args.log_dir, 'all.log')

    ERROR_LOG_DIR = os.path.join(args.log_dir, 'error.log')

    # clear start log
    if clean_log:
        if os.path.exists(ALL_LOG_DIR): os.remove(ALL_LOG_DIR)
        if os.path.exists(ERROR_LOG_DIR): os.remove(ERROR_LOG_DIR)


    # clear handlers
    logger = logging.getLogger()
    for hdl in [hdl for hdl in logger.handlers]: logger.removeHandler(hdl)


    # set formatter
    formatter = logging.Formatter(
        "%(levelname)s -{} %(asctime)s: %(message)s".format(log_tag),
        '%H:%M:%S'
    )

    # handler
    hdl_all = logging.FileHandler(ALL_LOG_DIR)
    hdl_error = logging.FileHandler(ERROR_LOG_DIR)

    # formater
    hdl_all.setFormatter(formatter)
    hdl_error.setFormatter(formatter)

    # level
    hdl_all.setLevel(logging.INFO)
    hdl_error.setLevel(logging.ERROR)

    # logger
    logging.getLogger().addHandler(hdl_all)
    logging.getLogger().addHandler(hdl_error)

    logging.getLogger().setLevel(logging.INFO)



