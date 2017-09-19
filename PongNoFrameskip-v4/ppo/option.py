import argparse
import logging
import os
from easydict import EasyDict as edict
from mpi4py import MPI

def str2bool(v): return v.lower() in ("yes", "true", "t", "1", True)

# base args
parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('--env', type=str, default="PongNoFrameskip-v4", help="gym env")
parser.add_argument('--log-dir', type=str, default="./log", help="Log directory path")
parser.add_argument('--mode', default='train', type=str, choices=['train', 'play'], help='train or play')
parser.add_argument('--render', type=str2bool, default=False, help="Need Render")
parser.add_argument('--backend', default='cpu', type=str, choices=['cpu', 'gpu'], help='cpu, gpu')
parser.add_argument('--gpu-count', default=1, type=int, help='total gpu count')
args = edict(parser.parse_args().__dict__)


# process index
args.index = MPI.COMM_WORLD.Get_rank()


# log
ALL_LOG_DIR = os.path.join(args.log_dir, 'all.log')
ERROR_LOG_DIR = os.path.join(args.log_dir, 'error.log')

# clear old log
if args.index == 0 and args.mode == 'train':
    if os.path.exists(ALL_LOG_DIR): os.remove(ALL_LOG_DIR)
    if os.path.exists(ERROR_LOG_DIR): os.remove(ERROR_LOG_DIR)

# recreate log dir
if not os.path.exists(args.log_dir): os.mkdir(args.log_dir)

# clear handlers
logger = logging.getLogger()
for hdl in [hdl for hdl in logger.handlers]: logger.removeHandler(hdl)

# set formatter
formatter = logging.Formatter(
    "%(levelname)s -{} %(asctime)s: %(message)s".format('{}_{}'.format(args.mode, args.index)),
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

