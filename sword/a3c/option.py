"""
handle cmd line option
"""

import logging
import argparse
from multiprocessing import cpu_count

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
parser.add_argument('--session-name', type=str, default="project", help="session name")


# extension args
parser.add_argument('--mode', default='train', type=str, choices=['train', 'play'], help='train, play')
parser.add_argument('--render', type=str2bool, default=False, help="Need Render")
parser.add_argument('--auto-save', type=str2bool, default=True, help="save automatically")
parser.add_argument('--save-model-secs', type=int, default=30, help="save model per seconds")
parser.add_argument('--save-summaries-secs', type=int, default=30, help="save summaries per seconds")

args = parser.parse_args()

arg_dict = {}
for a in parser._actions:
    if not hasattr(args, a.dest): continue
    v = getattr(args, a.dest)
    arg_dict[a.option_strings[-1]] = v


# cluster

PS_PORT = 12222

WORKER_PORT = 20000

EXTRA_WORKER_COUNT = 10

def create_hosts(ip, port, num):
    hosts = []
    for i in range(num):
        hosts.append('{}:{}'.format(ip, port))
        port += 1
    return hosts


cluster = {'ps': create_hosts('127.0.0.1', PS_PORT, 1),
           'worker': create_hosts('127.0.0.1', WORKER_PORT, args.num_workers + EXTRA_WORKER_COUNT)}




# log
START_LOG_DIR = 'start.log'
logging.basicConfig(level=logging.INFO,
                    filename=START_LOG_DIR,
                    )

